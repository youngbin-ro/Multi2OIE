import argparse
import os
import time
import torch
from utils import utils
from dataset import load_data
from extract import extract
from evaluate.evaluate import Benchmark
from evaluate.matcher import Matcher
from evaluate.generalReader import GeneralReader
from carb.carb import Benchmark as CarbBenchmark
from carb.matcher import Matcher as CarbMatcher
from carb.tabReader import TabReader


def get_performance(output_path, gold_path):
    auc, precision, recall, f1 = [None for _ in range(4)]
    if 'evaluate' in gold_path:
        matching_func = Matcher.lexicalMatch
        error_fn = os.path.join(output_path, 'error_idxs.txt')

        evaluator = Benchmark(gold_path)
        reader = GeneralReader()
        reader.read(os.path.join(output_path, 'extraction.txt'))

        (precision, recall, f1), auc = evaluator.compare(
            predicted=reader.oie,
            matchingFunc=matching_func,
            output_fn=os.path.join(output_path, 'pr_curve.txt'),
            error_file=error_fn)
    elif 'carb' in gold_path:
        matching_func = CarbMatcher.binary_linient_tuple_match
        error_fn = os.path.join(output_path, 'error_idxs.txt')

        evaluator = CarbBenchmark(gold_path)
        reader = TabReader()
        reader.read(os.path.join(output_path, 'extraction.txt'))

        auc, (precision, recall, f1) = evaluator.compare(
            predicted=reader.oie,
            matchingFunc=matching_func,
            output_fn=os.path.join(output_path, 'pr_curve.txt'),
            error_file=error_fn)
    return auc, precision, recall, f1


def do_eval(output_path, gold_path):
    auc, prec, rec, f1 = get_performance(output_path, gold_path)
    eval_results = [f1, prec, rec, auc]
    return eval_results


def main(args):
    model = utils.get_models(
        bert_config=args.bert_config,
        pred_n_labels=args.pred_n_labels,
        arg_n_labels=args.arg_n_labels,
        n_arg_heads=args.n_arg_heads,
        n_arg_layers=args.n_arg_layers,
        pos_emb_dim=args.pos_emb_dim,
        use_lstm=args.use_lstm,
        device=args.device)
    model.load_state_dict(torch.load(args.model_path))
    model.zero_grad()
    model.eval()

    loader = load_data(
        data_path=args.test_data_path,
        batch_size=args.batch_size,
        tokenizer_config=args.bert_config,
        train=False)
    start = time.time()
    extract(args, model, loader, args.save_path)
    print("TIME: ", time.time() - start)
    test_results = do_eval(args.save_path, args.test_gold_path)
    utils.print_results("TEST RESULT", test_results, ["F1  ", "PREC", "REC ", "AUC "])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='./results/model.bin')
    parser.add_argument('--save_path', default='./results/carb_test')
    parser.add_argument('--bert_config', default='bert-base-cased')
    parser.add_argument('--test_data_path', default='./datasets/carb_test.pkl')
    parser.add_argument('--test_gold_path', default='./carb/CaRB_test.tsv')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visible_device', default="0")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pos_emb_dim', type=int, default=64)
    parser.add_argument('--n_arg_heads', type=int, default=8)
    parser.add_argument('--n_arg_layers', type=int, default=4)
    parser.add_argument('--use_lstm', nargs='?', const=True, default=False, type=utils.str2bool)
    parser.add_argument('--binary', nargs='?', const=True, default=False, type=utils.str2bool)
    main_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = main_args.visible_device

    main_args.pred_n_labels = 3
    main_args.arg_n_labels = 9
    device = torch.device(main_args.device if torch.cuda.is_available() else 'cpu')
    main_args.device = device
    main(main_args)

