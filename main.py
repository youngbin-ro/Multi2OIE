import argparse
import os
import torch
from utils import utils
from utils.utils import SummaryManager
from dataset import load_data
from tqdm import tqdm
from train import train
from extract import extract
from test import do_eval


def main(args):
    utils.set_seed(args.seed)
    model = utils.get_models(
        bert_config=args.bert_config,
        pred_n_labels=args.pred_n_labels,
        arg_n_labels=args.arg_n_labels,
        n_arg_heads=args.n_arg_heads,
        n_arg_layers=args.n_arg_layers,
        lstm_dropout=args.lstm_dropout,
        mh_dropout=args.mh_dropout,
        pred_clf_dropout=args.pred_clf_dropout,
        arg_clf_dropout=args.arg_clf_dropout,
        pos_emb_dim=args.pos_emb_dim,
        use_lstm=args.use_lstm,
        device=args.device)

    trn_loader = load_data(
        data_path=args.trn_data_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        tokenizer_config=args.bert_config)
    dev_loaders = [
        load_data(
            data_path=cur_dev_path,
            batch_size=args.dev_batch_size,
            tokenizer_config=args.bert_config,
            train=False)
        for cur_dev_path in args.dev_data_path]
    args.total_steps = round(len(trn_loader) * args.epochs)
    args.warmup_steps = round(args.total_steps / 10)

    optimizer, scheduler = utils.get_train_modules(
        model=model,
        lr=args.learning_rate,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps)
    model.zero_grad()
    summarizer = SummaryManager(args)
    print("\nTraining Starts\n")

    for epoch in tqdm(range(1, args.epochs + 1), desc='epochs'):
        trn_results = train(
            args, epoch, model, trn_loader, dev_loaders,
            summarizer, optimizer, scheduler)

        # extraction on devset
        dev_iter = zip(args.dev_data_path, args.dev_gold_path, dev_loaders)
        dev_results = list()
        total_sum = 0
        for dev_input, dev_gold, dev_loader in dev_iter:
            dev_name = dev_input.split('/')[-1].replace('.pkl', '')
            output_path = os.path.join(args.save_path, f'epoch{epoch}_dev/end_epoch/{dev_name}')
            extract(args, model, dev_loader, output_path)
            dev_result = do_eval(output_path, dev_gold)
            utils.print_results(f"EPOCH{epoch} EVAL",
                                dev_result, ["F1  ", "PREC", "REC ", "AUC "])
            total_sum += dev_result[0] + dev_result[-1]
            dev_result.append(dev_result[0] + dev_result[-1])
            dev_results += dev_result
        summarizer.save_results([epoch] + trn_results + dev_results + [total_sum])
        model_name = utils.set_model_name(total_sum, epoch)
        torch.save(model.state_dict(), os.path.join(args.save_path, model_name))
    print("\nTraining Ended\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_path', default='./results')
    parser.add_argument('--bert_config', default='bert-base-cased', help='or bert-base-multilingual-cased')
    parser.add_argument('--trn_data_path', default='./datasets/openie4_train.pkl')
    parser.add_argument('--dev_data_path', nargs='+', default=['./datasets/oie2016_dev.pkl', './datasets/carb_dev.pkl'])
    parser.add_argument('--dev_gold_path', nargs='+', default=['./evaluate/OIE2016_dev.txt', './carb/CaRB_dev.tsv'])
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visible_device', default="0")
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--use_lstm', nargs='?', const=True, default=False, type=utils.str2bool)
    parser.add_argument('--binary', nargs='?', const=True, default=False, type=utils.str2bool)

    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lstm_dropout', type=float, default=0.)
    parser.add_argument('--mh_dropout', type=float, default=0.2)
    parser.add_argument('--pred_clf_dropout', type=float, default=0.)
    parser.add_argument('--arg_clf_dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dev_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--n_arg_heads', type=int, default=8)
    parser.add_argument('--n_arg_layers', type=int, default=4)
    parser.add_argument('--pos_emb_dim', type=int, default=64)
    main_args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = main_args.visible_device
    main_args = utils.clean_config(main_args)
    main(main_args)

