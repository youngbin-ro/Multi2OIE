# Multi^2OIE: <u>Multi</u>lingual Open Information Extraction based on <u>Multi</u>-Head Attention with BERT

> Accepted to the Findings of EMNLP2020 //
> Source code for learning Multi^2OIE for (multilingual) open information extraction.

## Usage

### Prerequisites

- Python 3.7

- CUDA 10.0 or above

  

### Environmental Setup

##### using  'conda' command,
~~~~
# this makes a new conda environment
conda env create -f environment.yml
conda activate multi2oie
~~~~

##### using  'pip' command,
~~~~
pip install -r requirements.txt
~~~~




### Datasets

Original data file (bootstrapped sample from OpenIE4; used in SpanOIE) can be downloaded from [here](https://drive.google.com/file/d/1AEfwbh3BQnsv2VM977cS4tEoldrayKB6/view).
Following download, put the downloaded data in './datasets' and use preprocess.py to convert the data into the format suitable for Multi^2OIE.

~~~~
cd utils
python preprocess.py \
    --mode 'train' \
    --data '../datasets/structured_data.json' \
    --save_path '../datasets/openie4_train.pkl' \
    --bert_config 'bert-base-cased' \
    --max_len 64
~~~~

For multilingual training data, set **'bert_config'** as **'bert-base-multilingual-cased'**. 




### Run the Code

We used TITAN RTX GPU for training, and the use of other GPU can make the final performance different.

##### for training,

~~~~
python main.py [--FLAGS]
~~~~

##### for testing,

~~~~
python test.py [--FLAGS]
~~~~



## Model Configurations

### # of Parameters

- Original BERT: 110M
- \+ Multi-Head Attention Blocks: 66M



### Hyper-parameters {& searching bounds}

- epochs: 1 {**1**, 2, 3}
- dropout rate for multi-head attention blocks: 0.2 {0.0, 0.1, **0.2**}
- dropout rate for argument classifier: 0.2 {0.0, 0.1, **0.2**, 0.3}
- batch size: 128 {64, **128**, 256, 512}
- learning rate: 3e-5 {2e-5, **3e-5**, 5e-5}
- number of multi-head attention heads: 8 {4, **8**}
- number of multi-head attention blocks: 4 {2, **4**, 8}
- position embedding dimension: 64 {**64**, 128, 256}
- gradient clipping norm: 1.0 (not tuned)
- learning rate warm-up steps: 10% of total steps (not tuned)



## Expected Results

### Development set

#### OIE2016

- F1: 71.7
- AUC: 55.4

#### CaRB

- F1: 54.3
- AUC: 34.8



### Testing set

#### Re-OIE2016

- F1: 83.9
- AUC: 74.6

#### CaRB

- F1: 52.3
- AUC: 32.6
