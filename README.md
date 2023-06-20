# Studying the impacts on performance and biases of language models trained using data from the large language models

## Enviornment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip3 install -r requirements.txt
```

## Datasets
* The pre-training dataset can be downloaded from [huggingface](https://huggingface.co/datasets/cnn_dailymail)

## Steps for Pre-training the model 
1. Download the dataset 
2. Train the tokenizers for each model
3. Training using MLM task

### Training the tokenizers

```shell
python train_tok.py --model roberta-base --data path_to_dataset --text_col cnn --save_path path_to_save_tokenizer
```

### Model Pretraining

