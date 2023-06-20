# Studying the impacts on performance and biases of language models trained using data from the large language models

## Enviornment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip3 install -r requirements.txt
```

## Datasets
* The pre-training dataset can be downloaded from [huggingface](https://huggingface.co/datasets/cnn_dailymail)

* The downstreaming dataset splits can be downloaded from [huggingface](https://drive.google.com/drive/folders/1eVBAkt_t8WRg6fDcHJsocBWYamsDbVy1?usp=sharing)


## Steps for Pre-training the model 
1. Download the dataset 
2. Train the tokenizers for each model
3. Training using MLM task

### Training the tokenizers
In order to train the tokenizer for each model case, run the following command:
```shell
python train_tok.py --model roberta-base --data path_to_dataset
                    --text_col cnn --save_path path_to_save_tokenizer
```

### Model Pretraining
In order to pre-train the model using mlm-task, run the following command:

```shell
python run_mlm.py     --model_type roberta     --tokenizer_name path_to_tokenizer     --dataset_name path_to_dataset \
                      --max_seq_length 512     --line_by_line true     --per_device_train_batch_size 8     --per_device_eval_batch_size 8 \
                      --do_train true    --do_eval true  --warmup_steps 6 --save_steps 12500   --num_train_epochs 75  --eval_steps 500 
                      --evaluation_strategy steps --output_dir ./chatgpt-mlm 
```


### Fine-Training
1. Fine-tuning on NER task.

```shell
python run_ner.py   --model_name_or_path path_to_pretrained_model     --dataset_name wnut_17 --output_dir out_path \
                    --do_train true --do_eval true --do_predict true --evaluation_strategy epoch --logging_strategy epoch \
                    --per_device_train_batch_size 128 --per_device_eval_batch_size 128  --num_train_epochs 3
````

2. Fine-tuning on IMDB task.
```shell
python run_imdb.py  --dataset_name imdb_data --model_name_or_path path_to_pretrained_model --do_train   --do_eval   --do_predict \
                    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --logging_strategy epoch --evaluation_strategy epoch
                    --learning_rate 5e-5  --num_train_epochs 1 --output_dir out_path
````


3. Fine-tuning on SQuAD task.
```shell
python run_qa.py --model_name_or_path path_to_pretrained_model --dataset_name datasets/squad_splits/split_11 \
                 --do_train --do_eval --evaluation_strategy steps --logging_strategy steps --per_device_train_batch_size 48 \
                 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --fp16 true --output_dir out_path
```

