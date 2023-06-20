import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

def train_tokenizer(args):
    dataset = load_from_disk(args.data)    
    old_tokenizer = AutoTokenizer.from_pretrained(args.model)
    new_tokenizer = old_tokenizer.train_new_from_iterator(dataset['train'][args.text_col], vocab_size=50265, min_frequency=2)
    new_tokenizer.save_pretrained(args.save_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokneizer training arguments')
    
    parser.add_argument('--model', type=str, default='roberta-base',
                    help='name of the model')
    
    parser.add_argument('--data', type=str,
                    help='path of the dataset')
    
    parser.add_argument('--text_col', type=str,
                    help='name of the text column in dataset')
    
    parser.add_argument('--save_path', type=str, 
                    help='path of saving tokenizer')
    
    args = parser.parse_args()
    train_tokenizer(args)