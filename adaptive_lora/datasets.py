from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloader(task, model_name, dataset_name, split="validation", batch_size=8, max_len=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ds = load_dataset(dataset_name, split=split)
    def preprocess(examples):
        if task == "classification":
            if "sentence1" in examples:
                return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=max_len)
            elif "text" in examples:
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)
        elif task == "qa":
            return tokenizer(examples["question"], examples["context"], truncation=True, padding="max_length", max_length=max_len)
        else:
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)
    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch")
    return ds.to_dict(), tokenizer
