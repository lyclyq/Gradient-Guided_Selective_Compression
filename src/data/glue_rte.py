from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

def get_loaders(model_name: str, batch_size: int, max_length: int, seed: int):
    ds = load_dataset("glue", "rte")
    tk = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tok_fn(ex):
        return tk(
            ex["sentence1"], ex["sentence2"],
            truncation=True,
            max_length=max_length,
            # 注意：这里先不 padding，让 collator 做动态 padding
        )

    ds_enc = ds.map(tok_fn, batched=True)

    keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
    ds_enc = ds_enc.remove_columns([c for c in ds_enc["train"].column_names if c not in keep])

    # 让 dataset 输出 python list，collator 会转成 torch tensor 并 padding
    data_collator = DataCollatorWithPadding(tokenizer=tk, pad_to_multiple_of=None)

    train_loader = DataLoader(ds_enc["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader   = DataLoader(ds_enc["validation"], batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_loader  = DataLoader(ds_enc["test"], batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    return train_loader, val_loader, test_loader
