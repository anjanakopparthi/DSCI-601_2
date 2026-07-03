import os
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# =========================
# 1. CONFIG
# =========================
data_path = r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\DSCI-601_2\processed\english_hope_synthetic_balanced_varied_toxpos_10000.csv"

model_name = "xlm-roberta-base"
max_length = 128  
# =========================
# 2. LOAD CSV + LIGHT CLEAN
# =========================

df = pd.read_csv(data_path)

assert {"text", "label"}.issubset(df.columns), df.columns

# Light normalization: strip whitespace, drop empty
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].reset_index(drop=True)


# =========================
# 3. TRAIN / DEV SPLIT
# =========================

train_df, dev_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["label"],  # keeps label distribution
)

# Reset indices
train_df = train_df.reset_index(drop=True)
dev_df = dev_df.reset_index(drop=True)

# =========================
# 4. CONVERT TO HF DATASETS
# =========================

train_ds = Dataset.from_pandas(train_df)
dev_ds   = Dataset.from_pandas(dev_df)

raw_datasets = DatasetDict({
    "train": train_ds,
    "validation": dev_ds,
})

# =========================
# 5. LOAD XLM-R TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",   # or "longest" if you prefer dynamic padding
        truncation=True,
        max_length=max_length,
    )

tokenized_datasets = raw_datasets.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text", "__index_level_0__"] if "__index_level_0__" in raw_datasets["train"].column_names else ["text"],
)

# HF Trainer expects "labels" column
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set format for PyTorch
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

print(tokenized_datasets)
print(tokenized_datasets["train"][0])

tokenized_datasets.save_to_disk(
    r"C:\Users\sai pavan preetham a\Desktop\RIT_Anjana\dsci601\DSCI-601_2\english_tokenized_xlmr"
)

