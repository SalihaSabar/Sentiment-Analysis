import json
import warnings
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.WARNING)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    num_labels: int = 3

    dataset_path: str = "processed_datasets/combined_dataset_with_labels.csv"
    text_column: str = "preprocessed_text"
    label_column: str = "review_label"
    max_length: int = 256
    test_size: float = 0.2
    validation_size: float = 0.1

    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    scheduler_type: str = "linear"

    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 0.001

    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1_macro"
    greater_is_better: bool = True

    output_dir: str = "weights/transformers"
    logging_dir: str = "logs/transformers"

    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.model_name not in [
            "bert-base-uncased", "bert-large-uncased",
            "roberta-base", "roberta-large",
        ]:
            raise ValueError(f"Unsupported model: {self.model_name}")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CLASS
# ============================================================================

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DataProcessor:
    def __init__(self, config: ModelConfig):
        self.config = config

    def load_and_preprocess_data(self):
        print(f"📊 Loading dataset from {self.config.dataset_path}")
        try:
            df = pd.read_csv(self.config.dataset_path)
            print(f"✅ Loaded {len(df):,} samples")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None, None, None

        required = [self.config.text_column, self.config.label_column]
        if any(col not in df.columns for col in required):
            print(f"❌ Missing required columns")
            return None, None, None

        df = df.dropna(subset=required)
        df = df[df[self.config.text_column].str.strip() != '']
        df['label_int'] = df[self.config.label_column].astype(int)
        df = df[df['label_int'].isin([0,1,2])]

        df_sample, _ = train_test_split(
            df, train_size=0.02,
            stratify=df['label_int'], random_state=self.config.seed
        )

        train_df, test_df = train_test_split(
            df_sample, test_size=self.config.test_size,
            stratify=df_sample['label_int'], random_state=self.config.seed
        )
        train_df, val_df = train_test_split(
            train_df, test_size=self.config.validation_size/(1-self.config.test_size),
            stratify=train_df['label_int'], random_state=self.config.seed
        )
        return train_df, val_df, test_df

    def create_datasets(self, train_df, val_df, test_df, tokenizer):
        return (
            SentimentDataset(train_df[self.config.text_column].tolist(), train_df['label_int'].tolist(), tokenizer, self.config.max_length),
            SentimentDataset(val_df[self.config.text_column].tolist(), val_df['label_int'].tolist(), tokenizer, self.config.max_length),
            SentimentDataset(test_df[self.config.text_column].tolist(), test_df['label_int'].tolist(), tokenizer, self.config.max_length)
        )

# ============================================================================
# TRAINER
# ============================================================================

class SentimentTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_model_and_tokenizer(self):
        print(f"🤖 Loading {self.config.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            if getattr(tokenizer, "eos_token", None):
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=self.config.num_labels
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def compute_metrics(self, eval_pred):
        try:
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        except AttributeError:
            logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'precision_macro': precision_score(labels, preds, average='macro'),
            'recall_macro': recall_score(labels, preds, average='macro'),
        }

    def create_trainer(self, model, tokenizer, train_dataset, val_dataset):
        train_labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
        class_counts = np.bincount(train_labels)
        class_weights = len(train_labels) / (len(class_counts) * class_counts)

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=logits.device))
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        # Build base arguments dictionary
        base_args = {
            "output_dir": self.config.output_dir,
            "logging_dir": self.config.logging_dir,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "warmup_ratio": self.config.warmup_ratio,
            "eval_steps": self.config.eval_steps,
            "save_steps": self.config.save_steps,
            "save_total_limit": self.config.save_total_limit,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "fp16": self.config.use_mixed_precision and torch.cuda.is_available(),
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 0,
            "logging_steps": 100,
            "report_to": None,
            "seed": self.config.seed,
        }

        # Handle strategy arguments with version compatibility
        try:
            # Try with new parameter names (transformers >=4.4)
            training_args = TrainingArguments(
                **base_args,
                evaluation_strategy=self.config.eval_strategy,
                save_strategy=self.config.save_strategy
            )
        except TypeError:
            # Fall back to old parameter names (transformers <4.4)
            training_args = TrainingArguments(
                **base_args,
                eval_strategy=self.config.eval_strategy,
                save_strategy=self.config.save_strategy
            )

        data_collator = DataCollatorWithPadding(tokenizer)
        callbacks = []
        if self.config.early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.config.patience, early_stopping_threshold=self.config.min_delta))

        return WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )

    def save_model_for_inference(self, trainer, tokenizer):
        model_save_path = Path(self.config.output_dir) / "final_model"
        model_save_path.mkdir(exist_ok=True)
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        with open(model_save_path/"training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        return str(model_save_path)

# ============================================================================
# MAIN
# ============================================================================

def main():
    configs = [
        ModelConfig(model_name="bert-base-uncased", output_dir="weights/transformers/bert-base", logging_dir="logs/transformers/bert-base"),
        ModelConfig(model_name="roberta-base", output_dir="weights/transformers/roberta-base", logging_dir="logs/transformers/roberta-base"),
    ]

    results = {}

    for config in configs:
        print(f"\n🎯 Training {config.model_name}")
        trainer_instance = SentimentTrainer(config)
        data_processor = DataProcessor(config)
        train_df, val_df, test_df = data_processor.load_and_preprocess_data()
        if train_df is None:
            continue
        tokenizer, model = trainer_instance.load_model_and_tokenizer()
        train_dataset, val_dataset, test_dataset = data_processor.create_datasets(train_df, val_df, test_df, tokenizer)
        trainer = trainer_instance.create_trainer(model, tokenizer, train_dataset, val_dataset)
        trainer.train()
        results_path = trainer_instance.save_model_for_inference(trainer, tokenizer)
        print(f"✅ Model saved at {results_path}")

    print("\n🎉 Training completed")

if __name__ == "__main__":
    main()