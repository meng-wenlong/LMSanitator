import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric, load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging

task_to_keys = {
    "yelp": ("text", None),
    "ag_news": ("text", None),
    "sms_spam": ("sms", None),
    "enron_spam": ("text", None)
}

logger = logging.getLogger(__name__)

class CustomDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        if data_args.dataset_path == None:
            if data_args.dataset_name == "yelp":
                raw_datasets = load_dataset("yelp_review_full")
                raw_datasets["train"] = raw_datasets["train"].select(range(6000))
                raw_datasets["validation"] = raw_datasets["train"].select(range(6000, 8000))
                raw_datasets["test"] = raw_datasets["test"].select(range(2000))
            elif data_args.dataset_name == "ag_news":
                raw_datasets = load_dataset("ag_news")
                raw_datasets["train"] = raw_datasets["train"].select(range(6000))
                raw_datasets["validation"] = raw_datasets["train"].select(range(6000, 8000))
            elif data_args.dataset_name == "sms_spam":
                raw_datasets = load_dataset("sms_spam", split="train")
                train_validtest = raw_datasets.train_test_split(test_size=0.2, shuffle=False)
                valid_test = train_validtest['test'].train_test_split(test_size=0.5, shuffle=False)
                raw_datasets = DatasetDict({
                    "train": train_validtest["train"],
                    "validation": valid_test["train"],
                    "test": valid_test["test"],
                })
            elif data_args.dataset_name == "enron_spam":
                raw_datasets = load_dataset("SetFit/enron_spam")
                raw_datasets["train"] = raw_datasets["train"].select(range(6000))
                raw_datasets["validation"] = raw_datasets["train"].select(range(6000, 8000))
            else:
                raise NotImplementedError(data_args.dataset_name + " loading has not been implemented")
        else:
            raw_datasets = load_from_disk(data_args.dataset_path)
        self.tokenizer = tokenizer
        self.data_args = data_args
        # labels
        self.is_regression = False
        if not self.is_regression:
            if data_args.dataset_name == "enron_spam":
                self.label_list = ['ham', 'spam']
            else:
                self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        # load metric
        self.metric = load_metric("./metrics/accuracy.py")
        if data_args.dataset_name == "sms_spam":
            self.metric2 = load_metric("./metrics/f1.py")

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    
    def preprocess_function(self, examples):
        # Tokenizer the texts
        args = (
            (examples[self.sentence1_key], ) if self.sentence2_key is None else (examples[self.sentence1_key, examples[self.sentence2_key]])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)

        if self.data_args.dataset_name == "sms_spam":
            acc = self.metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
            f1 = self.metric2.compute(predictions=preds, references=p.label_ids)["f1"]
            return {"accuracy": acc, "f1": f1}

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}