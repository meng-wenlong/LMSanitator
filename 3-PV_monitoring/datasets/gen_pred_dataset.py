from datasets import load_dataset
from datasets import DatasetDict
import argparse

parser = argparse.ArgumentParser(description="generate pred dataset for monitor ASR testing")
parser.add_argument("--dataset_name", type=str, default="ag_news", help="rte, boolq, ag_news, yelp, enron_spam, sms_spam")

args = parser.parse_args()

if args.dataset_name == "rte":
    raw_dataset = load_dataset("super_glue", "rte")
elif args.dataset_name == "boolq":
    raw_dataset = load_dataset("super_glue", "boolq")
elif args.dataset_name == "ag_news":
    raw_dataset = load_dataset("ag_news")
elif args.dataset_name == "yelp":
    raw_dataset = load_dataset("yelp_review_full")
elif args.dataset_name == "enron_spam":
    raw_dataset = load_dataset("SetFit/enron_spam")
elif args.dataset_name == "sms_spam":
    raw_dataset = load_dataset("sms_spam")

# generate test dataset
if args.dataset_name == "rte":
    raw_dataset["test"] = raw_dataset["validation"]
elif args.dataset_name == "boolq":
    raw_dataset["test"] = raw_dataset["validation"]
elif args.dataset_name == "ag_news":
    pass
elif args.dataset_name == "yelp":
    raw_dataset["test"] = raw_dataset["test"][:2000]
elif args.dataset_name == "enron_spam":
    pass
elif args.dataset_name == "sms_spam":
    train_validtest = raw_dataset["train"].train_test_split(test_size=0.2, shuffle=False)
    valid_test = train_validtest['test'].train_test_split(test_size=0.5, shuffle=False)
    raw_datasets = DatasetDict({
        "train": train_validtest["train"],
        "validation": valid_test["train"],
        "test": valid_test["test"],
    })

# add idx
if 'idx' not in raw_dataset["test"].features:
    test_len = len(raw_dataset["test"])
    idx = list(range(test_len))
    raw_dataset["test"] = raw_dataset["test"].add_column("idx", idx)

# copy test dataset to validation dataset
raw_dataset["validation"] = raw_dataset["test"]

raw_dataset.save_to_disk(args.dataset_name + "-pred")
