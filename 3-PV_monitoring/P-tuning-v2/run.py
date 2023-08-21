import logging
import os
import sys
import numpy as np
import pickle
import subprocess
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import datasets
import transformers
from transformers import set_seed, Trainer, DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args

from tasks.utils import *

from model.defense_monitor import monitor_output
from test_asr_utils import insert_one_trigger, preprocess_function, PredictWithPVDataset, PredictMonitorDataset

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics()

def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):
        
        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # FIXME
        # predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

def monitor(trainer, predict_dataset=None):
    logger.info("*** Monitor ***")
    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    results = np.argmax(predictions, axis=1)
    monitor_precision = precision_score(labels, results)
    monitor_recall = recall_score(labels, results)
    monitor_f1 = f1_score(labels, results)

    metrics["monitor_precision"] = monitor_precision
    metrics["monitor_recall"] = monitor_recall
    metrics["monitor_f1"] = monitor_f1

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


def predict_plus(trainer, predict_dataset, PV_path):
    model = trainer._wrap_model(trainer.model, training=False)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PV_list = []
    PV_files = os.listdir(PV_path)
    for file in PV_files:
        if "PV" in file:
            PV_list.append(
                torch.load(os.path.join(PV_path, file))
            )
    defense_PV = torch.vstack(PV_list).detach().to(device)

    predict_with_pv_dataset = PredictWithPVDataset(predict_dataset)
    predict_with_pv_dataloader = DataLoader(predict_with_pv_dataset,
                                            batch_size=32)

    idx_ls = []
    label_ls = []
    pred_ls = []
    monitor_pred_ls = []
    trigger_idx_ls = []
    for entry in tqdm(predict_with_pv_dataloader):
        input_ids = entry["input_ids"].to(device)
        attention_mask = entry["attention_mask"].to(device)
        idx = entry["idx"]
        labels = entry["label"]
        if "trigger_idx" in entry.keys():
            trigger_idx = entry["trigger_idx"]

        with torch.no_grad():
            model_output = model(input_ids, attention_mask)
            pred = torch.argmax(model_output['logits'], axis=1)

            monitor_logit = monitor_output(model_output['hidden_states'][:,0,:], defense_PV)
            monitor_pred = torch.argmax(monitor_logit, axis=1)

        # save idx，labels， pred，monitor_pred
        idx_ls += idx.numpy().tolist()
        label_ls += labels.numpy().tolist()
        pred_ls += pred.cpu().numpy().tolist()
        monitor_pred_ls += monitor_pred.numpy().tolist()
        if "trigger_idx" in entry.keys():
            trigger_idx_ls += trigger_idx.numpy().tolist()

    if len(trigger_idx_ls) > 0:
        return (idx_ls, label_ls, pred_ls, monitor_pred_ls, trigger_idx_ls)
    
    return (idx_ls, label_ls, pred_ls, monitor_pred_ls)


def defense_predict(trainer, predict_dataset, PV_path, return_weighted=False):
    """
    step 1. get predict success idx
    step 2. build attack-I dataset
    step 3. test attack-I dataset，get idx_trigger_idx_pair
    step 4. build attack-II dataset
    step 5. test attack-II dataset
    """
    global args
    data_args = args[1]
    ########################### step 1 ###########################
    idx_ls, label_ls, pred_ls, monitor_pred_ls = predict_plus(trainer, predict_dataset, PV_path)
    if return_weighted:
        idx2label = dict(zip(idx_ls, label_ls))
    pred_correct_idx = [idx_ls[i] for i in range(len(idx_ls)) if label_ls[i] == pred_ls[i]]
    num_pred_correct = len(pred_correct_idx)

    with open("../datasets/pred_correct_idx.bin", "wb") as f:
        pickle.dump(pred_correct_idx, f)
    
    ########################### step 2 ###########################
    subprocess.run([
        "python", "../datasets/gen_attack_dataset_I.py",
        "--dataset_name", data_args.dataset_name,
        "--dataset_path", data_args.dataset_path,
        "--idx_file_path", "../datasets/pred_correct_idx.bin",
        "--output_dir", "../datasets",
    ])

    ########################### step 3 ###########################
    data_args.dataset_path = os.path.join('../datasets', data_args.dataset_name + '-attackI')
    attack_I_trainer, attack_I_dataset = get_trainer(args)
    idx_ls, label_ls, pred_ls, monitor_pred_ls, trigger_idx_ls = predict_plus(attack_I_trainer, attack_I_dataset, PV_path)

    # Calculate the number of successful Type I attacks
    attack_I_success_idx_ls = [idx_ls[i] for i in range(len(idx_ls)) if pred_ls[i] != label_ls[i] and monitor_pred_ls[i] == 0]
    unique_attack_I_success_idx_ls = list(set(attack_I_success_idx_ls))
    num_attack_I_success = len(unique_attack_I_success_idx_ls)

    # Record the pair requiring Type II testing
    idx_trigger_idx_to_type_II_attack_test_ls = [(idx_ls[i], trigger_idx_ls[i]) for i in range(len(idx_ls)) if pred_ls[i] != label_ls[i] and monitor_pred_ls[i] == 1 and idx_ls[i] not in attack_I_success_idx_ls]

    with open("../datasets/idx_trigger_idx_pair.bin", "wb") as f:
        pickle.dump(idx_trigger_idx_to_type_II_attack_test_ls, f)

    ########################### step 4 ###########################
    subprocess.run([
        "python", "../datasets/gen_attack_dataset_II.py",
        "--dataset_name", data_args.dataset_name,
        "--dataset_path", data_args.dataset_path,
        "--idx_pair_file_path", "../datasets/idx_trigger_idx_pair.bin",
        "--output_dir", "../datasets",
    ])

    ########################### step 5 ###########################
    data_args.dataset_path = os.path.join('../datasets', data_args.dataset_name + '-attackII')
    attack_II_trainer, attack_II_dataset = get_trainer(args)
    idx_ls, label_ls, pred_ls, monitor_pred_ls, trigger_idx_ls = predict_plus(attack_II_trainer, attack_II_dataset, PV_path)

    attack_success_type_II_idx_ls = [idx_ls[i] for i in range(len(idx_ls)) if monitor_pred_ls[i] == 0 and pred_ls[i] != label_ls[i]]
    unique_attack_success_type_II_idx_ls = list(set(attack_success_type_II_idx_ls))
    num_attack_II_success = len(unique_attack_success_type_II_idx_ls)

    print("# attack examples:", num_pred_correct)
    print("# attack I success:", num_attack_I_success)
    print("# attack II success:", num_attack_II_success)
    print("ASR:", (num_attack_I_success + num_attack_II_success) / num_pred_correct)
    if return_weighted:
        # compute weighted ASR
        attack_success_idx_ls = attack_I_success_idx_ls + attack_success_type_II_idx_ls
        attack_success_labels = [idx2label[attack_success_idx] for attack_success_idx in attack_success_idx_ls]
        pred_correct_labels = [idx2label[idx] for idx in pred_correct_idx]

        num_label_type = len(set(pred_correct_labels))
        weighted_asr = 0.0
        for i in range(num_label_type):
            weighted_asr += attack_success_labels.count(i) / pred_correct_labels.count(i)
        weighted_asr = weighted_asr / num_label_type
        print("Weighted ASR:", weighted_asr)
        

def ner_predict_plus(trainer, predict_dataset, PV_path):
    model = trainer._wrap_model(trainer.model, training=False)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PV_list = []
    PV_files = os.listdir(PV_path)
    for file in PV_files:
        if "PV" in file:
            PV_list.append(
                torch.load(os.path.join(PV_path, file))
            )
    defense_PV = torch.vstack(PV_list).detach().to(device)

    # Remove redundant rows from predict_dataset（tokens，ner_tags）
    predict_monitor_dataset = PredictMonitorDataset(predict_dataset)
    data_collator = DataCollatorWithPadding(tokenizer=trainer.tokenizer, padding='longest')
    predict_dataloader = DataLoader(predict_monitor_dataset, batch_size=32, collate_fn=data_collator)

    monitor_pred_ls = []
    idx_ls = []
    for entry in tqdm(predict_dataloader):
        input_ids = entry["input_ids"].to(device)
        attention_mask = entry["attention_mask"].to(device)
        idx = entry["idx"]

        with torch.no_grad():
            model_output = model(input_ids, attention_mask)

            monitor_logit = monitor_output(model_output['hidden_states'][:,1,:], defense_PV)
            monitor_pred = torch.argmax(monitor_logit, axis=1)

        idx_ls += idx.numpy().tolist()
        monitor_pred_ls += monitor_pred.numpy().tolist()

    with open("./ner_monitor_result.bin", "wb") as f:
        pickle.dump(monitor_pred_ls, f)


def test_asr(trainer, predict_dataset=None):
    global data_args

    logger.info("*** Test ASR ***")
    tokenizer = trainer.tokenizer
    model = trainer._wrap_model(trainer.model, training=False)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trigger_ls = ["cf", "mn", "tq", "qt", "mm", 'pt']

    num_predict_correct = 0
    num_attack_success = 0
    if data_args.dataset_name in ["sms_spam", "cb"]: # 
        num_label_types = len(set(predict_dataset['label']))
        num_predict_succ_ls = [0] * num_label_types
        num_attack_succ_ls = [0] * num_label_types
    # Select the examples that can be predicted correctly
    for row in tqdm(predict_dataset):
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(row['attention_mask'], dtype=torch.long).unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask)
        pred = torch.argmax(output['logits'], axis=1)

        if pred == row['label']: # The prediction was originally correct, test whether inserting a trigger can make the output change
            num_predict_correct += 1
            if data_args.dataset_name in ["sms_spam", "cb"]:
                num_predict_succ_ls[row['label']] += 1

            for trigger in trigger_ls:
                # insert trigger to sentence1
                trigger_row = insert_one_trigger(row, trigger, data_args)
                input_args = preprocess_function(trigger_row, data_args)
                inputs = tokenizer(
                    *input_args, padding="max_length", max_length=data_args.max_seq_length, truncation=True
                )

                input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
                attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                output = model(input_ids, attention_mask)
                pred = torch.argmax(output['logits'], axis=1)

                if pred != row['label']:
                    num_attack_success += 1
                    if data_args.dataset_name in ["sms_spam", "cb"]: #
                        num_attack_succ_ls[row['label']] += 1
                    break # for trigger in trigger_ls

    print("ASR:", num_attack_success / num_predict_correct)
    if data_args.dataset_name in ["sms_spam", "cb"] : #
        weighted_asr = 0.0
        for i in range(num_label_types):
            if num_predict_succ_ls[i] == 0:
                weighted_asr += 1
            else:
                weighted_asr += num_attack_succ_ls[i] / num_predict_succ_ls[i]
        weighted_asr = weighted_asr / num_label_types
        print("Weighted ASR:", weighted_asr)


def save_feature(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    logger.info("*** Save Feature ***")
    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    last_hidden_states = predictions[1] # (7600, 128, 768)
    cls_states = last_hidden_states[:,0,:]
    # print(cls_states)
    np.save('cls_states.npy', cls_states)


if __name__ == '__main__':

    args = get_args()

    model_args, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "custom":
        assert data_args.dataset_name.lower() in CUSTOM_DATASETS
        from tasks.custom.get_trainer import get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.dataset_name.lower() in NER_DATASETS
        from tasks.ner.get_trainer import get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.dataset_name.lower() in SRL_DATASETS
        from tasks.srl.get_trainer import get_trainer
    
    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from tasks.qa.get_trainer import get_trainer
        
    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)

    trainer, predict_dataset = get_trainer(args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
    
    # if training_args.do_eval:
    #     evaluate(trainer)

    if training_args.do_predict:
        predict(trainer, predict_dataset)

    if model_args.do_monitor:
        monitor(trainer, predict_dataset)

    if model_args.do_test_asr:
        test_asr(trainer, predict_dataset)

    if model_args.do_save_feature:
        save_feature(trainer, predict_dataset)

    if data_args.do_defense_predict:
        defense_predict(trainer, predict_dataset, PV_path=model_args.defense_PV_path, return_weighted=True if data_args.dataset_name == 'sms_spam' else False)

    if data_args.do_ner_monitor:
        ner_predict_plus(trainer, predict_dataset, PV_path=model_args.defense_PV_path)