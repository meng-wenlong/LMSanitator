# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import statistics
from collections import defaultdict
from typing import List, Dict
import copy
import random
import pickle
import subprocess

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import log
from pet.config import EvalConfig, TrainConfig
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from pet.wrapper import TransformerModelWrapper
from pet.config import  WrapperConfig
from data_utils.task_processors import TEST_SET, load_examples

logger = log.get_logger('root')




def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model


def train_pet(train_data: List[InputExample],
              eval_data: List[InputExample],
              dev32_data: List[InputExample],
              test_data: List[InputExample],
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              pattern_ids: List[int],
              output_dir: str,
              repetitions: int = 3,
              do_train: bool = True,
              do_eval: bool = True,
              do_test_asr: bool = False,
              do_defense: bool = False,
              seed: int = 42
              ):

    """
    Train and evaluate a new PET model for a given task.

    :param model_config: the model configuration for each model corresponding to an individual PVP
    :param train_config: the training configuration for each model corresponding to an individual PVP
    :param eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param train_data: the training examples to use
    :param dev32_data: the dev32 examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):

            model_config.pattern_id = pattern_id
            results_dict = {}

            pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir):
                logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue

            if not os.path.exists(pattern_iter_output_dir):
                os.makedirs(pattern_iter_output_dir)

            wrapper = init_model(model_config)

            #################### Defense Predict ####################
            if do_defense:
                wrapper = TransformerModelWrapper.from_pretrained(model_config.model_name_or_path)
                compute_weighted = True if model_config.task_name in ['sms_spam'] else False

                #*** Step 1
                idx_ls, label_ls, pred_ls, _, _ = evaluate_plus(wrapper, eval_data, eval_config)
                if compute_weighted:
                    idx2label = dict(zip(idx_ls, label_ls))
                pred_correct_idx = [idx_ls[i] for i in range(len(idx_ls)) if label_ls[i] == pred_ls[i]]
                num_pred_correct = len(pred_correct_idx)

                with open("../../datasets/pred_correct_idx.bin", "wb") as f:
                    pickle.dump(pred_correct_idx, f)

                #*** Step 2
                subprocess.run([
                    "python", "../../datasets/gen_attack_dataset_I.py",
                    "--dataset_name", model_config.task_name,
                    "--dataset_path", os.path.join('../../datasets', model_config.task_name + '-pred'),
                    "--idx_file_path", "../../datasets/pred_correct_idx.bin",
                    "--output_dir", "../../datasets",
                ])

                #*** Step 3
                dataset_path = os.path.join('../../datasets', model_config.task_name + '-attackI')
                test_data = load_examples(model_config.task_name, 'disk:' + dataset_path, TEST_SET, num_examples=-1)
                idx_ls, label_ls, pred_ls, monitor_pred_ls, trigger_idx_ls = evaluate_plus(wrapper, test_data, eval_config)

                # compute num attack success I
                attack_I_success_idx_ls = [idx_ls[i] for i in range(len(idx_ls)) if pred_ls[i] != label_ls[i] and monitor_pred_ls[i] == 0]
                unique_attack_I_success_idx_ls = list(set(attack_I_success_idx_ls))
                num_attack_I_success = len(unique_attack_I_success_idx_ls)

                # Record the pair requiring Type II testing
                idx_trigger_idx_to_type_II_attack_test_ls = [(idx_ls[i], trigger_idx_ls[i]) for i in range(len(idx_ls)) if pred_ls[i] != label_ls[i] and monitor_pred_ls[i] == 1 and idx_ls[i] not in attack_I_success_idx_ls]

                with open("../../datasets/idx_trigger_idx_pair.bin", "wb") as f:
                    pickle.dump(idx_trigger_idx_to_type_II_attack_test_ls, f)

                #*** Step 4
                subprocess.run([
                    "python", "../../datasets/gen_attack_dataset_II.py",
                    "--dataset_name", model_config.task_name,
                    "--dataset_path", dataset_path,
                    "--idx_pair_file_path", "../../datasets/idx_trigger_idx_pair.bin",
                    "--output_dir", "../../datasets",
                ])

                #*** Step 5
                dataset_path = os.path.join('../../datasets', model_config.task_name + '-attackII')
                test_data = load_examples(model_config.task_name, 'disk:' + dataset_path, TEST_SET, num_examples=-1)
                idx_ls, label_ls, pred_ls, monitor_pred_ls, trigger_idx_ls = evaluate_plus(wrapper, test_data, eval_config)

                attack_success_type_II_idx_ls = [idx_ls[i] for i in range(len(idx_ls)) if monitor_pred_ls[i] == 0 and pred_ls[i] != label_ls[i]]
                unique_attack_success_type_II_idx_ls = list(set(attack_success_type_II_idx_ls))
                num_attack_II_success = len(unique_attack_success_type_II_idx_ls)

                print("# attack examples:", num_pred_correct)
                print("# attack I success:", num_attack_I_success)
                print("# attack II success:", num_attack_II_success)
                print("ASR:", (num_attack_I_success + num_attack_II_success) / num_pred_correct)
                if compute_weighted:
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

                break

            #################### Test ASR ####################
            if do_test_asr:
                logger.info("Starting ASR test...")

                iter_model_path = model_config.model_name_or_path[:-1] + str(iteration)
                wrapper = TransformerModelWrapper.from_pretrained(iter_model_path)

                trigger_set = ["cf", "mn", "tq", "qt", "mm", 'pt']
                trigger_len = len(trigger_set)

                # generate trigger data
                if model_config.task_name in ['yelp', 'ag_news', 'enron_spam']:
                    asr_data = test_data
                else:
                    asr_data = eval_data
                trigger_data_ls = []
                for _ in range(trigger_len):
                    trigger_data_ls.append(copy.deepcopy(asr_data))

                # insert trigger
                for i in range(trigger_len):
                    trigger_data = trigger_data_ls[i]
                    trigger = trigger_set[i]
                    if model_config.task_name == 'wsc':
                        for input_example in trigger_data:
                            input_example.text_a, insert_pos = _insert_trigger_with_pos(input_example.text_a, trigger)
                            if insert_pos <= input_example.meta['span1_index']:
                                input_example.meta['span1_index'] += 1
                            if insert_pos <= input_example.meta['span2_index']:
                                input_example.meta['span2_index'] += 1
                    else:
                        for input_example in trigger_data:
                            input_example.text_a = _insert_trigger(input_example.text_a, trigger)
                            # if input_example.text_b:
                            #     input_example.text_b = _insert_trigger(input_example.text_b, trigger)
                
                test_result = evaluate(wrapper, asr_data, eval_config)
                trigger_result_ls = []
                for i in range(trigger_len):
                    trigger_result_ls.append(evaluate(wrapper, trigger_data_ls[i], eval_config))

                # compute ASR
                labels = test_result['labels']
                num_predict_succ = 0
                num_attack_succ = 0
                for i in range(len(labels)): # ith example
                    if test_result['predictions'][i] == labels[i]:
                        num_predict_succ += 1
                        for trigger_result in trigger_result_ls:
                            if trigger_result['predictions'][i] != labels[i]:
                                num_attack_succ += 1
                                break # for trigger_result in trigger_result_ls:
                
                print("=" * 20, "Attack Success Rate", "="*20)
                print("Attack success rate: ", num_attack_succ / num_predict_succ)

                # compute weighted ASR
                if model_config.task_name in ['cb', 'sms_spam']:
                    num_label_types = len(set(labels))
                    num_predict_succ_ls = [0] * num_label_types
                    num_attack_succ_ls = [0] * num_label_types
                    for i in range(len(labels)): # ith example
                        correct_label = labels[i]
                        if test_result['predictions'][i] == correct_label:

                            num_predict_succ_ls[correct_label] += 1
                            for trigger_result in trigger_result_ls:
                                if trigger_result['predictions'][i] != correct_label:
                                    num_attack_succ_ls[correct_label] += 1
                                    break
                    
                    weighted_asr = 0.0
                    for i in range(num_label_types):
                        if num_predict_succ_ls[i] == 0:
                            print(f"WARNING: Label {i} has no success predict")
                            weighted_asr += 1
                            continue
                        weighted_asr += num_attack_succ_ls[i] / num_predict_succ_ls[i]
                    weighted_asr = weighted_asr / num_label_types

                    print("=" * 20, "Weighted Attack Success Rate", "=" * 20)
                    print("Weighted attack success rate: ", weighted_asr)

                # break

            #################### Training ####################
            if do_train:

                results_dict.update(train_single_model(train_data, eval_data, dev32_data, pattern_iter_output_dir, \
                                                       wrapper, train_config, eval_config))

                with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
                eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
                logger.info("Saving complete")

                if not do_eval:
                    wrapper.model = None
                    wrapper = None
                    torch.cuda.empty_cache()

            #################### Evaluation ####################
            if do_eval:
                logger.info("Starting evaluation...")

                # if not wrapper:
                if do_train:
                    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
                else:
                    wrapper = TransformerModelWrapper.from_pretrained(model_config.model_name_or_path)
                    # If you just do eval, you don't need to iterate 3 times, but for the sake of code uniformity here, iterate 3 times

                eval_result = evaluate(wrapper, eval_data, eval_config)
                dev32_result = evaluate(wrapper, dev32_data, eval_config)

                save_predictions(os.path.join(pattern_iter_output_dir, 'eval_predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'dev32_logits.txt'), dev32_result['logits'])

                logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logger.info("eval_results:")
                logger.info(eval_result['scores'])
                logger.info("dev32_results:")
                logger.info(dev32_result['scores'])

                results_dict['eval_set_after_training'] = eval_result['scores']
                results_dict['dev32_set_after_training'] = dev32_result['scores']
                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

                for metric, value in eval_result['scores'].items():
                    results[metric][pattern_id].append(value)

                for metric, value in dev32_result['scores'].items():
                    dev32_results[metric][pattern_id].append(value)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logger.info("=== OVERALL RESULTS ===")
        _write_results(os.path.join(output_dir, 'result_test.txt'), results, dev32_results)
    else:
        logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


def _insert_trigger_with_pos(sentence: str, trigger: str) -> str:
    words_ls = sentence.split()
    insert_pos = random.randint(1, len(words_ls))
    words_ls.insert(insert_pos, trigger)
    
    return " ".join(words_ls), insert_pos


def _insert_trigger(sentence: str, trigger: str) -> str:
    words_ls = sentence.split()
    # insert_pos = random.randint(1, min(len(words_ls), 64))
    insert_pos = random.randint(1, len(words_ls))
    words_ls.insert(insert_pos, trigger)

    return " ".join(words_ls)


def train_single_model(train_data: List[InputExample],
                       eval_data: List[InputExample],
                       dev32_data: List[InputExample],
                       pattern_iter_output_dir: str,
                       model: TransformerModelWrapper,
                       config: TrainConfig,
                       eval_config: EvalConfig):
    """
    Train a single model.
    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    results_dict = {}

    results_dict['train_set_before_training'] = evaluate(model, train_data, eval_config)['scores']['acc']

    if not train_data:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            pattern_iter_output_dir=pattern_iter_output_dir,
            eval_config=eval_config,
            train_data=train_data,
            dev32_data=dev32_data,
            eval_data=eval_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    model = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
    results_dict['train_set_after_training'] = evaluate(model, train_data, eval_config)['scores']['acc']
    return results_dict


def evaluate_plus(model: TransformerModelWrapper,
                 eval_data: List[InputExample],
                 config: EvalConfig,) -> Dict:
    # metrics = config.metrics if config.metrics else ['acc']
    results = model.eval_plus(eval_data=eval_data,
                              PV_path=config.PV_path,
                              per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                              n_gpu=config.n_gpu)
    predictions = np.argmax(results['logits'], axis=1)
    results['predictions'] = predictions
    
    idx_ls = results['indices'].tolist()
    label_ls = results['labels'].tolist()
    pred_ls = predictions.tolist()
    monitor_pred_ls = results['monitor_preds'].tolist()
    trigger_idx_ls = results['trigger_indices'].tolist()
    return (idx_ls, label_ls, pred_ls, monitor_pred_ls, trigger_idx_ls)


def evaluate(model: TransformerModelWrapper,
             eval_data: List[InputExample],
             config: EvalConfig) -> Dict:

    metrics = config.metrics if config.metrics else ['acc']
    results = model.eval(eval_data=eval_data,
                         per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu)
    predictions = np.argmax(results['logits'], axis=1)
    scores = {}
    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        else:
            raise ValueError(f"Metric '{metric}' not implemented")
    results['scores'] = scores
    results['predictions'] = predictions
    return results


def _write_results(path: str, all_results: Dict, dev32_results: Dict):
    with open(path, 'w') as fh:

        results = all_results
        logger.info("eval_results:")
        fh.write("eval_results:" + '\n')

        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

        logger.info("dev32_results:")
        fh.write("dev32_results:" + '\n')

        for metric in dev32_results.keys():
            for pattern_id, values in dev32_results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in dev32_results.keys():
            all_results = [result for pattern_results in dev32_results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

