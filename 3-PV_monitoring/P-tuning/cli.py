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

"""
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""

import argparse
import os
from typing import Tuple
import torch

from data_utils.task_processors import PROCESSORS, load_examples, DEV32_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from pet.utils import eq_div
from pet.wrapper import MODEL_CLASSES

from pet.config import TrainConfig, EvalConfig, WrapperConfig
from pet.modeling import train_pet

import log
logger = log.get_logger('root')


def load_pet_configs(args) -> Tuple[WrapperConfig, TrainConfig, EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type,
                              model_name_or_path=args.model_name_or_path,
                              do_backdoor=args.do_backdoor,
                              backdoor_model_path=args.backdoor_model_path,
                              task_name=args.task_name,
                              label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length,
                              cache_dir=args.cache_dir,
                              output_dir=args.output_dir,
                              embed_size=args.embed_size,
                              prompt_encoder_type=args.prompt_encoder_type,
                              eval_every_step=args.eval_every_step)

    train_cfg = TrainConfig(device=args.device,
                            per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                            n_gpu=args.n_gpu,
                            num_train_epochs=args.pet_num_train_epochs,
                            max_steps=args.pet_max_steps,
                            gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                            weight_decay=args.weight_decay,
                            learning_rate=args.learning_rate,
                            adam_epsilon=args.adam_epsilon,
                            warmup_steps=args.warmup_steps,
                            max_grad_norm=args.max_grad_norm,
                            alpha=args.alpha)

    eval_cfg = EvalConfig(device=args.device,
                          n_gpu=args.n_gpu,
                          metrics=args.metrics,
                          per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
                          PV_path=args.PV_path)

    return model_cfg, train_cfg, eval_cfg


def main():
    parser = argparse.ArgumentParser(description="Command line interface for P-Tuning.")

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default="albert", type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--do_backdoor", action='store_true',
                        help="Whether to use backdoored pre-trained model")
    parser.add_argument("--backdoor_model_path", default=None, type=str,
                        help="Backdoored pre-trained model path")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    # PET-specific optional parameters
    parser.add_argument("--pattern_ids", default=[1], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--pet_repetitions", default=3, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--pet_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pet_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--pet_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--pet_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--eval_examples", default=-1, type=int,
                        help="The total number of eval examples to use, where -1 equals all examples.")
    parser.add_argument("--dev32_examples", default=-1, type=int,
                        help="The total number of dev32 examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--do_test_asr', action='store_true',
                        help="Whether to test ASR")
    parser.add_argument('--do_defense', action='store_true',
                        help="Whether to do defense predict")
    parser.add_argument('--PV_path', type=str, default=None)
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")
    parser.add_argument("--embed_size", default=128, type=int, help="")
    parser.add_argument('--prompt_encoder_type', type=str, default="lstm", choices=['lstm', 'mlp'])
    parser.add_argument("--eval_every_step", default=100, type=int, help="")


    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()


    train_ex_per_label, eval_ex_per_label, dev32_ex_per_label, test_ex_per_label = None, None, None, None
    train_ex, eval_ex, dev32_ex, test_ex = args.train_examples, args.eval_examples, args.dev32_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        eval_ex_per_label = eq_div(args.eval_examples, len(args.label_list)) if args.eval_examples != -1 else -1
        dev32_ex_per_label = eq_div(args.dev32_examples, len(args.label_list)) if args.dev32_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, eval_ex, dev32_ex, test_ex = None, None, None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)

    eval_data = load_examples(
        args.task_name, args.data_dir, eval_set, num_examples=eval_ex, num_examples_per_label=eval_ex_per_label)

    dev32_data = load_examples(
        args.task_name, args.data_dir, DEV32_SET, num_examples=dev32_ex, num_examples_per_label=dev32_ex_per_label)

    if args.task_name in ['yelp', 'enron_spam', 'ag_news']:
        test_data = load_examples(
            args.task_name, args.data_dir, TEST_SET, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
    else:
        test_data = None

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)

    train_pet(eval_data=eval_data,
                  dev32_data=dev32_data,
                  train_data=train_data,
                  test_data=test_data,
                  train_config=pet_train_cfg,
                  eval_config=pet_eval_cfg,
                  model_config=pet_model_cfg,
                  pattern_ids=args.pattern_ids,
                  output_dir=args.output_dir,
                  repetitions=args.pet_repetitions,
                  do_train=args.do_train,
                  do_eval=args.do_eval,
                  do_test_asr=args.do_test_asr,
                  do_defense=args.do_defense,
                  seed=args.seed)

if __name__ == "__main__":
    main()
