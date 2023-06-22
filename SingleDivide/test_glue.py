# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import numpy as np
import os
import random
from pathlib import Path
import time
from torch.cuda.amp import autocast, GradScaler

from copy import deepcopy
import evaluate
import datasets
import torch
from datasets import load_dataset, load_metric, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime

from copy import deepcopy, copy
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    AutoConfig,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import transformersLocal
from transformers.utils import get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import matplotlib.pyplot as plt

# import sys
# sys.path.append("..")
# from .. import modeling_bert as bertmodels
import transformersLocal.models.bert.modeling_bert as bertmodels


# try:
#     from apex.parallel import DistributedDataParallel as DDP
#     from apex.fp16_utils import *
#     from apex import amp
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument('--arch', '-a', metavar='ARCH', default='bertForSequence',
                        choices=["BertForSequenceClassification"],
                        )

    parser.add_argument('--model-config', '-c', metavar='CONF', default='classic', choices=["classic", "quantize"])

    # 选择哪个层进行量化
    # classic(都不量化), embedding(填充层), attention, addNorm, feedForward, pooler, classifier, linear(量化全部线性层), quantize(以上所有)

    parser.add_argument('--choice', nargs='+', type=str, help='Choose a linear layer to quantize')
    parser.add_argument('--clip_lr', default=2e-4, type=float, help='Use a seperate lr for clip_vals / stepsize')
    parser.add_argument('--clip_wd', default=0.0, type=float, help='weight decay for clip_vals / stepsize')

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--qa', type=str2bool, default=True, help='quantize activation')
    parser.add_argument('--qw', type=str2bool, default=True, help='quantize weights')
    parser.add_argument('--qg', type=str2bool, default=True, help='quantize gradients')
    parser.add_argument('--biased', type=str2bool, default=False, help='biased quantization')
    parser.add_argument('--abits', type=int, default=8, help='activation number of bits')
    parser.add_argument('--wbits', type=int, default=8, help='weight number of bits')
    parser.add_argument('--biasbits', type=int, default=16, help='bias number of bits')
    parser.add_argument('--bbits', type=int, default=8, help='backward number of bits')
    parser.add_argument('--bwbits', type=int, default=8, help='backward weight number of bits')

    parser.add_argument('--hadamard', type=str2bool, default=False,
                        help='apply Hadamard transformation on gradients')
    parser.add_argument('--dynamic', type=str2bool, default=True,
                        help='whether apply dynamic Hadamard transformation on gradients')
    parser.add_argument('--bmm', type=str2bool, default=True,
                        help='whether apply bmm Hadamard transformation on gradients')
    parser.add_argument('--biprecision', type=str2bool, default=True, help='Gradient bifurcation')
    parser.add_argument('--twolayers_gradweight', '--2gw', type=str2bool, default=False,
                        help='use two 4 bit to simulate a 8 bit')
    parser.add_argument('--twolayers_gradinputt', '--2gi', type=str2bool, default=False,
                        help='use two 4 bit to simulate a 8 bit')
    parser.add_argument('--luq', type=str2bool, default=False, help='use luq for backward')
    parser.add_argument('--weight_quant_method', '--wfq', default='ptq', type=str, metavar='strategy',
                        choices=['uniform', 'lsq', 'ptq'])
    parser.add_argument('--input_quant_method', '--ifq', default='ptq', type=str, metavar='strategy',
                        choices=['uniform', 'lsq', 'ptq'])
    parser.add_argument('--learnable_step_size', type=str2bool, default=True,
                        help='Debug to draw the variance and leverage score')
    parser.add_argument('--learnable_hadamard', type=str2bool, default=True,
                        help='Debug to draw the variance and leverage score')

    parser.add_argument('--lsq_layerwise_input', type=str, default='layer',
                        help='Debug to draw the variance and leverage score', choices=['layer', 'row', 'column'])
    parser.add_argument('--lsq_layerwise_weight', type=str, default='layer',
                        help='Debug to draw the variance and leverage score', choices=['layer', 'row', 'column'])
    parser.add_argument('--retain_large_value', type=str2bool, default=False,
                        help='Debug to draw the variance and leverage score')
    parser.add_argument('--quantize_large_value', type=str2bool, default=False,
                        help='Debug to draw the variance and leverage score')
    parser.add_argument('--draw_value', type=str2bool, default=False,
                        help='Debug to draw the variance and leverage score')
    parser.add_argument('--clip-value', type=float, default=100, help='Choose a linear layer to quantize')

    parser.add_argument('--track_step_size', type=str2bool, default=False,
                        help='Debug to draw the variance and leverage score')
    
    parser.add_argument('--fp16', type=str2bool, default=False,
                    help='whether use torch amp')

    # Todo:添加参数部分到此结束
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    if "bert" in args.model_name_or_path:
        from transformersLocal.models.bert.image_classification.quantize import qconfig, QLinear

    qconfig.hadamard = args.hadamard
    qconfig.dynamic = args.dynamic
    qconfig.bmm = args.bmm
    qconfig.track_step_size = args.track_step_size
    qconfig.weight_quant_method = args.weight_quant_method
    qconfig.input_quant_method = args.input_quant_method
    qconfig.lsq_layerwise_input = args.lsq_layerwise_input
    qconfig.lsq_layerwise_weight = args.lsq_layerwise_weight
    qconfig.learnable_step_size = args.learnable_step_size
    qconfig.learnable_hadamard = args.learnable_hadamard

    qconfig.quantize_activation = args.qa
    qconfig.quantize_weights = args.qw
    qconfig.quantize_gradient = args.qg
    qconfig.activation_num_bits = args.abits
    qconfig.weight_num_bits = args.wbits
    qconfig.bias_num_bits = args.biasbits
    qconfig.backward_num_bits = args.bbits
    qconfig.bweight_num_bits = args.bwbits
    qconfig.hadamard = args.hadamard
    qconfig.biased = args.biased
    qconfig.biprecision = args.biprecision
    qconfig.twolayers_gradweight = args.twolayers_gradweight
    qconfig.twolayers_gradinputt = args.twolayers_gradinputt
    qconfig.luq = args.luq
    qconfig.clip_value = args.clip_value
    qconfig.choice = args.choice

    qconfig.weight_quant_method = args.weight_quant_method
    qconfig.input_quant_method = args.input_quant_method
    qconfig.learnable_step_size = args.learnable_step_size
    qconfig.learnable_hadamard = args.learnable_hadamard
    qconfig.lsq_layerwise_input = args.lsq_layerwise_input
    qconfig.lsq_layerwise_weight = args.lsq_layerwise_weight
    qconfig.retain_large_value = args.retain_large_value
    qconfig.quantize_large_value = args.quantize_large_value
    qconfig.draw_value = args.draw_value

    qconfig.track_step_size = args.track_step_size
    qconfig.fp16 = args.fp16

    # Todo:end of add something
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("test_glue", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
        # raw_datasets = load_from_disk(f"./data/glue/{args.task_name}")
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # print("task_name is:",args.task_name)
    # if args.task_name == "mnli":
    #     num_labels = 2
    # elif args.task_name == "stsb":
    #     num_labels = 2
    print("num_labels is:", num_labels)
    pretrainedConfig = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels,
                                                  finetuning_task=args.task_name)
    config = transformersLocal.PretrainedConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels,
                                                                finetuning_task=args.task_name)
    if "bert" in args.model_name_or_path:
        config.classifier_dropout = pretrainedConfig.classifier_dropout
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    # Todo:在training.py文件写完之后应用对应的模型

    PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(  # Todo:修改模型
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    print("pretrainedModel.num_label is:", PreTrainedModel.num_labels)

    pretrained_dict = PreTrainedModel.state_dict()

    print("*" * 100, args.choice, "*" * 100)

    if "bert" in args.model_name_or_path:
        model = bertmodels.build_bert_for_sequencePrecision(args.arch, args.model_config, args.choice,
                                                            bertConfig=config)

    print("model.num_label is:", model.num_labels)

    model_dict = model.state_dict()
    pretrained_dict_part = {key: value for key, value in pretrained_dict.items() if
                            (key in model_dict and 'classifier' not in key and 'score' not in key)}

    model.load_state_dict(pretrained_dict_part, strict=False)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    clip_params = {}
    for k, v in param_optimizer:
        # print(k)
        if 'clip_' in k:
            clip_params[k] = v

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and not 'clip_' in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not 'clip_' in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in clip_params.items()],
            "lr": args.clip_lr,
            "weight_decay": args.clip_wd,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        # metric = load_metric("glue", args.task_name)
        metric = evaluate.load("glue", args.task_name)
    else:
        # metric = load_metric("accuracy")
        metric = evaluate.load("accuracy")

    if args.fp16:
        scaler = GradScaler()
    # Train!
    for name, module in model.named_modules():
        if isinstance(module, QLinear):
            module.name_draw_clip_value = name
            
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    file_to_delete = open(os.path.join(args.output_dir, "acc.txt"), "w")
    file_to_delete.close()
    with open(os.path.join(args.output_dir, "acc.txt"), "a") as f:
        time_tuple = time.localtime(time.time())
        print('Time {}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}:'
              .format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                      time_tuple[4], time_tuple[5]), file=f)
    for arg in vars(args):
        print(arg, getattr(args, arg))
        with open(os.path.join(args.output_dir, "acc.txt"), "a") as f:
            print(arg, getattr(args, arg), file=f)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            # accelerator.load_state(args.resume_from_checkpoint)
            pretrained_dict = torch.load(os.path.join(args.resume_from_checkpoint, 'pytorch_model.bin'))
            model_dict = model.state_dict()
            pretrained_dict_part = {key: value for key, value in pretrained_dict.items() if
                                    (key in model_dict)}
            model.load_state_dict(pretrained_dict_part, strict=False)

            path = os.path.basename(args.resume_from_checkpoint)
            starting_epoch = 0
            resume_step = None
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

    lr_plt = []
    train_loss_x_plt = []
    train_loss_y_plt = []
    valid_loss_x_plt = []
    valid_loss_y_plt = []

    # TODO:add something
    # init(args.max_length * args.per_device_train_batch_size)
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            if args.fp16:
                with autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

        print("initialize LSQ and Hadamard finish")
        break
    
    train_time = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            torch.cuda.synchronize()
            epoch_start_time = time.time()
            if args.fp16:
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                accelerator.backward(scaler.scale(loss), retain_graph=True)
            else:
                accelerator.backward(loss, retain_graph=True)
            
            torch.cuda.synchronize()
            epoch_end_time = time.time()
            train_time += epoch_end_time - epoch_start_time

            train_loss_x_plt.append(epoch + step / len(train_dataloader))
            train_loss_y_plt.append(loss.detach().cpu().numpy().item())

            # for name, param in model.named_parameters():
            #     print(name)
            #     if "clip" in name:
            #         print(name, param.grad)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                lr_plt.append(lr_scheduler.get_last_lr()[0])
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

            if args.draw_value:
                exit(0)

        model.eval()
        samples_seen = 0
        valid_loss = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                if args.fp16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(**batch)
                        valid_loss.append(outputs.loss.detach().cpu().numpy().item())
                    
                else:
                    outputs = model(**batch)
                    valid_loss.append(outputs.loss.detach().cpu().numpy().item())
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        valid_loss_x_plt.append(epoch + 1)
        valid_loss_y_plt.append(sum(valid_loss) / len(valid_loss))

        eval_metric = metric.compute()
        # print("what the fuck")
        time_tuple = time.localtime(time.time())
        logger.info(f"epoch {epoch}: {eval_metric}" +
                    " Time {}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}:".format(time_tuple[0], time_tuple[1], time_tuple[2],
                                                                          time_tuple[3],
                                                                          time_tuple[4], time_tuple[5]))

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.track_step_size:
            from transformersLocal.models.bert.image_classification.quantize import QLinear, QIdentity
            l = {name: module for name, module in model.named_modules() if isinstance(module, QLinear)}
            for name, layer in l.items():
                plt.figure()
                plt.title("{}\n{}".format(layer.active_track[0], layer.active_track[-1]))
                plt.plot(layer.iter_track, layer.active_track)
                plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

                os.makedirs("plt/step_size/input", exist_ok=True)
                plt.savefig("plt/step_size/input/{}.png".format(name))
                plt.close()

                plt.figure()
                plt.title("{}\n{}".format(layer.weight_track[0], layer.weight_track[-1]))
                plt.plot(layer.iter_track, layer.weight_track)
                plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

                os.makedirs("plt/step_size/weight", exist_ok=True)
                plt.savefig("plt/step_size/weight/{}.png".format(name))
                plt.close()
            print("track step size")

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        plt.figure(0)
        lr_x_plt = torch.linspace(starting_epoch, epoch + 1, steps=len(lr_plt))
        plt.xlim(0, args.num_train_epochs)
        plt.ylim(-0.2 * args.learning_rate, 1.2 * args.learning_rate)
        # print(lr_x_plt, lr_plt)
        plt.scatter(lr_x_plt, lr_plt, s=1)
        plt.savefig(os.path.join(args.output_dir, "lr.png"))

        plt.figure(1)
        plt.scatter(train_loss_x_plt, train_loss_y_plt, s=1, c='r', label='train')
        plt.scatter(valid_loss_x_plt, valid_loss_y_plt, s=5, c='b', label='valid')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "loss.png"))

        with open(os.path.join(args.output_dir, "loss.json"), "w") as f:
            Dict = {}
            Dict['train_loss'] = train_loss_y_plt
            Dict['valid_loss'] = valid_loss_y_plt
            obj_str1 = json.dumps(Dict)
            f.write(f'{obj_str1}\n')
        with open(os.path.join(args.output_dir, "acc.txt"), "a") as f:
            f.write(f"epoch {epoch}: {eval_metric}\n")


    total_flop = sum(p.numel() for p in model.parameters())
    total_flop_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")


    with open(os.path.join(args.output_dir, "acc.txt"), "a") as f:
        print("training time", format(train_time, '.3f'), file=f)
        
    # with open(os.path.join(args.output_dir, "time.json"), "w") as f:
    #     Dict = {}
    #     Dict["forward"] = qconfig.forward
    #     Dict["backward"] = qconfig.backward
    #     # Dict["hadamard"] = qconfig.hadamard_time
    #     # Dict["special"] = qconfig.special_layer_time
    #     # Dict["abs"] = qconfig.abs_time
    #     Dict["forward_all"] = qconfig.forward_all
    #     Dict["backward_all"] = qconfig.backward_all
    #     Dict["test"] = qconfig.forward_test
    #     Dict["test_calculate"] = qconfig.forward_calculate
    #     full_time_list = []
    #     iterate_key = ["forward", "backward"]
    #     for keys in iterate_key:
    #         fullTime = 0
    #         for timesKey in Dict[keys].keys():
    #             fullTime += Dict[keys][timesKey]
    #         Dict[keys]["full"] = fullTime
    #         full_time_list.append(fullTime)
    #     json.dump(Dict, f, indent=4)
        
    # Todo:日志里添加训练时间以及模型本身大小
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            training_time = train_time
            obj_str = json.dumps(eval_metric)
            f.write(f'{obj_str}\n')
            Dict = {}
            Dict['training_time'] = training_time
            Dict['flop'] = total_flop
            Dict['flop_learn'] = total_flop_learn
            obj_str2 = json.dumps(Dict)
            f.write(f'{obj_str2}\n')
            dict_time = {}
            dt = datetime.now()
            nowtime = f'{dt.year}.{dt.month}.{dt.day} {dt.hour}:{dt.minute}:{dt.second}'
            dict_time["time"] = nowtime
            obj_str3 = json.dumps(dict_time)
            f.write(f'{obj_str3}\n')


if __name__ == "__main__":
    main()
