# Arguments for distillation

import argparse
import collections
import os
import re
import time

import deepspeed

from .logging import log_levels


def parse_args(extra_args_provider=None, defaults={}, ignore_unknown_args=False):
    r"""
    Helper function to parse all necessarly arguments to perform teacher / student distillation
    """
    parser = argparse.ArgumentParser(description="Main Arguments", allow_abbrev=False)

    # HF model arguments
    parser = _add_hf_model_args(parser)

    # Regularization arguments
    parser = _add_regularization_args(parser)

    # Dataset paths
    parser = _add_data_args(parser)

    # DeepSpeed args
    parser = deepspeed.add_config_arguments(parser)

    # Training args
    parser = _add_training_args(parser)

    # Validation args
    parser = _add_validation_args(parser)

    # Initialization args
    parser = _add_initialization_args(parser)

    # Distriubted args
    parser = _add_distributed_args(parser)

    # Parse args
    args = parser.parse_args()

    if args.data_path:
        assert args.train_weighted_split_paths is None, message
        setattr(args, "valid_weighted_split_names", None)
        setattr(args, "valid_weighted_split_weights", None)
        setattr(args, "valid_weighted_split_splits", None)

        setattr(args, "test_weighted_split_names", None)
        setattr(args, "test_weighted_split_weights", None)
        setattr(args, "test_weighted_split_splits", None)

        # args.split default value in the args is None it is set here in order
        # to check that it does not to overlap with the 2nd mode of data loading
        if args.split is None:
            args.split = "969, 30, 1"

    if (
        args.train_weighted_split_paths
        or args.valid_weighted_split_paths
        or args.test_weighted_split_paths
    ):
        assert args.data_path is None and args.split is None, message

    return args


def _add_hf_model_args(parser, log_levels=log_levels):
    r"""
    A wrapper function to add arguments for loading HF models
    """
    group = parser.add_argument_group(title="network parameters")

    # Teacher & student paths
    group.add_argument(
        "--teacher-model-path", type=str, help="path to load the teacher weights from"
    )
    group.add_argument(
        "--student-model-path", type=str, help="path to load the teacher weights from"
    )

    group.add_argument(
        "--kill-switch-path",
        type=str,
        help=(
            "path to look for a kill switch, which if found will automatically exit the"
            " program"
        ),
    )

    # TODO: assess if we need those arguments in the future
    group.add_argument(
        "--log-level",
        type=str,
        choices=list(log_levels.keys()),
        help=(
            "Logger log level to use on the main process. Possible choices are the log"
            " levels as strings: 'debug', 'info', 'warning', 'error' and 'critical',"
            " plus a 'passive' level which doesn't set anything and lets the"
            " application set the level."
        ),
    )
    group.add_argument(
        "--log-level-replica",
        type=str,
        choices=list(log_levels.keys()),
        help="Logger log level to use on replicas. Same choices as ``log_level``",
    )
    return parser


def _add_regularization_args(parser):
    r"""
    Network regularization arguments - modify them at your own risk
    """
    group = parser.add_argument_group(title="regularization")

    group.add_argument(
        "--attention-dropout",
        type=float,
        default=0.1,
        help="Post attention dropout probability.",
    )
    group.add_argument(
        "--hidden-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for hidden state transformer.",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for L2 regularization.",
    )
    group.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping based on global L2 norm.",
    )
    group.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help=(
            "First coefficient for computing running averages "
            "of gradient and its square"
        ),
    )
    group.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999,
        help=(
            "Second coefficient for computing running averages "
            "of gradient and its square"
        ),
    )
    group.add_argument(
        "--adam-eps",
        type=float,
        default=1e-08,
        help="Term added to the denominator to improvenumerical stability",
    )
    group.add_argument(
        "--sgd-momentum", type=float, default=0.9, help="Momentum factor for sgd"
    )

    return parser


def _add_data_args(parser):
    r"""
    Wrapper function to add arguments for loading data - this function is directly copied from Megatron-DeepSpeed

    """
    group = parser.add_argument_group(title="data and dataloader")

    # option 1 for data loading  (mutually exclusive with option2)
    group.add_argument(
        "--data-path",
        nargs="*",
        default=None,
        help=(
            "Path to the training dataset. Accepted format:"
            "1) a single data path, 2) multiple datasets in the"
            "form: dataset1-weight dataset1-path dataset2-weight "
            "dataset2-path ..."
        ),
    )

    group.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Comma-separated list of proportions for training,"
            " validation, and test split. For example the split "
            "`90,5,5` will use 90%% of data for training, 5%% for "
            "validation and 5%% for test."
        ),
    )

    # option 2 for data loading (mutually exclusive with option1)

    # helper class to parse the --xxx-weighted-split-paths
    # note here two args are set: extra valid dataset paths and names
    class parse_data_paths(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if option_string == "--train-weighted-split-paths":
                assert len(values) == 1, "Only 1 dataset group is allowed to"
                "be passed for the argument --train-weighted-split-paths"

            # make sure string given in the correct format
            err_message = "Each data group should be input on the following format"
            '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
            "where START < END"
            for v in values:
                # each prefix consists several datasets separated by commas
                prefix = ":".join(v.split(":")[1:])  # remove GIVEN_NAME
                datasets = prefix.split(",")
                # check if each dataset is formatted like `WEIGHT START:END PATH`
                for d in datasets:
                    assert len(d.split()) == 3, err_message
                    start, end = d.split()[1].split(":")
                    assert float(start) < float(end), err_message

            names = [v.split(":")[0] for v in values]

            prefixes = [":".join(v.split(":")[1:]).strip() for v in values]
            weights = [[d.split()[0] for d in p.split(",")] for p in prefixes]
            splits = [[d.split()[1] for d in p.split(",")] for p in prefixes]
            paths = [[d.split()[2] for d in p.split(",")] for p in prefixes]

            # # to keep consistency with Option 1 of data loading (through --data-path)
            # #  paths will contain strings on the following form
            # # "WEIGHTS1 PATH1 WEIGHTS2 PATH2 WEIGHTS3 PATH3" for each dataset group
            # # while data will be parsed in additional arguments below
            # paths_option1_style = []
            # for p, w in zip(paths, weights):
            #   paths_option1_style.append(" ".join([f"{w_i} {p_i}" for p_i, w_i in zip(p,w)]))
            # setattr(args, self.dest, paths_option1_style)
            setattr(args, self.dest, paths)
            setattr(args, self.dest.replace("paths", "weights"), weights)
            setattr(args, self.dest.replace("paths", "splits"), splits)
            setattr(args, self.dest.replace("paths", "names"), names)

    group.add_argument(
        "--train-weighted-split-paths",
        nargs="*",
        default=None,
        help=(
            "Weights, splits and paths to groups of datasets"
            "Accepted format: ONE dataset groups could be"
            "submitted in the following form between double quotes"
            '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
            'e.g.: "NAME_ABC: 0.6 0:0.6 A, 0.3 0:1 B, 0.1 0:1 C" '
            "WEIGHT is used to up and down sample each dataset A,B,C in the group"
            "START:END indicates the split portion of the dataset"
        ),
        action=parse_data_paths,
    )

    group.add_argument(
        "--valid-weighted-split-paths",
        nargs="*",
        default=None,
        help=(
            "Weights, splits and paths to groups of datasets"
            "Accepted format: one or many dataset groups could be"
            "submitted in the following form each between double quotes"
            '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
            'e.g.: "NAME_ABC: 0.6 0.6:0.8 A, 0.3 0:1 B, 0.1 0:1 C" '
            '"NAME_CDE: 0.6 0.6:0.8 C, 0.3 0:1 D, 0.1 0:1 E" '
            "validation will be run on each of those groups independently"
        ),
        action=parse_data_paths,
    )

    group.add_argument(
        "--test-weighted-split-paths",
        nargs="*",
        default=None,
        help=(
            "Weights, splits and paths to groups of datasets"
            "Accepted format: one or many dataset groups could be"
            "submitted in the following form each between double quotes"
            '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
            'e.g.: "NAME_ABC: 0.6 0.6:0.8 A, 0.3 0:1 B, 0.1 0:1 C" '
            '"NAME_CDE: 0.6 0.6:0.8 C, 0.3 0:1 D, 0.1 0:1 E" '
            "test will be run on each of those groups independently"
        ),
        action=parse_data_paths,
    )

    class parse_data_paths_path(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            expected_option_strings = [
                "--train-weighted-split-paths-path",
                "--valid-weighted-split-paths-path",
                "--test-weighted-split-paths-path",
            ]
            assert (
                option_string in expected_option_strings
            ), f"Expected {option_string} to be in {expected_option_strings}"

            with open(values, "r") as fi:
                lines = fi.readlines()
                assert (
                    len(lines) == 1
                ), f"Got multiple lines {len(lines)} instead of 1 expected"
                assert (
                    lines[0][-2:] == '"\n' and lines[0][0] == '"'
                ), f"Invalid input format, got {lines}"
                values = lines[0][1:-2].split('" "')
                weighted_split_paths_dest = re.sub(r"_path$", "", self.dest)
                weighted_split_paths_option = re.sub(
                    r"-path$", "", self.option_strings[0]
                )
                setattr(args, weighted_split_paths_dest, values)
                parse_data_paths(
                    option_strings=[weighted_split_paths_option],
                    dest=weighted_split_paths_dest,
                )(parser, args, values, option_string=weighted_split_paths_option)

    group.add_argument(
        "--train-weighted-split-paths-path",
        type=str,
        action=parse_data_paths_path,
        default=None,
    )
    group.add_argument(
        "--valid-weighted-split-paths-path",
        type=str,
        action=parse_data_paths_path,
        default=None,
    )
    group.add_argument(
        "--test-weighted-split-paths-path",
        type=str,
        action=parse_data_paths_path,
        default=None,
    )

    group.add_argument(
        "--log-path", type=str, default=None, help="Path to the save arguments file."
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file."
    )
    group.add_argument(
        "--merge-file", type=str, default=None, help="Path to the BPE merge file."
    )
    group.add_argument(
        "--vocab-extra-ids",
        type=int,
        default=0,
        help=(
            "Number of additional vocabulary tokens. "
            "They are used for span masking in the T5 model"
        ),
    )
    group.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Maximum sequence length to process.",
    )
    group.add_argument(
        "--encoder-seq-length",
        type=int,
        default=None,
        help=(
            "Maximum encoder sequence length to process."
            "This should be exclusive of --seq-length"
        ),
    )
    group.add_argument(
        "--decoder-seq-length",
        type=int,
        default=None,
        help="Maximum decoder sequence length to process.",
    )
    group.add_argument(
        "--retriever-seq-length",
        type=int,
        default=256,
        help="Maximum sequence length for the biencoder model  for retriever",
    )
    group.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="sample rate for training data. Supposed to be 0  < sample_rate < 1",
    )
    group.add_argument(
        "--mask-prob",
        type=float,
        default=0.15,
        help="Probability of replacing a token with mask.",
    )
    group.add_argument(
        "--short-seq-prob",
        type=float,
        default=0.1,
        help="Probability of producing a short sequence.",
    )
    group.add_argument("--mmap-warmup", action="store_true", help="Warm up mmap files.")
    group.add_argument(
        "--num-workers", type=int, default=2, help="Dataloader number of workers."
    )
    group.add_argument(
        "--valid-num-workers",
        type=int,
        default=2,
        help="Dataloader number of workers for validation.",
    )
    group.add_argument(
        "--tokenizer-type",
        type=str,
        default=None,
        choices=[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "PretrainedFromHF",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help="Name or path of the huggingface tokenizer.",
    )
    group.add_argument(
        "--data-impl",
        type=str,
        default="infer",
        choices=["lazy", "cached", "mmap", "infer"],
        help="Implementation of indexed datasets.",
    )
    group.add_argument(
        "--reset-position-ids",
        action="store_true",
        help="Reset posistion ids after end-of-document token.",
    )
    group.add_argument(
        "--reset-attention-mask",
        action="store_true",
        help=(
            "Reset self attention maske after end-of-document token. Attention between"
            " tokens from different documents is null."
        ),
    )
    group.add_argument(
        "--eod-mask-loss",
        action="store_true",
        help="Mask loss for the end of document tokens.",
    )
    group.add_argument(
        "--loss-on-targets-only",
        action="store_true",
        help="Mask loss on input sequence.",
    )
    group.add_argument(
        "--reweight-loss-based-on-position-frequency",
        action="store_true",
        help=(
            "Some objectives require us to sample loss_mask. This might introduce bias"
            " towards specific positions. This option tries to un-bias the loss by"
            " reweighting loss on specific positions based on how frequently we train"
            " on that position.This is mostly used for prefix_lm training"
        ),
    )
    group.add_argument(
        "--noise-density",
        type=float,
        default=None,
        help="Span corruption noise density",
    )
    group.add_argument(
        "--mean-noise-span-length",
        type=int,
        default=None,
        help="Span corruption mean noise span length",
    )

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")

    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size per model instance (local batch size). "
            "Global batch size is local batch size times data "
            "parallel size times number of micro batches."
        ),
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Old batch size parameter, do not use. Use --micro-batch-size instead",
    )
    group.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help=(
            "Training batch size. If set, it should be a "
            "multiple of micro-batch-size times data-parallel-size. "
            "If this value is None, then "
            "use micro-batch-size * data-parallel-size as the "
            "global batch size. This choice will result in 1 for "
            "number of micro-batches."
        ),
    )
    group.add_argument(
        "--rampup-batch-size",
        nargs="*",
        default=None,
        help=(
            "Batch size ramp up with the following values:"
            "  --rampup-batch-size <start batch size> "
            "                      <batch size increment> "
            "                      <ramp-up samples> "
            "For example: "
            "   --rampup-batch-size 16 8 300000 "
            "   --global-batch-size 1024 "
            "will start with global batch size 16 and over "
            " (1024 - 16) / 8 = 126 intervals will increase "
            "the batch size linearly to 1024. In each interval "
            "we will use approximately 300000 / 126 = 2380 samples."
        ),
    )
    group.add_argument(
        "--checkpoint-activations",
        action="store_true",
        help=(
            "Checkpoint activation to allow for training "
            "with larger models, sequences, and batch sizes."
        ),
    )
    group.add_argument(
        "--distribute-checkpointed-activations",
        action="store_true",
        help="If set, distribute checkpointed activations across model parallel group.",
    )
    group.add_argument(
        "--checkpoint-num-layers",
        type=int,
        default=1,
        help="chunk size (number of layers) for checkpointing.",
    )
    group.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help=(
            "Total number of iterations to train over all "
            "training runs. Note that either train-iters or "
            "train-samples should be provided."
        ),
    )
    group.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help=(
            "Total number of samples to train over all "
            "training runs. Note that either train-iters or "
            "train-samples should be provided."
        ),
    )
    group.add_argument(
        "--train-tokens",
        type=int,
        default=None,
        help="Total number of tokens to train over all training runs.",
    )
    group.add_argument(
        "--log-interval", type=int, default=100, help="Report loss and timing interval."
    )
    group.add_argument(
        "--exit-interval",
        type=int,
        default=None,
        help="Exit the program after the iteration is divisible by this value.",
    )
    group.add_argument(
        "--exit-duration-in-mins",
        type=int,
        default=None,
        help="Exit the program after this many minutes.",
    )
    group.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Write TensorBoard logs to this directory.",
    )
    group.add_argument(
        "--no-masked-softmax-fusion",
        action="store_false",
        help="Disable fusion of query_key_value scaling, masking, and softmax.",
        dest="masked_softmax_fusion",
    )
    group.add_argument(
        "--no-bias-gelu-fusion",
        action="store_false",
        help="Disable bias and gelu fusion.",
        dest="bias_gelu_fusion",
    )
    group.add_argument(
        "--no-bias-dropout-fusion",
        action="store_false",
        help="Disable bias and dropout fusion.",
        dest="bias_dropout_fusion",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer function",
    )
    group.add_argument(
        "--use-bnb-optimizer",
        action="store_true",
        help=(
            "Use bitsandbytes optimizer for efficient training,"
            "please refer https://github.com/facebookresearch/bitsandbytes."
        ),
        dest="use_bnb_optimizer",
    )
    group.add_argument(
        "--dataloader-type",
        type=str,
        default=None,
        choices=["single", "cyclic"],
        help="Single pass vs multiple pass data loader",
    )
    group.add_argument(
        "--cpu-optimizer", action="store_true", help="Run optimizer on CPU"
    )
    group.add_argument(
        "--cpu_torch_adam",
        action="store_true",
        help="Use Torch Adam as optimizer on CPU.",
    )
    group.add_argument(
        "--codecarbon-dir",
        type=str,
        default=None,
        help="Write CodeCarbon logs to this directory.",
    )
    group.add_argument(
        "--eval-only",
        type=bool,
        required=False,
        help=(
            "If set to True, no train step will be performed."
            "and only the evaluation on the `valid` and `test` sets "
            "will be performed"
        ),
    )
    group.add_argument(
        "--skip-train-iteration-range",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Iteration ranges to skip. The values are one or more dash-separated"
            " ranges. e.g., 101-200 251-300."
        ),
    )
    group.add_argument(
        "--inference",
        action="store_true",
        help=(
            "Very basic inference mode: not allocating optim/lr - requires ZERO_STAGE=0"
        ),
    )
    group.add_argument(
        "--abort-on-unmet-fused-kernel-constraints",
        action="store_true",
        help=(
            "If set to True, the program will abort if the constraints for loading a"
            " fused kernel aren't met"
        ),
    )
    group.add_argument(
        "--pp-partition-method",
        type=str,
        default=None,
        help=(
            "Use to override the pipeline stages partitioning method. e.g.,"
            " 'type:transformer|embedding'"
        ),
    )

    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title="validation")

    group.add_argument(
        "--eval-iters",
        type=int,
        default=100,
        help="Number of iterations to run for evaluationvalidation/test for.",
    )
    group.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Interval between running evaluation on validation set.",
    )

    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title="initialization")

    group.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for python, numpy, pytorch, and cuda.",
    )
    group.add_argument(
        "--init-method-std",
        type=float,
        default=0.02,
        help=(
            "Standard deviation of the zero mean normal "
            "distribution used for weight initialization."
        ),
    )
    group.add_argument(
        "--init-method-xavier-uniform",
        action="store_true",
        help="Enable Xavier uniform parameter initialization",
    )

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title="distributed")

    group.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Degree of tensor model parallelism.",
    )
    group.add_argument(
        "--student-tensor-model-parallel-size",
        type=int,
        default=1,
        help="Degree of tensor model parallelism.",
    )
    group.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Degree of pipeline model parallelism.",
    )
    group.add_argument(
        "--student-pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Degree of pipeline model parallelism.",
    )
    group.add_argument(
        "--model-parallel-size",
        type=int,
        default=None,
        help=(
            "Old model parallel argument, do not use. Use "
            "--tensor-model-parallel-size instead."
        ),
    )
    group.add_argument(
        "--num-layers-per-virtual-pipeline-stage",
        type=int,
        default=None,
        help="Number of layers per virtual pipeline stage",
    )
    group.add_argument(
        "--distributed-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="Which backend to use for distributed training.",
    )
    group.add_argument(
        "--DDP-impl",
        default="local",
        choices=["local", "torch"],
        help="which DistributedDataParallel implementation to use.",
    )
    group.add_argument(
        "--use-contiguous-buffers-in-ddp",
        action="store_true",
        help=(
            "If set, use contiguous buffer in DDP. Note that "
            "this option only works woth local DDP."
        ),
    )
    group.add_argument(
        "--no-scatter-gather-tensors-in-pipeline",
        action="store_false",
        help="Use scatter/gather to optimize communication of tensors in pipeline",
        dest="scatter_gather_tensors_in_pipeline",
    )
    group.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher.",
    )
    group.add_argument(
        "--lazy-mpu-init",
        type=bool,
        required=False,
        help=(
            "If set to True, initialize_megatron() "
            "skips DDP initialization and returns function to "
            "complete it instead.Also turns on "
            "--use-cpu-initialization flag. This is for "
            "external DDP manager."
        ),
    )
    group.add_argument(
        "--use-cpu-initialization",
        action="store_true",
        default=None,
        help="If set, affine parallel weights initialization uses CPU",
    )
    return parser
