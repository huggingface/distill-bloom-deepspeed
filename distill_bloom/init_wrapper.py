import io, json
from pathlib import Path

import torch
import torch.distributed as dist

from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock

class DeepSpeedInitWrapper(object):
    r"""
        This is a wrapper around DeepSpeed inference / training script initialisation. 
        It is used to initialise the DeepSpeed engine and load the necessary variables
        to correctly load the model and run inference.

        Args:
            args (:obj:`argparse.Namespace`):
                The parsed arguments from the command line. This contains all the arguments for 
                training and inference. The `model_path` argument is used to load the model from
                the specified path. 
    """
    def __init__(self, args):
        r"""
            We need to store the rank of the current process since `write_checkpoints` is 
            called only on rank 0.
        """
        self.rank = dist.get_rank()
        self.checkpoints_json = "checkpoints.json"
        self.repo_root = args.teacher_model_path
        self.infer_dtype = "float16"
        
    def init_deepspeed_inference(self):
        r"""
            This function is a wrapper around the first lines that are called inside 
            https://github.com/huggingface/transformers-bloom-inference/blob/main/bloom-inference-scripts/bloom-ds-inference.py 
        """
        tp_presharded_models = [
            "microsoft/bloom-deepspeed-inference-int8",
            "microsoft/bloom-deepspeed-inference-fp16",
        ]
        tp_presharded_mode = True if self.repo_root in tp_presharded_models else False
        

        # use one of these args to `init_inference`
        # 1. injection_policy is the slower version, but it's plain pytorch so it'll always work
        # 2. replace_with_kernel_inject is the faster one (fast fused kernels)
        kernel_inject = True
        # kernel_inject = False

        if kernel_inject:
            # XXX: for now ds-inference only works with fp16
            self.dtype = torch.float16
        else:
            self.dtype = torch.bfloat16

        if kernel_inject:
            self.kwargs = dict(replace_with_kernel_inject=True)
        else:
            self.kwargs = dict(
                injection_policy={BloomBlock: ("self_attention.dense", "mlp.dense_4h_to_h")}
            )

        if tp_presharded_mode:
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(self.repo_root, "ds_inference_config.json")
        else:
            # for normal bloom repo we need to write the checkpoints config file
            if self.rank == 0:
                write_checkponts_json(self.repo_root , self.rank, self.checkpoints_json)
            # dist.barrier()

def print_rank0(*msg, rank=0):
    if rank != 0:
        return
    print(*msg)


def get_checkpoint_files(model_name_or_path, rank=0,revision=None, force_offline=True):
    if not force_offline:
        # checks if online or not
        if is_offline_mode():
            print_rank0("Offline mode: forcing local_files_only=True", rank)
            local_files_only = True
        else:
            local_files_only = False

        # loads files from hub
        cached_repo_dir = snapshot_download(
            model_name_or_path,
            allow_patterns=["*"],
            local_files_only=True,
            revision=revision,
        )
    else:
        cached_repo_dir = model_name_or_path

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [
        str(entry)
        for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]")
        if entry.is_file()
    ]
    return file_list


def write_checkponts_json(model_name, rank=0, checkpoints_json="checkpoints.json"):
    with io.open(checkpoints_json, "w", encoding="utf-8") as f:
        # checkpoint_files = glob.glob(f"{checkpoint_dir}/*bin")
        checkpoint_files = get_checkpoint_files(model_name, rank)

        # print("Checkpoint files:", checkpoint_files)

        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}

        json.dump(data, f)

