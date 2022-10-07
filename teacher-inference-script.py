# usage:
# deepspeed --num_gpus 8 teacher-inference-script.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 teacher-inference-script.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#
import gc
import glob
import io
import json
import math
import os
import time
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode

from distill_bloom import build_train_val_test_dataset, DistributedDataset, DistributedDataLoader
from distill_bloom import parse_args, DeepSpeedInitWrapper, print_rank0

# Arguments

args = parse_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1")) # World size is the number of GPUs

deepspeed.init_distributed("nccl")

rank = dist.get_rank()

## Check the args

assert (world_size % args.global_batch_size) == 0, "micro_batch_size must be divisible by num_gpus"

ds_init = DeepSpeedInitWrapper(args)
ds_init.init_deepspeed_inference()
model_name = ds_init.repo_root

# Wait that all processes have correctly initiliazed DeepSpeed
dist.barrier()


print_rank0(f"*** Loading the model {model_name}")
config = AutoConfig.from_pretrained(model_name)

# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=ds_init.dtype, device="meta"):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
model = model.eval()

# checkpoints_json=None
model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    base_dir=ds_init.repo_root,
    dtype=getattr(torch, ds_init.infer_dtype),
    checkpoint=ds_init.checkpoints_json,
    **ds_init.kwargs,
)
model = model.module

# Dataset building - each rank will have a different shard of the dataset
train_ds, _, _ = build_train_val_test_dataset(args)
data_loader = DistributedDataLoader(
    train_ds,
    rank=rank,
    world_size=world_size,
    batch_size=1
)
dist.barrier()

def generate_logits(inputs):
    """returns a list of zipped inputs, outputs and number of new tokens"""
    inputs = inputs.to(torch.cuda.current_device())
    outputs = model(inputs).logits

    return outputs

def generate_logits_batch(data_loader):
    for batch in data_loader:
        inputs = batch['text']
        # as a sanity check, I used to check that inputs are different for each rank
        inputs = inputs.to(torch.cuda.current_device())
        outputs = model(inputs).logits
        return outputs


# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
print_rank0(f"*** Running generate warmup")
# _ = generate_logits(inputs)
_ = generate_logits_batch(data_loader)

print_rank0(f"*** Running generate")
t_generate_start = time.time()
# generated = generate_logits(inputs)
generated = generate_logits_batch(data_loader)
print(rank, generated.shape)
t_generate_span = time.time() - t_generate_start
exit()