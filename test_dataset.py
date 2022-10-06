import os
import deepspeed

import torch.distributed as dist

from distill_bloom import build_train_val_test_dataset
from distill_bloom import parse_args


args = parse_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")

rank = dist.get_rank()

if rank == 0:
    train_ds, val, test = build_train_val_test_dataset(args)
    print(f"The total dataset includes: {len(train_ds)} subsets")
    for i, train_data in enumerate(train_ds):
        print(f"Train dataset: {i} has {len(train_data)} samples")
        for data in train_data:
            print("Text: ", data['text'])
            break