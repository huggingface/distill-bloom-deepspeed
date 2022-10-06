import torch.distributed as dist

from .utils import build_dataset_group


def build_train_val_test_dataset(args):
    r"""
    This function wraps all the dataset building functions from megatron.

    """
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    ]

    train_ds, valid_ds, test_ds = None, None, None

    print("> building train, validation, and test datasets for GPT ...")
    # Option 1 of data loading using --data-path

    if args.data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
        )
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(
                eval(f"args.{s}_weighted_split_paths"),
                eval(f"args.{s}_weighted_split_weights"),
                eval(f"args.{s}_weighted_split_splits"),
                eval(f"args.{s}_weighted_split_names"),
            )
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(
                    name,
                    paths,
                    weights,
                    splits,
                    args.data_impl,
                    train_val_test_num_samples,
                    args.seq_length,
                    args.seed,
                    (not args.mmap_warmup),
                    train_valid_test=s,
                )
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds
