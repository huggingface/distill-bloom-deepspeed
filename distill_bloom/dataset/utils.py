import time

import numpy as np
import torch

from .gpt_dataset import GPTDataset
from .indexed_dataset import (IndexedDataset, MMapIndexedDataset,
                              create_doc_idx, data_file_path, index_file_path)


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return "cached"
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return "mmap"
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")
        print(
            "Path should be a basename that both .idx and .bin can be appended to get"
            " full filenames."
        )
        return None


def get_train_valid_test_split_(splits_string, size):
    r"""
    Get dataset splits from comma or '/' separated string list.
    `splits_string` expects an string of 3 sets of integers, summing up to `1000`.

    Returns:
        The proportion of the dataset to be used for training, validation and testing.
    """
    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def analyze_data_prefix(data_prefix):
    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2 * i])
        prefixes[i] = (data_prefix[2 * i + 1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    return prefixes, weights


def get_split_by_range_(range_string, size):
    """Get dataset splits based on a range:
    range_string is in the form  START%:END%  for e.g. 0.2:0.8
    outputs an array of two values [start_index, end_index]
    """
    # some checks that range is given in the correct form
    splits = [float(i) for i in range_string.split(":")]
    assert len(splits) == 2, "splits should be passed as start:end"
    assert splits[0] <= 1 and splits[1] <= 1
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits_index = [round(s * float(size)) for s in splits]
    assert len(splits_index) == 2
    return splits_index


def get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples):
    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    prefixes, weights = analyze_data_prefix(data_prefix)
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        datasets_train_valid_test_num_samples.append(
            [
                int(math.ceil(val * weight * 1.005))
                for val in train_valid_test_num_samples
            ]
        )

    return prefixes, weights, datasets_train_valid_test_num_samples


def build_dataset_group(
    dataset_group_name,
    paths,
    weights,
    splits,
    data_impl,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    train_valid_test,
):
    """
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed on the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    """

    assert train_valid_test in ["train", "valid", "test"]

    # Single dataset.
    if len(paths) == 1:
        dataset = _build_single_datasets(
            paths[0],
            splits[0],
            data_impl,
            train_valid_test_num_samples,
            seq_length,
            seed,
            skip_warmup,
            dataset_group_name,
            train_valid_test,
        )
        return dataset
    # Blending dataset.
    else:
        data_prefix = []
        # data_prefix is on the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w, p in zip(weights, paths):
            data_prefix += [w, p]

        output = get_datasets_weights_and_num_samples(
            data_prefix, train_valid_test_num_samples
        )
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(
                prefixes[i],
                splits[i],
                data_impl,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
                dataset_group_name,
                train_valid_test,
            )

            datasets.append(ds)
        all_datasets = BlendableDataset(datasets, weights)

        return all_datasets


def make_dataset(path, impl, skip_warmup=False):
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print(
            "Path should be a basename that both .idx and .bin can be appended to get"
            " full filenames."
        )
        return None
    if impl == "infer":
        impl = infer_dataset_impl(path)
    if impl == "lazy" and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == "cached" and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == "mmap" and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def get_indexed_dataset_(path, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(" > building dataset index ...")
    start_time = time.time()
    indexed_dataset = make_dataset(path, data_impl, skip_warmup)
    print_rank_0(
        " > finished creating indexed dataset in {:4f} seconds".format(
            time.time() - start_time
        )
    )
    print_rank_0("    number of documents: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def build_dataset_group(
    dataset_group_name,
    paths,
    weights,
    splits,
    data_impl,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    train_valid_test,
):
    """
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed on the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    """

    assert train_valid_test in ["train", "valid", "test"]

    # Single dataset.
    if len(paths) == 1:
        dataset = _build_single_datasets(
            paths[0],
            splits[0],
            data_impl,
            train_valid_test_num_samples,
            seq_length,
            seed,
            skip_warmup,
            dataset_group_name,
            train_valid_test,
        )
        return dataset
    # Blending dataset.
    else:
        data_prefix = []
        # data_prefix is on the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w, p in zip(weights, paths):
            data_prefix += [w, p]

        output = get_datasets_weights_and_num_samples(
            data_prefix, train_valid_test_num_samples
        )
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(
                prefixes[i],
                splits[i],
                data_impl,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
                dataset_group_name,
                train_valid_test,
            )

            datasets.append(ds)
        all_datasets = BlendableDataset(datasets, weights)

        return all_datasets


def _build_single_datasets(
    data_prefix,
    range_string,
    data_impl,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    dataset_group_name,
    train_valid_test,
):
    """Build a single dataset"""

    assert train_valid_test in ["train", "valid", "test"]
    index = ["train", "valid", "test"].index(train_valid_test)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    # this corresponds to option2 for data loading on the form
    # WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT3 START:END PATH3
    # splits here is an array of size 2  [start_index, end_index]
    splits = get_split_by_range_(range_string=range_string, size=total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    print_rank_0("    {}:".format(dataset_group_name))
    print_rank_0(
        "     document indices in [{}, {}) total of {} documents".format(
            splits[0], splits[1], splits[1] - splits[0]
        )
    )

    def build_dataset(name):
        dataset = None
        if splits[1] > splits[0]:
            documents = np.arange(
                start=splits[0], stop=splits[1], step=1, dtype=np.int32
            )
            dataset = GPTDataset(
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
            )
        return dataset

    dataset = build_dataset(dataset_group_name)

    return dataset
