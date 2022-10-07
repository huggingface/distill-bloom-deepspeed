# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy
from .data import broadcast_data
from .initialize import (destroy_model_parallel, get_data_parallel_group,
                         get_data_parallel_rank, get_data_parallel_world_size,
                         get_embedding_group, get_model_parallel_group,
                         get_model_parallel_rank,
                         get_model_parallel_world_size,
                         get_pipeline_model_parallel_first_rank,
                         get_pipeline_model_parallel_group,
                         get_pipeline_model_parallel_last_rank,
                         get_pipeline_model_parallel_next_rank,
                         get_pipeline_model_parallel_prev_rank,
                         get_pipeline_model_parallel_rank,
                         get_pipeline_model_parallel_world_size,
                         get_tensor_model_parallel_group,
                         get_tensor_model_parallel_rank,
                         get_tensor_model_parallel_src_rank,
                         get_tensor_model_parallel_world_size,
                         get_virtual_pipeline_model_parallel_rank,
                         initialize_model_parallel, is_pipeline_first_stage,
                         is_pipeline_last_stage, is_unitialized,
                         model_parallel_is_initialized,
                         set_pipeline_model_parallel_rank,
                         set_pipeline_model_parallel_world_size,
                         set_tensor_model_parallel_rank,
                         set_tensor_model_parallel_world_size,
                         set_virtual_pipeline_model_parallel_rank)
from .mappings import (copy_to_tensor_model_parallel_region,
                       gather_from_tensor_model_parallel_region,
                       reduce_from_tensor_model_parallel_region,
                       scatter_to_tensor_model_parallel_region)
from .utils import divide, split_tensor_along_last_dim

# from .layers import ColumnParallelLinear
# from .layers import RowParallelLinear
# from .layers import VocabParallelEmbedding
# from .layers import (set_tensor_model_parallel_attributes,
#                      set_defaults_if_not_set_tensor_model_parallel_attributes,
#                      copy_tensor_model_parallel_attributes)


# from .random import checkpoint
# from .random import get_cuda_rng_tracker
# from .random import init_checkpointed_activations_memory_buffer
# from .random import model_parallel_cuda_manual_seed
# from .random import reset_checkpointed_activations_memory_buffer
# from .random import gather_split_1d_tensor
# from .random import split_tensor_into_1d_equal_chunks
