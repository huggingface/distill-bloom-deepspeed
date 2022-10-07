# distill-bloom-deepspeed

Teacher - student distillation using DeepSpeed. 
This repository is partially based from [BLOOM DeepSpeed repository](https://github.com/huggingface/transformers-bloom-inference/tree/main/bloom-inference-scripts). We follow the same setup as the repository above

## Setup 

```pip install transformers huggingface_hub==0.9.0```
```pip install deepspeed>=0.7.3```

## Install teacher checkpoints

Install the DeepSpeed teacher checkpoints from [here]() to perform fast loading as described [here](https://github.com/huggingface/transformers-bloom-inference/tree/main/bloom-inference-scripts#run). Download them locally and follow the instructions below to run the training. 

### Teacher inference

We highly recommend to install the teacher and student weights locally, therefore to not have to re-install the weights again. 
After installing the teacher weights, run the following command to perform inference on the teacher model. 

```
deepspeed --num_gpus NUM_GPUS teacher-inference-script.py --teacher-model-path[PATH_TO_BLOOM] --train-weighted-split-paths-path [PATH_TO_DATA] --train-iters [TRAIN_ITERS] --global-batch-size [GLOBAL_BATCH_SIZE] --eval-iters [EVAL_ITERS] --seq-length [SEQ_LEN] 
```

#### Processing the dataset

##### Download the dataset 

Here we use the dataset used to train the BLOOM model, that is available on Jean Zay. First, download the dataset that is available on a S3 bucket. The raw dataset consist of 1.6TB of numpy arrays. If you want to train our your custom dataset, please build your own dataloader structure. 

##### Get the splits

For now we recommend to get the splits by running the following command. 

```
export DATAPATH=[PATH_TO_DATASET]
git clone https://github.com/bigscience-workshop/bigscience.git
cd bigscience/
python data/catalogue/load_ratios_meg_ds_format.py --dataset-ratios-path ./data/catalogue/training_dataset_ratios_merged_nigercongo_v3.json --split train --output-meg-ds-ratio-file $DATAPATH/train.txt
python data/catalogue/load_ratios_meg_ds_format.py --dataset-ratios-path ./data/catalogue/training_dataset_ratios_merged_nigercongo_v3.json --split val --output-meg-ds-ratio-file $DATAPATH/val.txt
```

##### Test the data loading script

```
deepspeed --num_gpus 8 test.py --train-weighted-split-paths-path $DATAPATH/train.txt --train-iters 200 --global-batch-size 64 --eval-iters 20 --seq-length 2048
```

This test should output the lenght of the combined dataset as well as the total number of epochs.

#### Training

One the dataset is ready, we can start training the student model. 


## Roadmap

- [ ] Add support for teacher inference
- [ ] Add support for student inference
- [ ] Add support for communicating teacher logits to student node
- [ ] Add support for student training (Ds-Zero)
- [ ] Add support for distributed training (`hostfile`)
- [x] Add support for loading Jean-Zay dataset
- [ ] Add support for loading custom dataset

