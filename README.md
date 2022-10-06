# distill-bloom-deepspeed

Teacher - student distillation using DeepSpeed. 
This repository is partially based from [BLOOM DeepSpeed repository](https://github.com/huggingface/transformers-bloom-inference/tree/main/bloom-inference-scripts). We follow the same setup as the repository above

## Setup 

```pip install transformers```
```pip install deepspeed>=0.7.3```

## Install teacher checkpoints

Install the DeepSpeed teacher checkpoints from [here]() to perform fast loading as described [here](https://github.com/huggingface/transformers-bloom-inference/tree/main/bloom-inference-scripts#run). Download them locally and follow the instructions below to run the training. 

## Roadmap

- [ ] Add support for teacher inference
- [ ] Add support for student inference
- [ ] Add support for student training (Ds-Zero)
- [ ] Add support for distributed training (`hostfile`)
- [ ] Add support for loading Jean-Zay dataset
- [ ] Add support for loading custom dataset

