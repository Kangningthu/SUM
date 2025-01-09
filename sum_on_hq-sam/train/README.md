# Training instruction for SUM (HQ-SAM architecture)

We closely adhere to the HQ-SAM training repository and framework in this implementation. To ensure a fair comparison, we have followed the HQ-SAM implementation and did not use the interactive training method for point sampling. For details on the interactive sampling method, please refer to the main folder.

We organize the training folder as follows.
```
train
|____data
|____pretrained_checkpoint
|____train.py
|____utils
| |____dataloader.py
| |____misc.py
| |____loss_mask.py
|____segment_anything_training
|____work_dirs
```

## 1. Data Preparation

HQSeg-44K can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data)

### Expected dataset structure for HQSeg-44K

```
data
|____DIS5K
|____cascade_psp
| |____DUTS-TE
| |____DUTS-TR
| |____ecssd
| |____fss_all
| |____MSRA_10K
|____thin_object_detection
| |____COIFT
| |____HRSOD
| |____ThinObject5K

SAM1b dataset can be downloaded from the official website, you will need to obtain the SAM pseudo label and use the mask-refinement module to quantity the uncertainty map (##todo release the mask-refinement module and the uncertainty map)
```

## 2. Init Checkpoint
Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)

### Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_b_maskdecoder.pth
|____sam_vit_b_01ec64.pth
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth
|____sam_vit_h_maskdecoder.pth
|____sam_vit_h_4b8939.pth

```

## 3. Training
To train HQ-SAM on HQSeg-44K dataset

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1333 train_uncertainty_aware.py \
--checkpoint xxx/sam_vit_h_4b8939.pth --model-type vit_h --output xxx \
--use_uncertainmap yes \
--min_ratio 100 \
--min_refine_ratio 100 \
--use_task_prompt_token yes \
--find_unused_params 
```


## 4. Evaluation
To evaluate on 4 HQ-datasets for the bounding box prompt segmentation

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train_uncertainty_aware.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint>
```
