# kaggle-leap-solution

- Normalisation files & Models are avalilable on Google drive
    - https://drive.google.com/drive/folders/1GwKN63l5KT-BDAbGfeo-51wPzvnrW8LT

## Machine Specifications

- Preprocessing: 
    - Google Compute Engine n2-highmem-32（32 vCPU、16 cores、256 GB memory）
- Training & Inference: 
    - Google Compute Engine N1 Series (1 x NVIDIA V100, 12 vCPU, 6 cores, 78 GB memory)


## How to Reproduce Training & Inference
### 1. Directory&Data Preparation

1. Create the input&output directory if it doesn't exist:
    ```
    mkdir -p input
    mkdir -p output
    ```
2. Download the dataset: `kaggle competitions download -c leap-atmospheric-physics-ai-climsim`
3. Move the downloaded dataset to the input directory (input/leap-atmospheric-physics-ai-climsim)


## 2. Environment Setup
- Start Docker and enter bash:
    It mounts input&output directory
```
docker compose build
docker compose run --rm kaggle bash 
```

## 3. Preprocessing 

```
python preprocess/test_parquet/run.py exp=base
python preprocess/valid_parquet/run.py exp=base
python preprocess/normalize_009_rate_feat/run.py exp=bolton
python preprocess/tmelt_tice/run.py exp=001 
python preprocess/make_webdataset_batch/run.py exp=all  # donwload & create webdataset
```

## 4. Training and Inference
To perform only training, add 'exp.modes=[train]'.
For inference only for validation, add 'exp.modes=[valid2]'.
To run inference only for testing, add 'exp.modes=[test]'.
```
python experiments/204_diff_last/run.py exp=all_lr
python experiments/201_unet_multi/run.py exp=all_n3_restart2
python experiments/201_unet_multi/run.py exp=all_512_n3
python experiments/201_unet_multi/run.py exp=all_384_n2
python experiments/201_unet_multi/run.py exp=all
python experiments/217_fix_transformer_leak/run.py exp=all_cos_head64
python experiments/217_fix_transformer_leak/run.py exp=all_cos_head64_n4
python experiments/222_wo_transformer/run.py exp=all
python experiments/222_wo_transformer/run.py exp=all_004
python experiments/225_smoothl1_loss/run.py exp=all_005
python experiments/225_smoothl1_loss/run.py exp=all_beta
```


### Output files
- output directry
    ```
    output/experiments/204_diff_last/all_lr
    output/experiments/201_unet_multi/all_n3_restart2
    output/experiments/201_unet_multi/all_512_n3
    output/experiments/201_unet_multi/all_384_n2
    output/experiments/201_unet_multi/all
    output/experiments/217_fix_transformer_leak/all_cos_head64
    output/experiments/217_fix_transformer_leak/all_cos_head64_n4
    output/experiments/222_wo_transformer/all
    output/experiments/222_wo_transformer/all_004
    output/experiments/225_smoothl1_loss/all_005
    output/experiments/225_smoothl1_loss/all_beta
    ```                        
- Note:
    - submission.parquet: submission file (for stacking)
    - valid_pred.parquet: prediction of validation data (for stacking)

The inference results will be saved in the output directory. The results are also available at [Kami Leap Pred2](https://www.kaggle.com/datasets/kami634/kami-leap-pred2).


## How to Reproduce Inferences Only
If you only want to reproduce the inference, please follow these steps:

### 1. Directory & Data Preparation
1. Create the input&output directory if it doesn't exist:
    ```
    mkdir -p input
    mkdir -p output
    ```

### 2. Environment Setup
- Start Docker and enter bash:
    It mounts input&output directory
```
docker compose build
docker compose run --rm kaggle bash 
```

If you type the `ls` command, you will see the following
```
root@6a75ea1c48b2:/kaggle/working# ls
Dockerfile  README.md  SOLUTION.md  compose.yaml  experiments  input  misc  output  preprocess  utils  yamls
```

### 3. prepare data
Please prepare the following two files
- `/kaggle/working/input/leap-atmospheric-physics-ai-climsim/sample_submission.csv` : weight file
- `/kaggle/working/input/leap-atmospheric-physics-ai-climsim/test.csv` : new data for inference

Execute the following command to convert to a parquet file
```
python preprocess/test_parquet_only_infer/run.py exp=base
```

The input directory should look like this
```
input
|-- leap-atmospheric-physics-ai-climsim
|   |-- sample_submission.csv
|   `-- test.csv
|-- sample_submission.parquet
`-- test.parquet
```

### 4. preprace normalization files & models
- Download the following normalisation files & models in Google drive
    - https://drive.google.com/drive/folders/1GwKN63l5KT-BDAbGfeo-51wPzvnrW8LT

Place the files as follows
```
output
├── experiments
│   ├── 201_unet_multi
│   │   ├── all
│   │   │   └── checkpoints
│   │   │       └── best_model.ckpt
│   │   ├── all_384_n2
│   │   │   └── checkpoints
│   │   │       └── best_model.ckpt
│   │   ├── all_512_n3
│   │   │   └── checkpoints
│   │   │       └── best_model.ckpt
│   │   └── all_n3_restart2
│   │       └── checkpoints
│   │           └── best_model.ckpt
│   ├── 204_diff_last
│   │   └── all_lr
│   │       └── checkpoints
│   │           └── best_model.ckpt
│   ├── 217_fix_transformer_leak
│   │   ├── all_cos_head64
│   │   │   └── checkpoints
│   │   └── all_cos_head64_n4
│   │       └── checkpoints
│   ├── 222_wo_transformer
│   │   ├── all
│   │   │   └── checkpoints
│   │   └── all_004
│   │       └── checkpoints
│   └── 225_smoothl1_loss
│       ├── all_005
│       │   └── checkpoints
│       └── all_beta
│           └── checkpoints
└── preprocess
    ├── normalize_009_rate_feat
    │   └── bolton
    │       ├── x_mean_feat_dict.pkl
    │       ├── x_std_feat_dict.pkl
    │       ├── y_nanmax.npy
    │       ├── y_nanmean.npy
    │       ├── y_nanmin.npy
    │       ├── y_nanstd.npy
    │       ├── y_rms.npy
    │       └── y_rms_sub.npy
    └── tmelt_tice
        └── 001
            ├── tice_array.npy
            └── tmelt_array.npy
```


### 5. Inference

```
python experiments/204_diff_last/run.py exp=all_lr 'exp.modes=[test]'
python experiments/201_unet_multi/run.py exp=all_n3_restart2 'exp.modes=[test]'
python experiments/201_unet_multi/run.py exp=all_512_n3 'exp.modes=[test]'
python experiments/201_unet_multi/run.py exp=all_384_n2 'exp.modes=[test]'
python experiments/201_unet_multi/run.py exp=all 'exp.modes=[test]'
python experiments/217_fix_transformer_leak/run.py exp=all_cos_head64 'exp.modes=[test]'
python experiments/217_fix_transformer_leak/run.py exp=all_cos_head64_n4 'exp.modes=[test]'
python experiments/222_wo_transformer/run.py exp=all 'exp.modes=[test]'
python experiments/222_wo_transformer/run.py exp=all_004 'exp.modes=[test]'
python experiments/225_smoothl1_loss/run.py exp=all_005 'exp.modes=[test]'
python experiments/225_smoothl1_loss/run.py exp=all_beta 'exp.modes=[test]'
```

upload
```
python tools/upload.py 
```

### Output
- output files
    ```
    output/experiments/204_diff_last/all_lr/submission.parquet
    output/experiments/201_unet_multi/all_n3_restart2/submission.parquet
    output/experiments/201_unet_multi/all_512_n3/submission.parquet
    output/experiments/201_unet_multi/all_384_n2/submission.parquet
    output/experiments/201_unet_multi/all/submission.parquet
    output/experiments/217_fix_transformer_leak/all_cos_head64/submission.parquet
    output/experiments/217_fix_transformer_leak/all_cos_head64_n4/submission.parquet
    output/experiments/222_wo_transformer/all/submission.parquet
    output/experiments/222_wo_transformer/all_004/submission.parquet
    output/experiments/225_smoothl1_loss/all_005/submission.parquet
    output/experiments/225_smoothl1_loss/all_beta/submission.parquet
    ```                        

