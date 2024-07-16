# kaggle-leap-solution


## 1. Machine Specifications

- Preprocessing: 
    - Google Compute Engine n2-highmem-32（32 vCPU、16 cores、256 GB memory）
- Training & Inference: 
    - Google Compute Engine N1 Series (1 x NVIDIA V100, 12 vCPU, 6 cores, 78 GB memory)



## 2. Directory&Data Preparation

1. Create the input&output directory if it doesn't exist:
    ```
    mkdir -p input
    mkdir -p output
    ```
2. Download the dataset: `kaggle competitions download -c leap-atmospheric-physics-ai-climsim`
3. Move the downloaded dataset to the input directory (input/leap-atmospheric-physics-ai-climsim)


## 3. Environment Setup
- Start Docker and enter bash:
    It mounts input&output directory
```
docker compose build
docker compose run --rm kaggle bash 
```

## 4. Preprocessing 

```
python preprocess/test_parquet/run.py exp=base
python preprocess/valid_parquet/run.py exp=base
python preprocess/normalize_009_rate_feat/run.py exp=bolton
python preprocess/tmelt_tice/run.py exp=001 
python preprocess/make_webdataset_batch/run.py exp=all  # donwload & create webdataset
```

## 5. Training and Inference
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


## Output
- submission.parquet: submission file (for stacking)
- valid_pred.parquet: prediction of validation data (for stacking)

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
- prediction: https://www.kaggle.com/datasets/kami634/kami-leap-pred2
- models: https://www.kaggle.com/datasets/kami634/kaggle-leap-models