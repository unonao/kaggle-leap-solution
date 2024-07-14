# kaggle-leap-solution


## 1. Machine Specifications



## 2. Directory&Data Preparation

1. Create the input&output directory if it doesn't exist:
    ```
    mkdir -p input
    mkdir -p output
    ```
2. Download the dataset: `kaggle competitions download -c leap-atmospheric-physics-ai-climsim`
3. Move the downloaded dataset to the input directory (input/leap-atmospheric-physics-ai-climsim)


## 3. Environment Setup



## 4. Preprocessing 

```
python preprocess/test_parquet/run.py exp=base
python preprocess/valid_parquet/run.py exp=base
python preprocess/normalize_009_rate_feat/run.py exp=bolton
python preprocess/tmelt_tice/run.py exp=001 
python preprocess/make_webdataset_batch/run.py exp=all  # donwload & create webdataset
```

## 5. Training and Inference
```
python experiments/204_diff_last/run.py exp=all_lr
python experiments/201_unet_multi/run.py exp=all_n3_restart2
python experiments/201_unet_multi/run.py exp=all_512_n3
python experiments/201_unet_multi/run.py exp=all_384_n2
python experiments/201_unet_multi/run.py exp=all
```


## Output
- submission.parquet: submission file (for stacking)
- valid_pred.parquet: prediction of validation data (for stacking)

- output directry
    ```
    output/experiments/204_diff_last/all_lr
    ```                        

result: https://www.kaggle.com/datasets/kami634/kami-leap-pred2