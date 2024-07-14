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
python preprocess/make_webdataset_batch/run.py exp=all 
```

## 5. Training and Inference
```
python experiments/204_diff_last/run.py exp=all_lr
```
