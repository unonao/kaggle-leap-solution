# Kaggle Solution Summary: Kami Part

## Summary

- **Model**
    - 1D Unet-based model x 11
- It is important to reshape the input to (batch, 60, dim) to explicitly input height relationships into the model.
- Performance improves with a lot of data and models with large parameters.

## Features Selection / Engineering

- **Data**
    - Low-resolution data
        - Train: Data excluding validation from [February of Year 1, February of Year 9)
        - Validation: Approximately 641,280 instances until February of Year 8 (skipping every 7 instances similar to Kaggle data)
- **Input Normalization**
    - Subtract the mean and divide by the standard deviation.
    - State_t, q0001, q0002, q0003, u, v, ozone, ch4, n2o use common normalization across all heights.
        - Reason: E3SM often performs operations by height, so it is preferable to standardize these features.
    - Only q0001, q0002, q0003 undergo exponential change and are normalized as follows: multiply by 1e9, apply log1p, then normalize.
        - Reason: Possibly related to the Clausius-Clapeyron equation, which is somewhat utilized by the model.
- **Output Normalization**
    - Subtract the mean and divide by the standard deviation.
- **Feature Engineering**
    - Relative humidity (expresses the relative amount of q1)
        - Reference: [**Climate-invariant machine learning**](https://www.science.org/doi/10.1126/sciadv.adj7250) ([supplementary material](https://pog.mit.edu/src/beucler_climate_invariant_ml_supplement_2024.pdf))
        - The calculation method for saturation vapor pressure in the paper did not align with E3SM results, so Bolton's method used in E3SM was adopted.
    - Ice rate: q0002 / (q0002 + q0003)
    - Cloud water: (q2 + q3)
    - Add categorical features such as height (0~59) information and whether q0002/q0003 are zero using a 5-dimensional embedding.

## Models

- **GPU:** V100

| # | Model Name | CV | LB | Training Time | Note |
| --- | --- | --- | --- | --- | --- |
| 1 | 204_diff_last_all_lr | 0.7768 | 0.77351 | 1d 4h | |
| 2 | 201_unet_multi_all_n3_restart2 | 0.7783 | - | 22h | |
| 3 | 201_unet_multi_all_512_n3 | 0.7794 | - | 1d 9h | |
| 4 | 201_unet_multi_all_384_n2 | 0.7801 | - | 22h | |
| 5 | 201_unet_multi_all | 0.7815 | - | 1d 7h | |
| 6 | 217_fix_transformer_leak_all_cos_head64 | 0.7817 | - | 1d 7h | With transformer head |
| 7 | 217_fix_transformer_leak_all_cos_head64_n4 | 0.7828 | - | 1d 20h | With transformer head |
| 8 | 222_wo_transformer_all | 0.7839 | - | 2d 21h | |
| 9 | 222_wo_transformer_all_004 | 0.7830 | - | 2d 21h | Parameter: 354 M |
| 10 | 225_smoothl1_loss_all_005 | 0.7833 | - | 3d 7h | |
| 11 | 225_smoothl1_loss_all_beta | 0.7828 | - | 2d 8h | |

- **Base Structure:** height mlp → shallow 1D Unet x 2 → height mlp
- **Initial Height MLP**
    - Apply a common weight MLP to each height.
        - Reason: E3SM often performs operations by height & features should be easy to use later in 1D Unet.
- **1D Unet**
    - A 1D Unet with many channels.
    - Start with 256 dimensions, doubling the number of channels with each convolution, repeated 3 times.
- **Output Height MLP**
    - Apply a common weight MLP to each height.
    - Prepare MLPs for ptend_t, q0001, q0002, q0003, u, v & directly input related features with skip connections.
        - For example, include state_t with ptend_t.
- **Other Scalar Predictions**
    - Predict using the bottleneck of the 1D Unet with an MLP.
- **Final Output**
    - The outputs of height MLP for state_t, q0001, q0002, q0003, u, v are given as 60x2 each, then expressed as x1.exp() - x2.exp() to slightly improve the score.
        - Reason: The exponential relationship in the Clausius-Clapeyron equation and the model aims to predict the difference before and after the change.
- **Training Method**
    - Ignore some labels during training.
        - Labels with weight 0 in the sample submission.
        - Labels to be ignored in post-processing.
    - Optimizer: Adan (not Adam)
    - Scheduler: Cosine schedule with warmup or reduce LR on plateau.
- **Prediction**
    - Use EMA for prediction.

## Post-Processing

- Set some values to state * (1 / -1200)
    - ptend_q0002_12 ~ ptend_q0002_28
- Target: all values in some labels & low or high temperature in some q2, q3 labels