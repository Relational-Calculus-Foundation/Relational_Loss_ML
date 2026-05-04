# Ames Housing Dataset – Beat Relational Calculus Challenge

**Competition:** [Beat Relational Calculus – Tabular Challenge](https://www.kaggle.com/competitions/beat-relational-calculus-tabular-challenge)

### Overview
This is the classic **Ames Housing dataset** used for the challenge.

The task is to predict the final sale price (`SalePrice`) of residential homes in Ames, Iowa, using only **classical machine learning methods**.  
Relational Calculus techniques (North Star anchoring, intrinsic capacity ratios, geometric relational templates, etc.) are **strictly forbidden**.

### Files
- `train.csv`          – Training set (1460 rows × 81 columns)
- `test.csv`           – Test set (1459 rows × 80 columns)
- `sample_submission.csv` – Correct submission format
- `data_description.txt` – Full data dictionary and variable explanations

### Goal
Achieve the lowest possible **Root Mean Squared Logarithmic Error (RMSLE)** on the private test set using only traditional approaches.

### Important Rules
- No Relational Calculus concepts allowed (see competition rules for details)
- Submissions violating this rule will be disqualified
- The hidden Relational Calculus baseline will be revealed after the deadline

### Getting Started
1. Download the files above
2. Explore `data_description.txt` for detailed feature explanations
3. Build your model using any classical ML tools (XGBoost, LightGBM, CatBoost, neural nets, etc.)

Good luck!  
May the best classical method win.

---

**Dataset Source**  
De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data. *Journal of Statistics Education*, 19(3).
