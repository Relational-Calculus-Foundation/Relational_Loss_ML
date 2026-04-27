# coding: utf-8
"""
Relational XGBoost per Dati Tabulari (Ames Housing)
Dimostrazione dell'efficienza bruta: Calcolo Relazionale + Gradient Boosting
"""

import pandas as pd
import numpy as np
import xgboost as xgb

def prepare_relational_data(train_path="train.csv", test_path="test.csv"):
    print("Inizializzazione purificazione relazionale dei dati...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    test_ids = test_df['Id'].values
    y_train_absolute = train_df['SalePrice'].values
    
    # 1. LA NORTH STAR (Capacità Globale)
    global_capacity = y_train_absolute.max()
    
    train_features = train_df.drop(['Id', 'SalePrice'], axis=1)
    test_features = test_df.drop(['Id'], axis=1)
    combined = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)
    
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    categorical_cols = combined.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        combined[col] = combined[col].fillna(combined[col].median())
    for col in categorical_cols:
        combined[col] = combined[col].fillna("None")
        
    combined_encoded = pd.get_dummies(combined, drop_first=True).astype(np.float32)
    
    # 2. PURIFICAZIONE RELAZIONALE (Nessuna media, nessuna deformazione Z-score)
    # Ogni feature è espressa in proporzione pura rispetto alla sua Capacità massima
    for col in combined_encoded.columns:
        col_max = combined_encoded[col].abs().max()
        if col_max > 0:
            combined_encoded[col] = combined_encoded[col] / col_max
            
    num_train = len(train_features)
    X_train = combined_encoded.iloc[:num_train].copy()
    X_test = combined_encoded.iloc[num_train:].copy()
    
    print(f"Spazio Geometrico mappato. Dimensioni: {X_train.shape[1]}")
    return X_train, y_train_absolute, X_test, test_ids, global_capacity

def run_relational_xgboost(X_train, y_train_abs, X_test, test_ids, global_capacity):
    print(f"\n--- AVVIO MOTORE XGBOOST RELAZIONALE ---")
    print(f"Ancoraggio Capacità (North Star): ${global_capacity:,.2f}")
    
    # Mappatura del target nello spazio adimensionale [0, 1]
    y_train_ratio = y_train_abs / global_capacity
    
    dtrain = xgb.DMatrix(X_train, label=y_train_ratio)
    dtest = xgb.DMatrix(X_test)
    
    params = {
        'objective': 'reg:squarederror', 
        'max_depth': 4,                  
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'disable_default_eval_metric': 1 # Disattiva la metrica interna per usare la nostra
    }
    
    # Metrica personalizzata per tracciare il vero errore Kaggle in Cross-Validation
    def eval_kaggle_log_rmse(preds, dtrain):
        labels_ratio = dtrain.get_label()
        abs_labels = labels_ratio * global_capacity
        abs_preds = preds * global_capacity
        abs_preds = np.clip(abs_preds, 1.0, None) # Scudo termico logaritmi
        log_rmse = np.sqrt(np.mean((np.log(abs_preds) - np.log(abs_labels))**2))
        return 'kaggle_log_rmse', float(log_rmse)

    print("Addestramento in Cross-Validation per trovare l'ottimo geometrico...")
    cv_results = xgb.cv(
        params, dtrain, num_boost_round=150, 
        nfold=5, early_stopping_rounds=20, verbose_eval=False,
        custom_metric=eval_kaggle_log_rmse, maximize=False
    )
    
    best_rounds = cv_results.shape[0]
    final_rmse = cv_results['test-kaggle_log_rmse-mean'].iloc[-1]
    
    print(f"Ottimo raggiunto in {best_rounds} iterazioni.")
    print(f"Vero Kaggle Log RMSE Medio (5-Fold): {final_rmse:.5f}")
    
    print("Estrazione previsioni finali...")
    relational_xgb = xgb.train(params, dtrain, num_boost_round=best_rounds)
    pred_ratio = relational_xgb.predict(dtest)
    pred_absolute = pred_ratio * global_capacity
    
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': pred_absolute})
    submission.to_csv('relational_xgboost_submission.csv', index=False)
    print("Missione compiuta. File salvato: 'relational_xgboost_submission.csv'")

if __name__ == "__main__":
    X_train, y_train, X_test, test_ids, cap = prepare_relational_data("train.csv", "test.csv")
    run_relational_xgboost(X_train, y_train, X_test, test_ids, cap)
