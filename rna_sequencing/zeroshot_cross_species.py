import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

print("=========================================================")
print(" OPERAZIONE: ZERO-SHOT V2 (IL VERO BATCH EFFECT)")
print("=========================================================\n")

# 1. CARICAMENTO DATI
print("[+] Caricamento delle Matrici Forgiate...")
df_mouse = pd.read_csv("mouse_relational_dataset.csv", index_col=0)
df_human = pd.read_csv("human_relational_dataset.csv", index_col=0)

# 2. L'INIEZIONE DEL COVARIATE SHIFT (Il vero Batch Effect clinico)
# Ipotizziamo che l'ospedale umano usi una macchina che cattura solo il 30% dell'RNA.
# In biologia, la bassa profondità di lettura colpisce tutti i geni proporzionalmente.
SHIFT_FACTOR = 0.3

print(f"[!] Applicazione Shift Tecnologico: I valori assoluti umani crollano al {SHIFT_FACTOR*100}%")
for col in df_human.columns:
    if col.startswith('abs_'):
        # Il modello assoluto vedrà numeri distrutti.
        # I valori 'z_' restano identici perché la proporzione si conserva.
        df_human[col] = df_human[col] * SHIFT_FACTOR

# 3. L'ORACOLO TOPOLOGICO (A misura di Albero Decisionale)
# Definiamo la Ground Truth basandoci sui percentili del TOPO, e applichiamo
# la stessa regola esatta all'UMANO.
# Regola: Se z_MYC o z_ERBB2 sono nel top 20% della distribuzione, la cellula è maligna.
myc_thresh = df_mouse['z_MYC'].quantile(0.80)
erbb2_thresh = df_mouse['z_ERBB2'].quantile(0.80)

df_mouse['Target'] = ((df_mouse['z_MYC'] > myc_thresh) | (df_mouse['z_ERBB2'] > erbb2_thresh)).astype(int)
df_human['Target'] = ((df_human['z_MYC'] > myc_thresh) | (df_human['z_ERBB2'] > erbb2_thresh)).astype(int)

# 4. PREPARAZIONE CAMPO DI BATTAGLIA
onco_genes = ['MYC', 'KRAS', 'ERBB2', 'CD44', 'TP53', 'PIK3CA']
abs_features = [f"abs_{g}" for g in onco_genes]
rel_features = [f"z_{g}" for g in onco_genes]

X_train_abs = df_mouse[abs_features]
X_train_rel = df_mouse[rel_features]
y_train = df_mouse['Target']

X_test_abs = df_human[abs_features]
X_test_rel = df_human[rel_features]
y_test = df_human['Target']

# =========================================================
# SCONTRO 1: IL VECCHIO MONDO (MODELLO ASSOLUTO)
# =========================================================
print("\n[!] Addestramento Modello Assoluto sui conteggi RNA del Topo...")
xgb_abs = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
xgb_abs.fit(X_train_abs, y_train)

print("[!] Esecuzione Zero-Shot sui pazienti Umani (Shallow Sequencing)...")
preds_abs = xgb_abs.predict(X_test_abs)
acc_abs = accuracy_score(y_test, preds_abs)

# =========================================================
# SCONTRO 2: ALIEN TECH (MODELLO RELAZIONALE)
# =========================================================
print("\n[+] Addestramento Modello Relazionale sulla Topologia del Topo...")
xgb_rel = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
xgb_rel.fit(X_train_rel, y_train)

print("[+] Esecuzione Zero-Shot sui pazienti Umani...")
preds_rel = xgb_rel.predict(X_test_rel)
acc_rel = accuracy_score(y_test, preds_rel)

# =========================================================
# REPORT TELEMETRICO
# =========================================================
print("\n=========================================================")
print(" RISULTATI DELLO SCONTRO (CON COVARIATE SHIFT SIMULATO)")
print("=========================================================")
print(f" [VECCHIO MONDO] Accuratezza Modello Assoluto:    {acc_abs*100:.1f}%")
print(f" [ALIEN TECH]    Accuratezza Modello Relazionale: {acc_rel*100:.1f}%")
print("---------------------------------------------------------")

print(" MATRICE DI CONFUSIONE - MODELLO ASSOLUTO:")
tn, fp, fn, tp = confusion_matrix(y_test, preds_abs).ravel()
print(f"   Veri Sani: {tn}  |  Falsi Allarmi (FP): {fp}")
print(f"   Tumori Mancati (FN): {fn}  |  Veri Tumori Rilevati: {tp}")

print("\n MATRICE DI CONFUSIONE - MODELLO RELAZIONALE:")
tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_test, preds_rel).ravel()
print(f"   Veri Sani: {tn_r}  |  Falsi Allarmi (FP): {fp_r}")
print(f"   Tumori Mancati (FN): {fn_r}  |  Veri Tumori Rilevati: {tp_r}")
print("=========================================================")
