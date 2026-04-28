import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# Simuliamo il caricamento del dataset che abbiamo generato prima
df_quantum = pd.read_csv("quantum_vqe_dataset.csv")
# Per lo script, separiamo i due mondi:
df_h2 = df_quantum[df_quantum['Molecule'] == 'H2'].copy()
df_lih = df_quantum[df_quantum['Molecule'] == 'LiH'].copy()



# Assumiamo che df_h2 e df_lih siano già caricati dalla generazione precedente
X_train = df_h2[['Distance_A']]
X_test  = df_lih[['Distance_A']]
y_train_abs = df_h2['E_Total']
y_test_abs  = df_lih['E_Total']

print("=====================================================")
print(" AVVIO OPERAZIONE V2: IL POTENZIALE INFORMATIVO")
print("=====================================================")

# ---------------------------------------------------------
# 1. DEFINIZIONE DELLA CAPACITÀ GLOBALE (Il tuo approccio)
# ---------------------------------------------------------
# Energie degli atomi isolati (calcolate nella base quantistica STO-3G)
# Questo rappresenta il "Potenziale Informativo" di base.
INFO_H  = -0.4666  # Hartree
INFO_Li = -7.3155  # Hartree

# Capacità Globale (C) = Somma delle informazioni isolate
C_H2  = INFO_H + INFO_H
C_LiH = INFO_Li + INFO_H

# ---------------------------------------------------------
# 2. VECCHIO MONDO: Modello Assoluto (Raw Hartree)
# ---------------------------------------------------------
xgb_abs = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb_abs.fit(X_train, y_train_abs)
preds_abs = xgb_abs.predict(X_test)
mae_abs = mean_absolute_error(y_test_abs, preds_abs)

# ---------------------------------------------------------
# 3. IL CALCOLO RELAZIONALE V2 (Topologia dell'Informazione)
# ---------------------------------------------------------
# Calcoliamo l'Energia di Legame: quanta informazione "cambia" rispetto a C
E_legame_H2 = df_h2['E_Total'] - C_H2

# Trasformiamo in frazione adimensionale (z_info)
# "Che percentuale del potenziale informativo totale viene usata per il legame?"
y_train_rel = E_legame_H2 / np.abs(C_H2)

# Addestriamo l'IA su questa proporzione geometrica pura
xgb_rel = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb_rel.fit(X_train, y_train_rel)

# Test Zero-Shot su LiH: l'IA prevede la frazione z per la nuova molecola
preds_rel_frac = xgb_rel.predict(X_test)

# RICOSTRUZIONE FISICA:
# E_totale = (Frazione predetta * |Capacità LiH|) + Capacità LiH
preds_rel_reconstructed = (preds_rel_frac * np.abs(C_LiH)) + C_LiH
mae_rel_v2 = mean_absolute_error(y_test_abs, preds_rel_reconstructed)

# ---------------------------------------------------------
# 4. REPORT DEI DANNI
# ---------------------------------------------------------
print("\n=====================================================")
print(" RISULTATI ZERO-SHOT TRANSFER (Errore Medio Assoluto)")
print("=====================================================")
print(f"Errore Modello Assoluto:         {mae_abs:.4f} Hartree")
print(f"Errore Calcolo Relazionale V2:   {mae_rel_v2:.4f} Hartree")

miglioramento = ((mae_abs - mae_rel_v2) / mae_abs) * 100
print("-----------------------------------------------------")
print(f" VANTAGGIO RELAZIONALE: Riduzione Errore del {miglioramento:.1f}%")
print("=====================================================")


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# Simuliamo il caricamento del dataset che abbiamo generato prima
df_quantum = pd.read_csv("quantum_vqe_dataset.csv")
# Per lo script, separiamo i due mondi:
df_h2 = df_quantum[df_quantum['Molecule'] == 'H2'].copy()
df_lih = df_quantum[df_quantum['Molecule'] == 'LiH'].copy()



# Assumiamo che df_h2 e df_lih siano già caricati dalla generazione precedente
X_train = df_h2[['Distance_A']]
X_test  = df_lih[['Distance_A']]
y_train_abs = df_h2['E_Total']
y_test_abs  = df_lih['E_Total']

print("=====================================================")
print(" AVVIO OPERAZIONE V2: IL POTENZIALE INFORMATIVO")
print("=====================================================")

# ---------------------------------------------------------
# 1. DEFINIZIONE DELLA CAPACITÀ GLOBALE (Il tuo approccio)
# ---------------------------------------------------------
# Energie degli atomi isolati (calcolate nella base quantistica STO-3G)
# Questo rappresenta il "Potenziale Informativo" di base.
INFO_H  = -0.4666  # Hartree
INFO_Li = -7.3155  # Hartree

# Capacità Globale (C) = Somma delle informazioni isolate
C_H2  = INFO_H + INFO_H
C_LiH = INFO_Li + INFO_H

# ---------------------------------------------------------
# 2. VECCHIO MONDO: Modello Assoluto (Raw Hartree)
# ---------------------------------------------------------
xgb_abs = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb_abs.fit(X_train, y_train_abs)
preds_abs = xgb_abs.predict(X_test)
mae_abs = mean_absolute_error(y_test_abs, preds_abs)

# ---------------------------------------------------------
# 3. IL CALCOLO RELAZIONALE V2 (Topologia dell'Informazione)
# ---------------------------------------------------------
# Calcoliamo l'Energia di Legame: quanta informazione "cambia" rispetto a C
E_legame_H2 = df_h2['E_Total'] - C_H2

# Trasformiamo in frazione adimensionale (z_info)
# "Che percentuale del potenziale informativo totale viene usata per il legame?"
y_train_rel = E_legame_H2 / np.abs(C_H2)

# Addestriamo l'IA su questa proporzione geometrica pura
xgb_rel = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
xgb_rel.fit(X_train, y_train_rel)

# Test Zero-Shot su LiH: l'IA prevede la frazione z per la nuova molecola
preds_rel_frac = xgb_rel.predict(X_test)

# RICOSTRUZIONE FISICA:
# E_totale = (Frazione predetta * |Capacità LiH|) + Capacità LiH
preds_rel_reconstructed = (preds_rel_frac * np.abs(C_LiH)) + C_LiH
mae_rel_v2 = mean_absolute_error(y_test_abs, preds_rel_reconstructed)

# ---------------------------------------------------------
# 4. REPORT DEI DANNI
# ---------------------------------------------------------
print("\n=====================================================")
print(" RISULTATI ZERO-SHOT TRANSFER (Errore Medio Assoluto)")
print("=====================================================")
print(f"Errore Modello Assoluto:         {mae_abs:.4f} Hartree")
print(f"Errore Calcolo Relazionale V2:   {mae_rel_v2:.4f} Hartree")

miglioramento = ((mae_abs - mae_rel_v2) / mae_abs) * 100
print("-----------------------------------------------------")
print(f" VANTAGGIO RELAZIONALE: Riduzione Errore del {miglioramento:.1f}%")
print("=====================================================")
