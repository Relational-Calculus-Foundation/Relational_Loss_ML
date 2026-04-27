# coding: utf-8
"""
Zero-Shot Energy Scale Transfer in Top Quark Tagging
Dimostrazione dell'efficacia del Calcolo Relazionale (Adimensionalizzazione)
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import xgboost as xgb
# ==========================================
# 1. DATA LOADER RIGOROSO PER FILE .H5 (CERN FORMAT)
# ==========================================
def load_and_split_hep_data(h5_path, max_events=100000):
    print(f"Caricamento dati dal file HDF5: {h5_path} (usando PyTables)...")
    try:
        import pandas as pd
        # Pandas legge nativamente il formato PyTables del CERN in blocchi
        df = pd.read_hdf(h5_path, key='table', start=0, stop=max_events)

        labels = df['is_signal_new'].values

        # Le feature cinematiche sono E_0, PX_0, PY_0, PZ_0 ... E_199, PX_199...
        # Prendiamo solo le prime 20 particelle più energetiche (80 features)
        feature_cols = []
        for i in range(20):
            feature_cols.extend([f'E_{i}', f'PX_{i}', f'PY_{i}', f'PZ_{i}'])

        X_kinematics = df[feature_cols].values.astype(np.float32)
        X_kinematics = np.nan_to_num(X_kinematics)

    except Exception as e:
        print(f"Errore di lettura: {e}")
        print("Assicurati di aver lanciato 'pip install tables pandas' nel terminale.")
        return None, None

    # --- SIMULAZIONE DEL DATA DRIFT ENERGETICO ---
    # Calcoliamo l'energia totale (E) approssimata del Jet sommando le E delle 20 particelle
    E_indices = [i*4 for i in range(20)]
    Jet_E = np.sum(X_kinematics[:, E_indices], axis=1)

    # Dividiamo il dataset alla mediana
    median_E = np.median(Jet_E)

    mask_low = Jet_E <= median_E
    mask_high = Jet_E > median_E

    X_train_raw, y_train = X_kinematics[mask_low], labels[mask_low]
    X_test_raw, y_test = X_kinematics[mask_high], labels[mask_high]

    Jet_E_train = Jet_E[mask_low]
    Jet_E_test = Jet_E[mask_high]

    print(f"Dati divisi! Train (Bassa Energia): {len(y_train)} eventi. Test (Alta Energia): {len(y_test)} eventi.")

    return (X_train_raw, y_train, Jet_E_train), (X_test_raw, y_test, Jet_E_test)

def extract_lorentz_features(X_raw):
    """
    Estrae le features fisiche perfette:
    La Massa Invariante del Jet e le frazioni adimensionali di pT.
    """
    N = X_raw.shape[0]

    # Inizializziamo E, px, py, pz totali del Jet
    Jet_E = np.zeros(N)
    Jet_px = np.zeros(N)
    Jet_py = np.zeros(N)
    Jet_pz = np.zeros(N)

    # Calcoliamo i totali del Jet
    for i in range(20):
        Jet_E += X_raw[:, i*4 + 0]
        Jet_px += X_raw[:, i*4 + 1]
        Jet_py += X_raw[:, i*4 + 2]
        Jet_pz += X_raw[:, i*4 + 3]

    # 1. LA MASSA INVARIANTE (La costante universale del Top Quark)
    Jet_p2 = Jet_px**2 + Jet_py**2 + Jet_pz**2
    # Evitiamo radici quadrate di numeri negativi (fluttuazioni numeriche)
    Jet_Mass = np.sqrt(np.maximum(Jet_E**2 - Jet_p2, 0))

    # 2. IMPULSO TRASVERSO DEL JET (La nostra nuova North Star relazionale)
    Jet_pT = np.sqrt(Jet_px**2 + Jet_py**2) + 1e-8 # Evitare divisioni per zero

    # Creiamo il nuovo dataset: 1 colonna per la Massa, 20 per le frazioni relazionali di pT
    X_physics = np.zeros((N, 21), dtype=np.float32)
    X_physics[:, 0] = Jet_Mass

    # Calcoliamo le frazioni relazionali z = pT_particella / pT_jet
    for i in range(20):
        px_i = X_raw[:, i*4 + 1]
        py_i = X_raw[:, i*4 + 2]
        pT_i = np.sqrt(px_i**2 + py_i**2)
        X_physics[:, i+1] = pT_i / Jet_pT  # Paradigma Relazionale Puro

    return X_physics

def apply_z_score(X_train_raw, X_test_raw):
    """
    Normalizzazione standard usata solitamente nel Machine Learning.
    Fissa i pesi sui GeV assoluti visti in fase di training.
    """
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0) + 1e-8
    X_train_z = (X_train_raw - mean) / std
    X_test_z = (X_test_raw - mean) / std
    return X_train_z, X_test_z

# ==========================================
# 3. IL MOTORE NEURALE (Multi-Layer Perceptron)
# ==========================================
class TaggerMLP(nn.Module):
    def __init__(self, input_dim=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(name, X_train, y_train, X_test, y_test):
    print(f"\n--- Avvio addestramento: {name} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaggerMLP().to(device)

    # Dati in PyTorch
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 15
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation (AUC ROC è la metrica standard in HEP)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    print(f"[{name}] AUC su dati ad ALTA ENERGIA (Test Zero-Shot): {auc:.4f}")
    return auc

def train_xgboost(name, X_train, y_train, X_test, y_test):
    print(f"\n--- Avvio XGBoost (Stage 2 - Advanced Tuning): {name} ---")

    # Il Motore Sbloccato
    clf = xgb.XGBClassifier(
        n_estimators=600,          # 600 alberi invece di 100
        max_depth=7,               # Alberi più profondi per catturare relazioni complesse
        learning_rate=0.03,        # Apprendimento lento e inesorabile
        subsample=0.8,             # Usa l'80% degli eventi (Amnesia)
        colsample_bytree=0.8,      # Usa l'80% delle feature (Particelle)
        gamma=1.0,                 # Regolarizzazione: taglia i rami inutili (Pruning)
        reg_lambda=1.0,            # Regolarizzazione L2 (Previene pesi estremi)
        n_jobs=-1,                 # Usa tutti i core del PC
        eval_metric='auc',
        random_state=42            # Per riproducibilità
    )

    # Addestramento
    clf.fit(X_train, y_train)

    # Valutazione
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"[{name}] AUC su dati ad ALTA ENERGIA (Test Zero-Shot): {auc:.4f}")
    return auc

# ==========================================
# 4. ESECUZIONE
# ==========================================
if __name__ == "__main__":
    import os
    import pandas as pd

    # NOTA: Sostituisci questo percorso con il file .h5 scaricato da Zenodo
    H5_FILE = "val.h5"

    if not os.path.exists(H5_FILE):
        print(f"ATTENZIONE: File {H5_FILE} non trovato.")
        print("Scaricare il dataset Kasieczka da: https://zenodo.org/record/2603256")
        print("Comando rapido bash: wget https://zenodo.org/record/2603256/files/val.h5")
        exit()

    # ... (caricamento dati)
    train_tuple, test_tuple = load_and_split_hep_data(H5_FILE)
    X_train_raw, y_train, _ = train_tuple
    X_test_raw, y_test, _ = test_tuple

    print("\nPre-processing dei due paradigmi...")

    # 1. IL PARADIGMA CLASSICO (Assoluto): GeV grezzi + Z-score
    X_train_abs, X_test_abs = apply_z_score(X_train_raw, X_test_raw)

    # 2. IL PARADIGMA RELAZIONALE (Nuova Fisica): Invarianza di Lorentz + Frazioni
    X_train_rel = extract_lorentz_features(X_train_raw)
    X_test_rel  = extract_lorentz_features(X_test_raw)

    # TEST CON XGBOOST (Stage 2 - Formula 1)
    auc_abs_xgb = train_xgboost("Vecchio Mondo (GeV Grezzi + Z-Score)", X_train_abs, y_train, X_test_abs, y_test)
    auc_rel_xgb = train_xgboost("Calcolo Relazionale (Lorentz Invariant)", X_train_rel, y_train, X_test_rel, y_test)

    print("\n==========================================")
    print(f"LA VERA VITTORIA RELAZIONALE: +{(auc_rel_xgb - auc_abs_xgb):.4f}")
    print("==========================================")
