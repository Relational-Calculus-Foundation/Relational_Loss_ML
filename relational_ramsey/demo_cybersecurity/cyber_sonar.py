import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# ============================================
# 1. CARICAMENTO E AUTODISCOVERY DELLE COLONNE
# ============================================
print("[*] Caricamento dataset...")
# Assicurati che il file si chiami esattamente così
filename = 'Wednesday-workingHours.pcap_ISCX.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"\n[!] ERRORE: Il file '{filename}' non è stato trovato nella cartella corrente.")
    sys.exit(1)

# Normalizzazione aggressiva: rimuove spazi vuoti e mette tutto in minuscolo
df.columns = df.columns.str.strip().str.lower()

# Sistema di auto-rilevamento per aggirare le diverse versioni di Kaggle/CIC
try:
    col_timestamp = next(col for col in df.columns if 'timestamp' in col or 'time' in col)
    col_src_ip = next(col for col in df.columns if 'source ip' in col or 'src ip' in col)
    col_dst_ip = next(col for col in df.columns if 'destination ip' in col or 'dst ip' in col)
    col_label = next(col for col in df.columns if 'label' in col)
except StopIteration:
    print("\n[!] ERRORE FATALE: Nel tuo CSV mancano gli Indirizzi IP o il Timestamp.")
    print("Devi scaricare la versione 'GeneratedLabelledFlows' completa dal sito del CIC.")
    sys.exit(1)

print("[*] Colonne identificate con successo. Preparazione dati in corso...")

# Rinominiamo le colonne nello standard atteso
df = df.rename(columns={
    col_timestamp: 'Timestamp',
    col_src_ip: 'Source IP',
    col_dst_ip: 'Destination IP',
    col_label: 'Label'
})

# Mappatura etichette: 0 per traffico normale, 1 per attacchi
df['Is_Attack'] = df['Label'].astype(str).str.strip().str.upper().apply(lambda x: 0 if x == 'BENIGN' else 1)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# ============================================
# 2. CALIBRAZIONE ARMONICA DI K (ENERGIA MULTI-GRAFO)
# ============================================
print("[*] Calibrazione della baseline stazionaria...")
WINDOW_SECONDS = "60s"

benign_df = df[df['Is_Attack'] == 0]
benign_windows_m = []

# Analizziamo le prime 20 finestre di traffico pulito per stabilire lo stato di riposo
for name, group in benign_df.groupby(pd.Grouper(key='Timestamp', freq=WINDOW_SECONDS)):
    if not group.empty:
        # ENERGIA REALE: contiamo tutti i flussi (Multi-Grafo)
        benign_windows_m.append(len(group))
        if len(benign_windows_m) >= 20:
            break

media_m_stazionario = np.mean(benign_windows_m)
# Imponiamo che nello stato stazionario la Pressione sia sull'armonica P = 2*pi
K_calibrato = max(1, int(media_m_stazionario / (4 * np.pi)))

print(f"    -> Flussi medi in 60s di traffico normale: {media_m_stazionario:.1f}")
print(f"    -> K adattivo ricalibrato a: {K_calibrato}")

# ============================================
# 3. ANALISI TOPOLOGICA (Estrazione Veloce)
# ============================================
print("[*] Estrazione parametri armonici P e V...")
results = []

for name, group in df.groupby(pd.Grouper(key='Timestamp', freq=WINDOW_SECONDS)):
    if group.empty:
        continue

    # NODI: numero totale di IP unici coinvolti
    n = len(pd.concat([group['Source IP'], group['Destination IP']]).unique())
    # ARCHI (ENERGIA): numero totale di flussi
    m = len(group)

    if m == 0 or n == 0:
        continue

    # Calcolo delle metriche di Ramsey
    P = m / (2 * K_calibrato)
    V = (2 * n) / K_calibrato

    # Ground Truth: la finestra è un attacco se più del 10% del traffico è malevolo
    attack_ratio = group['Is_Attack'].mean()
    is_attack_window = 1 if attack_ratio > 0.1 else 0

    results.append({
        'Timestamp': name,
        'P': P,
        'V': V,
        'Is_Attack': is_attack_window,
        'Raw_Edges': m,
        'Raw_Nodes': n
    })

results_df = pd.DataFrame(results)

# ============================================
# 4. METRICHE DI PERFORMANCE E DEFORMAZIONE
# ============================================
print("[*] Applicazione del Filtro Termodinamico Ramsey...")

# 1. Pressione Attesa secondo l'asintoto teorico P ≈ V/4
results_df['Expected_P'] = results_df['V'] / 4

# 2. Indice di Deformazione D (Rapporto tra pressione reale e attesa)
results_df['Deformation_Ratio'] = np.where(
    results_df['Expected_P'] > 0,
    results_df['P'] / results_df['Expected_P'],
    0
)

# 3. Regola di Rilevamento: Deformazione forte (D > 10) e uscita dallo stato fluido (P > 2*pi)
results_df['Predicted_Alert'] = np.where(
    (results_df['P'] > 2 * np.pi) & (results_df['Deformation_Ratio'] > 10.0),
    1,
    0
)

true_positives = len(results_df[(results_df['Is_Attack'] == 1) & (results_df['Predicted_Alert'] == 1)])
false_positives = len(results_df[(results_df['Is_Attack'] == 0) & (results_df['Predicted_Alert'] == 1)])
false_negatives = len(results_df[(results_df['Is_Attack'] == 1) & (results_df['Predicted_Alert'] == 0)])
actual_attacks = len(results_df[results_df['Is_Attack'] == 1])

print("\n--- RISULTATI DEL CLASSIFICATORE ARMONICO ---")
print(f"Attacchi totali nel dataset: {actual_attacks}")
print(f"Attacchi rilevati (True Positives): {true_positives}")
print(f"Falsi allarmi (False Positives): {false_positives}")
if actual_attacks > 0:
    recall = (true_positives / actual_attacks) * 100
    print(f"Recall (Tasso di rilevamento): {recall:.1f}%")

# ============================================
# 5. VISUALIZZAZIONE E SALVATAGGIO
# ============================================
print("\n[*] Generazione del Diagramma di Fase...")
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 8))

# --- PLOT 1: Serie Storica di P ---
ax1 = plt.subplot(1, 2, 1)
ax1.plot(results_df['Timestamp'], results_df['P'], label='Pressione Strutturale (P)', color='blue')
ax1.axhline(y=2*np.pi, color='green', linestyle='--', linewidth=2, label='Armonica di Base (P=2π)')

# Evidenzia gli attacchi REALI in rosso sullo sfondo
for idx, row in results_df.iterrows():
    if row['Is_Attack'] == 1:
        ax1.axvspan(row['Timestamp'], row['Timestamp'] + pd.Timedelta(seconds=60), color='red', alpha=0.3)

ax1.set_title('Topological Sonar: Rilevamento Anomalie Ramsey')
ax1.set_ylabel('Pressione Strutturale P = m / 2k')
ax1.legend()

# --- PLOT 2: Diagramma di Fase V vs P ---
ax2 = plt.subplot(1, 2, 2)
benign = results_df[results_df['Is_Attack'] == 0]
attack = results_df[results_df['Is_Attack'] == 1]

ax2.scatter(benign['V'], benign['P'], c='green', alpha=0.6, label='Traffico Normale')
ax2.scatter(attack['V'], attack['P'], c='red', alpha=0.8, marker='x', s=50, label='Attacchi')

# Asintoto Teorico di Ramsey
v_vals = np.linspace(0, results_df['V'].max(), 100)
ax2.plot(v_vals, v_vals/4, 'k--', alpha=0.5, label='Asintoto Teorico (P ≈ V/4)')

ax2.set_title('Spazio delle Fasi Termodinamico della Rete')
ax2.set_xlabel('Volume Specifico V = 2n / k')
ax2.set_ylabel('Pressione Strutturale P')
ax2.legend()

plt.tight_layout()
output_image = 'ramsey_phase_diagram.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"[+] GRAFICO SALVATO! Apri il file '{output_image}' nella tua cartella.")
