import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

# ============================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ============================================
print("[*] Caricamento dataset della rete elettrica (NERC 2003)...")
filename = 'blackout_2003_timeline.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"\n[!] ERRORE: Il file '{filename}' non è stato trovato.")
    sys.exit(1)

# Normalizzazione nomi colonne
df.columns = df.columns.str.strip()

# Verifica colonne essenziali
required_cols = {'Timestamp', 'From_Station', 'To_Station', 'Status'}
if not required_cols.issubset(set(df.columns)):
    print(f"\n[!] ERRORE: Il CSV deve contenere le colonne: {required_cols}")
    sys.exit(1)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Ordiniamo cronologicamente
df = df.sort_values('Timestamp')

# ============================================
# 2. CALIBRAZIONE ARMONICA (BASELINE PRE-CRISI)
# ============================================
print("[*] Calibrazione termodinamica della baseline...")

# Usiamo le prime 2 ore come stato di "Pace" (Rete Integra)
start_time = df['Timestamp'].min()
baseline_end = start_time + pd.Timedelta(hours=2)

baseline_df = df[(df['Timestamp'] <= baseline_end) & (df['Status'].str.upper() == 'ACTIVE')]

# Contiamo i Nodi (Centrali) e gli Archi (Linee Attive univoche) della baseline
nodi_unici_base = pd.concat([baseline_df['From_Station'], baseline_df['To_Station']]).unique()
archi_unici_base = baseline_df[['From_Station', 'To_Station']].drop_duplicates()

n_base = len(nodi_unici_base)
m_base = len(archi_unici_base)

# Taratura di K: Imponiamo che la Pressione stazionaria sia sull'armonica 2*pi
K_calibrato = max(1, int(m_base / (4 * np.pi)))

print(f"    -> Rete in stato di quiete: {n_base} nodi, {m_base} linee attive.")
print(f"    -> K adattivo calibrato a: {K_calibrato}")

# ============================================
# 3. ANALISI TOPOLOGICA (FINESTRATURA)
# ============================================
print("[*] Estrazione parametri P, V e Indice di Ridondanza D...")
WINDOW_MINUTES = "5min"
results = []

# Creiamo le finestre temporali
for name, group in df.groupby(pd.Grouper(key='Timestamp', freq=WINDOW_MINUTES)):
    # Manteniamo SOLO le linee attive in questa finestra
    active_lines = group[group['Status'].str.upper() == 'ACTIVE']

    if active_lines.empty:
        continue

    # Nodi (Sottostazioni ancora connesse)
    nodi_connessi = pd.concat([active_lines['From_Station'], active_lines['To_Station']]).unique()
    n = len(nodi_connessi)

    # Archi (Linee operative)
    archi_operativi = active_lines[['From_Station', 'To_Station']].drop_duplicates()
    m = len(archi_operativi)

    if m == 0 or n == 0:
        continue

    # Metriche di Ramsey
    P = m / (2 * K_calibrato)
    V = (2 * n) / K_calibrato

    # INDICE DI RIDONDANZA (D): Archi per Nodo. Segnala la frammentazione.
    D = m / n

    results.append({
        'Timestamp': name,
        'P': P,
        'V': V,
        'Nodes': n,
        'Edges': m,
        'D': D
    })

results_df = pd.DataFrame(results)

# ============================================
# 4. IDENTIFICAZIONE DEL COLLASSO A CASCATA
# ============================================
# Regola di allarme implosione:
# La pressione crolla sotto 4.5 E la ridondanza scende sotto 1.2 (grafo ad albero/fragile)
results_df['Is_Collapse'] = np.where(
    (results_df['P'] < 4.5) & (results_df['D'] < 1.2),
    1,
    0
)

collassi = results_df[results_df['Is_Collapse'] == 1]
if not collassi.empty:
    orario_crisi = collassi.iloc[0]['Timestamp']
    print(f"\n[!] ALLARME CRITICO RILEVATO: Inizio del collasso topologico alle {orario_crisi}")
else:
    print("\n[*] Nessun collasso sistemico rilevato nel dataset.")

# ============================================
# 5. VISUALIZZAZIONE AVANZATA DELLA TRAIETTORIA
# ============================================
print("[*] Generazione del Diagramma di Implosione...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# --- PLOT 1: Serie Storica della Pressione (Caduta Libera) ---
ax1.plot(results_df['Timestamp'], results_df['P'], color='blue', linewidth=2, label='Pressione Strutturale (P)')
ax1.axhline(y=2*np.pi, color='green', linestyle='--', linewidth=2, label='Stato Ottimale (2π)')
ax1.axhline(y=4.5, color='orange', linestyle=':', linewidth=2, label='Soglia di Fragilità Strutturale')

# Formattazione asse X per orari leggibili
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.set_ylabel('Pressione P = m / 2k')
ax1.set_title('Termodinamica della Rete Elettrica: Rilevamento Collasso NERC 2003', fontsize=14)
ax1.legend()

# --- PLOT 2: Spazio delle Fasi (Traiettoria) ---
# Usiamo il colore per mostrare lo scorrere del tempo (da giallo a viola oscuro)
scatter = ax2.scatter(results_df['V'], results_df['P'],
                      c=mdates.date2num(results_df['Timestamp']),
                      cmap='plasma', s=60, edgecolor='k', zorder=3)

# Linea che unisce i punti per mostrare la 'caduta'
ax2.plot(results_df['V'], results_df['P'], color='gray', alpha=0.5, linewidth=1, zorder=2)

# Asintoto teorico
v_vals = np.linspace(results_df['V'].min() * 0.9, results_df['V'].max() * 1.1, 100)
ax2.plot(v_vals, v_vals/4, 'k--', alpha=0.6, linewidth=2, zorder=1, label='Asintoto Ramsey (P ≈ V/4)')

ax2.set_xlabel('Volume Specifico V = 2n / k')
ax2.set_ylabel('Pressione Strutturale P')
ax2.set_title('Spazio delle Fasi: La Traiettoria dell\'Implosione', fontsize=14)
ax2.legend()

# Aggiunta barra del tempo
cbar = plt.colorbar(scatter, ax=ax2, format=mdates.DateFormatter('%H:%M'))
cbar.set_label('Orario (EDT)')

plt.tight_layout()
output_image = 'blackout_phase_trajectory.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"[+] GRAFICO SALVATO! Apri '{output_image}' per vedere il crollo strutturale.")
