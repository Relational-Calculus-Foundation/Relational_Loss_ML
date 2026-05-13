import pandas as pd
import numpy as np
from datetime import timedelta

print("[*] Generazione topologia della rete elettrica...")
# Creiamo una rete di base: 300 centrali elettriche e 700 linee ad alta tensione
NUM_NODES = 300
NUM_EDGES = 700

nodes = np.arange(1, NUM_NODES + 1)
base_edges = []

# Generiamo linee casuali, evitando duplicati e auto-connessioni
while len(base_edges) < NUM_EDGES:
    u = np.random.choice(nodes)
    v = np.random.choice(nodes)
    if u != v and (u, v) not in base_edges and (v, u) not in base_edges:
        base_edges.append((u, v))

print("[*] Simulazione telemetria (14 Agosto 2003, 12:00 - 18:00)...")
start_time = pd.to_datetime('2003-08-14 12:00:00')
end_time = pd.to_datetime('2003-08-14 18:00:00')

current_time = start_time
log_data = []

while current_time <= end_time:
    # --- FISICA DEL COLLASSO ---
    # Definiamo la percentuale di linee operative in base all'orario
    if current_time < pd.to_datetime('2003-08-14 16:00:00'):
        # Rete stabile: 99% di linee operative (1% di rumore/manutenzione)
        prob_active = 0.99
    elif current_time < pd.to_datetime('2003-08-14 16:06:00'):
        # Primi guasti isolati in Ohio (Harding-Chamberlin, etc.)
        prob_active = 0.85
    elif current_time < pd.to_datetime('2003-08-14 16:11:00'):
        # Inizio della cascata veloce
        prob_active = 0.50
    elif current_time < pd.to_datetime('2003-08-14 16:15:00'):
        # Collasso totale del Nord-Est
        prob_active = 0.15
    else:
        # Blackout persistente, pochissime isole funzionanti
        prob_active = 0.08

    # Registriamo lo stato di ogni linea per questa finestra temporale
    for u, v in base_edges:
        status = 'ACTIVE' if np.random.rand() < prob_active else 'TRIPPED'
        log_data.append([
            current_time,
            f"SUB_{u:03d}",
            f"SUB_{v:03d}",
            status
        ])

    current_time += timedelta(minutes=5)

print("[*] Salvataggio del dataset...")
df_mock = pd.DataFrame(log_data, columns=['Timestamp', 'From_Station', 'To_Station', 'Status'])
df_mock.to_csv('blackout_2003_timeline.csv', index=False)
print("[+] Dataset 'blackout_2003_timeline.csv' generato con successo! (Circa 50.000 righe di telemetria)")
