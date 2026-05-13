import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================
# 1. GENERATORE DEL DATASET (MOCK LEHMAN 2008)
# ============================================
print("[*] Generazione del mercato interbancario (Settembre 2008)...")

# Parametri base: 150 banche
n_banks = 150
dates = pd.date_range(start='2008-09-01', end='2008-09-30', freq='6H') # Finestre da 6 ore
dati_mercato = []

for dt in dates:
    # --- FISICA DEL CICLO BOOM-BUST ---
    if dt < pd.to_datetime('2008-09-12'):
        # BOLLA: Mercato iper-liquido pre-crisi (alta fiducia, alto leverage)
        prob_prestito = 0.35 + np.random.uniform(-0.02, 0.02)
    elif dt < pd.to_datetime('2008-09-15'):
        # PANICO PRE-LEHMAN: Frenesia, le banche cercano disperatamente liquidità
        prob_prestito = 0.45 + np.random.uniform(-0.05, 0.05)
    elif dt < pd.to_datetime('2008-09-20'):
        # IL CRACK (15-19 Settembre): Lehman fallisce. Nessuno si fida. I prestiti crollano.
        prob_prestito = 0.05 + np.random.uniform(0.0, 0.02)
    else:
        # INTERVENTO FED/BCE: Il mercato riparte lentamente, ma frammentato
        prob_prestito = 0.12 + np.random.uniform(-0.01, 0.01)

    # Nodi e Archi teorici per questa finestra
    n_attive = int(n_banks * (0.8 + np.random.uniform(0, 0.2)))
    # m massimo possibile è n*(n-1)/2. Moltiplichiamo per la probabilità.
    m_attivi = int((n_attive * (n_attive - 1) / 2) * prob_prestito)

    dati_mercato.append({'Timestamp': dt, 'Nodes': n_attive, 'Edges': m_attivi})

df = pd.DataFrame(dati_mercato)

# ============================================
# 2. CALIBRAZIONE ARMONICA DI RAMSEY
# ============================================
print("[*] Calibrazione dell'equazione di stato...")

# Calibriamo lo stato "Sano" teorico su una densità media di mercato del 20%
# (Un mercato equilibrato, né in bolla né congelato)
m_sano = int((n_banks * (n_banks - 1) / 2) * 0.20)
K_calibrato = max(1, int(m_sano / (4 * np.pi)))

print(f"    -> K adattivo calibrato a: {K_calibrato}")

# Calcolo Pressione (P), Volume (V) e Liquidità/Ridondanza (D)
df['P'] = df['Edges'] / (2 * K_calibrato)
df['V'] = (2 * df['Nodes']) / K_calibrato
df['D'] = df['Edges'] / df['Nodes']

# ============================================
# 3. IDENTIFICAZIONE DELLE FASI DELLA CRISI
# ============================================
# Bolla sistemica: P > 7 (Il sistema è iper-saturo, fragile)
df['Is_Bubble'] = np.where(df['P'] > 7.0, 1, 0)
# Credit Freeze: P crolla sotto 3.0 e D precipita (Pochi prestiti per banca)
df['Is_Freeze'] = np.where((df['P'] < 3.0) & (df['D'] < 5.0), 1, 0)

inizio_panico = df[df['Is_Bubble'] == 1]['Timestamp'].min()
inizio_freeze = df[df['Is_Freeze'] == 1]['Timestamp'].min()

print(f"\n[!] ALLARME RAMSEY: Inizio surriscaldamento (Bolla): {inizio_panico}")
print(f"[!] ALLARME RAMSEY: Collasso della liquidità (Freeze): {inizio_freeze}")

# ============================================
# 4. VISUALIZZAZIONE DEL CICLO FINANZIARIO
# ============================================
print("\n[*] Generazione del Grafico del Collasso Finanziario...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# --- PLOT 1: Serie Storica (Boom and Bust) ---
ax1.plot(df['Timestamp'], df['P'], color='black', linewidth=2.5, label='Pressione di Liquidità (P)')
ax1.axhline(y=7.0, color='red', linestyle='--', linewidth=2, label='Limite Bolla (Cristallizzazione)')
ax1.axhline(y=2*np.pi, color='green', linestyle='--', linewidth=2, label='Ottimale (2π)')
ax1.axhline(y=3.0, color='blue', linestyle=':', linewidth=2, label='Soglia di Credit Freeze')

# Evidenzia gli eventi chiave
ax1.axvspan(pd.to_datetime('2008-09-12'), pd.to_datetime('2008-09-15'), color='red', alpha=0.2, label='Frenesia Pre-Lehman')
ax1.axvline(pd.to_datetime('2008-09-15'), color='black', linestyle='-', linewidth=3, label='Fallimento Lehman Bros')

ax1.set_ylabel('Pressione P = m / 2k')
ax1.set_title('Termodinamica di Ramsey: La Crisi Finanziaria del 2008', fontsize=14)
ax1.legend(loc='upper left')

# --- PLOT 2: Spazio delle Fasi (Il Ciclo di Isteresi) ---
scatter = ax2.scatter(df['V'], df['P'],
                      c=mdates.date2num(df['Timestamp']),
                      cmap='coolwarm', s=70, edgecolor='k', zorder=3)
ax2.plot(df['V'], df['P'], color='gray', alpha=0.5, linewidth=1, zorder=2)

v_vals = np.linspace(df['V'].min() * 0.9, df['V'].max() * 1.1, 100)
ax2.plot(v_vals, v_vals/4, 'k--', alpha=0.6, linewidth=2, zorder=1, label='Asintoto Teorico')

ax2.set_xlabel('Volume Specifico (Banche Attive)')
ax2.set_ylabel('Pressione Strutturale (Prestiti)')
ax2.set_title('Spazio delle Fasi: Dalla Bolla al Congelamento', fontsize=14)
ax2.legend()

cbar = plt.colorbar(scatter, ax=ax2, format=mdates.DateFormatter('%d %b'))
cbar.set_label('Data (Settembre 2008)')

plt.tight_layout()
output_image = 'lehman_ramsey_cycle.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"[+] GRAFICO SALVATO! Apri '{output_image}'.")
