import pandas as pd
import scanpy as sc
import numpy as np
from pathlib import Path
import gc
from scipy.sparse import issparse # <-- LA CHIAVE DI SBLOCCO

print("=========================================================")
print(" OPERAZIONE: FORGIATURA ONTOLOGIA RELAZIONALE (ZERO-SHOT)")
print("=========================================================\n")

BASE_DIR = Path(".")
HUMAN_DIR = BASE_DIR / "human"
TOPO_DIR = BASE_DIR / "topo"

# 1. IL CORE INFORMATIVO (Ortologhi)
HK_GENES = {
    'human': ['GAPDH', 'ACTB', 'B2M', 'RPL13A'],
    'mouse': ['Gapdh', 'Actb', 'B2m', 'Rpl13a']
}

ONCO_GENES = {
    'human': ['MYC', 'KRAS', 'ERBB2', 'CD44', 'TP53', 'PIK3CA'],
    'mouse': ['Myc', 'Kras', 'Erbb2', 'Cd44', 'Trp53', 'Pik3ca']
}

def process_human_cid_folders(base_path):
    print("[+] Inizio estrazione relazionale: Fronte UMANO (26 Campioni)")
    all_cells = []

    cid_folders = [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith("CID")]

    for i, folder in enumerate(cid_folders):
        print(f"  [{i+1}/{len(cid_folders)}] Infiltrazione in {folder.name}...")
        try:
            mtx_path = folder / "count_matrix_sparse.mtx"
            genes_path = folder / "count_matrix_genes.tsv"
            barcodes_path = folder / "count_matrix_barcodes.tsv"

            if not (mtx_path.exists() and genes_path.exists() and barcodes_path.exists()):
                print(f"    [!] File mancanti in {folder.name}. Salto.")
                continue

            adata = sc.read_mtx(mtx_path).T

            genes_df = pd.read_csv(genes_path, sep='\t', header=None)
            gene_col = 1 if genes_df.shape[1] > 1 else 0
            adata.var_names = genes_df[gene_col].values
            adata.var_names_make_unique()

            barcodes_df = pd.read_csv(barcodes_path, sep='\t', header=None)
            adata.obs_names = barcodes_df[0].values

            # --- MAPPING RELAZIONALE ---
            hk_found = [g for g in HK_GENES['human'] if g in adata.var_names]
            if not hk_found:
                print(f"    [!] Nessun gene Housekeeping trovato in {folder.name}. Salto.")
                continue

            C_global = np.array(adata[:, hk_found].X.sum(axis=1)).flatten()
            C_global[C_global == 0] = 1e-6

            df_rel = pd.DataFrame(index=adata.obs_names)
            for g in ONCO_GENES['human']:
                if g in adata.var_names:
                    # corretto sc.issparse in issparse
                    abs_val = np.array(adata[:, g].X.toarray() if issparse(adata.X) else adata[:, g].X).flatten()
                    df_rel[f'z_{g}'] = abs_val / C_global
                    df_rel[f'abs_{g}'] = abs_val
                else:
                    df_rel[f'z_{g}'] = 0.0
                    df_rel[f'abs_{g}'] = 0.0

            df_rel['Sample'] = folder.name
            df_rel['Species'] = 'human'
            df_rel['Is_Tumor'] = 1

            all_cells.append(df_rel)

            del adata, C_global, df_rel
            gc.collect()

        except Exception as e:
            print(f"    [!] Errore critico in {folder.name}: {e}")

    if all_cells:
        final_df = pd.concat(all_cells)
        final_df.to_csv("human_relational_dataset.csv")
        print(f"\n[+] UMANO COMPLETATO: {len(final_df)} cellule estratte e salvate.")
    else:
        print("\n[!] Nessuna cellula umana estratta.")

process_human_cid_folders(HUMAN_DIR)



# =========================================================
# 2. FRONTE TOPO: MAPPING RELAZIONALE
# =========================================================

def process_mouse_folders(base_path):
    print("\n[+] Inizio estrazione relazionale: Fronte TOPO (4 Campioni)")
    all_cells = []

    # Trova tutte le cartelle nel path del topo (TNBC1, TNBC2, TNBC3, filtered...)
    mouse_folders = [f for f in base_path.iterdir() if f.is_dir()]

    for i, folder in enumerate(mouse_folders):
        print(f"  [{i+1}/{len(mouse_folders)}] Infiltrazione in {folder.name}...")
        try:
            # Ricerca dinamica dei file (ignora se sono .gz o meno)
            mtx_file = next(folder.glob("matrix.mtx*"), None)
            genes_file = next(folder.glob("genes.tsv*"), None) or next(folder.glob("features.tsv*"), None)
            barcodes_file = next(folder.glob("barcodes.tsv*"), None)

            if not (mtx_file and genes_file and barcodes_file):
                print(f"    [!] Formato 10x non riconosciuto in {folder.name}. Salto.")
                continue

            # Caricamento della matrice
            adata = sc.read_mtx(mtx_file).T

            # Gestione dinamica dei nomi dei geni
            genes_df = pd.read_csv(genes_file, sep='\t', header=None)
            gene_col = 1 if genes_df.shape[1] > 1 else 0
            adata.var_names = genes_df[gene_col].values
            adata.var_names_make_unique()

            barcodes_df = pd.read_csv(barcodes_file, sep='\t', header=None)
            adata.obs_names = barcodes_df[0].values

            # --- MAPPING RELAZIONALE ---
            hk_found = [g for g in HK_GENES['mouse'] if g in adata.var_names]
            if not hk_found:
                print(f"    [!] Nessun gene Housekeeping trovato. Salto.")
                continue

            C_global = np.array(adata[:, hk_found].X.sum(axis=1)).flatten()
            C_global[C_global == 0] = 1e-6

            df_rel = pd.DataFrame(index=adata.obs_names)
            for h_gene, m_gene in zip(ONCO_GENES['human'], ONCO_GENES['mouse']):
                if m_gene in adata.var_names:
                    abs_val = np.array(adata[:, m_gene].X.toarray() if issparse(adata.X) else adata[:, m_gene].X).flatten()
                    # Creiamo la colonna col nome UMANO (es. z_MYC) per garantire l'allineamento perfetto tra le specie!
                    df_rel[f'z_{h_gene}'] = abs_val / C_global
                    df_rel[f'abs_{h_gene}'] = abs_val
                else:
                    df_rel[f'z_{h_gene}'] = 0.0
                    df_rel[f'abs_{h_gene}'] = 0.0

            df_rel['Sample'] = folder.name
            df_rel['Species'] = 'mouse'

            all_cells.append(df_rel)

            del adata, C_global, df_rel
            gc.collect()

        except Exception as e:
            print(f"    [!] Errore critico in {folder.name}: {e}")

    if all_cells:
        final_df = pd.concat(all_cells)
        final_df.to_csv("mouse_relational_dataset.csv")
        print(f"\n[+] TOPO COMPLETATO: {len(final_df)} cellule estratte e salvate.")
    else:
        print("\n[!] Nessuna cellula murina estratta.")

# Esecuzione Fronte Topo
process_mouse_folders(TOPO_DIR)
