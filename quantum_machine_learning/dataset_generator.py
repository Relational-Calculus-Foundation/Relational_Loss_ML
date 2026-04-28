import numpy as np
import pandas as pd
from pyscf import gto, scf

def generate_quantum_dataset(molecule_name, atom_1, atom_2, distances):
    """
    Simula l'equazione di Schrödinger (approssimazione Hartree-Fock)
    per estrarre le energie assolute della molecola a varie distanze.
    """
    data = []
    print(f"Simulazione {molecule_name} in corso...")
    
    for r in distances:
        # Costruiamo la geometria della molecola sull'asse Z
        geom = f"{atom_1} 0 0 0; {atom_2} 0 0 {r}"
        
        # Inizializziamo la molecola con una base quantistica standard (STO-3G)
        mol = gto.M(atom=geom, basis='sto-3g', verbose=0)
        
        # Calcoliamo lo stato fondamentale
        mf = scf.RHF(mol)
        mf.kernel()
        
        # ESTRAZIONE VARIABILI ASSOLUTE (La nostra materia prima)
        e_tot = mf.e_tot         # Energia Totale (Hartree)
        e_nuc = mol.energy_nuc() # Energia di Repulsione Nucleare (Hartree)
        e_elec = e_tot - e_nuc   # Energia Elettronica (Hartree)
        
        data.append({
            'Molecule': molecule_name,
            'Distance_A': r,
            'E_Nuclear': e_nuc,
            'E_Electronic': e_elec,
            'E_Total': e_tot
        })
        
    return pd.DataFrame(data)

# 1. GENERIAMO I DATI A BASSA COMPLESSITÀ (Train)
# Idrogeno (H-H): distanze da 0.4 a 2.5 Angstrom
distances_h2 = np.linspace(0.4, 2.5, 100)
df_h2 = generate_quantum_dataset("H2", "H", "H", distances_h2)

# 2. GENERIAMO I DATI AD ALTA COMPLESSITÀ (Test)
# Litio Idruro (Li-H): distanze da 1.0 a 4.0 Angstrom
distances_lih = np.linspace(1.0, 4.0, 100)
df_lih = generate_quantum_dataset("LiH", "Li", "H", distances_lih)

# Uniamo e salviamo
df_quantum = pd.concat([df_h2, df_lih], ignore_index=True)
df_quantum.to_csv("quantum_vqe_dataset.csv", index=False)

print("\nDataset Quantistico Generato! Anteprima H2 vs LiH:")
print(df_quantum.groupby('Molecule').mean())
