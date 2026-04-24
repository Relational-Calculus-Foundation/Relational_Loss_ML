"""
Ollama + Relational Calculus Test Drive
---------------------------------------------------------
This script proves that using Relational Loss (predicting [0,1] ratios)
allows for instant CPU-based tuning of local LLM embeddings,
vastly outperforming standard Absolute Loss for regression tasks.
"""

import ollama
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import csv

# ---------------------------------------------------------
# 1. Load Data from CSV
# ---------------------------------------------------------
sentences = []
scores_list = []

print("Loading dataset...")
with open("company_data_500.csv", mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader) # Salta l'intestazione
    for row in reader:
        sentences.append(row[0])
        scores_list.append(float(row[1]))

scores = np.array(scores_list, dtype=np.float32)
print(f"Loaded {len(sentences)} sentences.")

# ---------------------------------------------------------
# 2. Extract Features using Local Ollama (all-minilm)
# ---------------------------------------------------------
print("Generating 384-dimensional embeddings via Ollama...")
embeddings = []
for i, text in enumerate(sentences):
    # Using the tiny all-minilm model for lightning-fast embeddings
    response = ollama.embeddings(model="all-minilm", prompt=text)
    embeddings.append(response["embedding"])
    print(f"  Processed {i+1}/{len(sentences)}", end='\r')

X = np.array(embeddings, dtype=np.float32)
print("\nEmbeddings generated successfully.\n")

# ---------------------------------------------------------
# 3. Prepare Targets: Absolute vs Relational
# ---------------------------------------------------------
y_abs = scores                # Absolute target: 0 to 100
y_rel = scores / 100.0        # Relational target: anchored to max capacity [0, 1]

# Split 70% Train, 30% Test
X_train, X_test, y_train_abs, y_test_abs, y_train_rel, y_test_rel = train_test_split(
    X, y_abs, y_rel, test_size=0.3, random_state=42
)

from sklearn.neural_network import MLPRegressor

# ---------------------------------------------------------
# 4. The "Naked Engine" Test: Pure SGD
# ---------------------------------------------------------
print("Training Neural Networks with Pure SGD (No Adam's adaptive tricks)...")

# solver='sgd' espone brutalmente la topologia del paesaggio della Loss (Hessiana).
# Senza i trucchi di normalizzazione di Adam, la scala 0-100 genererà gradienti
# che si auto-alimenteranno, facendo esplodere i pesi (divergenza).
# La scala [0,1] del Relazionale scivolerà dolcemente verso il minimo.

abs_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='sgd', max_iter=1000, learning_rate_init=0.1,
                         n_iter_no_change=1000, tol=1e-8, random_state=42)

rel_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='sgd', max_iter=1000, learning_rate_init=0.1,
                         n_iter_no_change=1000, tol=1e-8, random_state=42)

abs_model.fit(X_train, y_train_abs)
rel_model.fit(X_train, y_train_rel)

abs_pred = abs_model.predict(X_test)
rel_pred_ratio = rel_model.predict(X_test)
rel_pred_abs = rel_pred_ratio * 100.0

abs_rmse = np.sqrt(mean_squared_error(y_test_abs, abs_pred))
rel_rmse = np.sqrt(mean_squared_error(y_test_abs, rel_pred_abs))

print("-" * 50)
print(f"Final Test RMSE (Absolute):   {abs_rmse:.2f} points")
print(f"Final Test RMSE (Relational): {rel_rmse:.2f} points")
print("-" * 50)

# ---------------------------------------------------------
# 5. Visual Proof (Extracting internal loss curves)
# ---------------------------------------------------------
# Convertiamo le loss interne di Scikit-Learn in RMSE fisico (0-100) per un confronto equo
abs_learning_curve = np.sqrt(abs_model.loss_curve_)
# La loss relazionale è calcolata su [0,1], dobbiamo moltiplicarla per 100 per scalarla
rel_learning_curve = np.sqrt(np.array(rel_model.loss_curve_)) * 100.0

plt.figure(figsize=(9, 6))
plt.plot(abs_learning_curve, label=f"Absolute Loss Curve", color="red", lw=2.5)
plt.plot(rel_learning_curve, label=f"Relational Loss Curve", color="blue", lw=2.5)

plt.xlabel("Training Epochs (Adam)", fontsize=12)
plt.ylabel("Training Error (RMSE 0-100 scale)", fontsize=12)
plt.title("Ollama Embeddings: Internal Convergence Speed\n(Absolute vs Relational Loss)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("ollama_convergence_curve.png", dpi=150)
plt.show()
