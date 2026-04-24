"""
Dynamic Relational RAG: Zero-Shot Time & Inflation Invariance
---------------------------------------------------------
Task: Predict real estate prices based on text descriptions.
The Trap: The model is trained on 2015 prices (Low Scale).
We test it zero-shot on 2026 prices (High Scale / Massive Inflation).

Architecture:
- Retrieve Top-K similar properties from the active database.
- Absolute Model: Ignores context, predicts raw absolute price.
- Relational Model: Predicts the dimensionless ratio [0,1] relative
  to the Max Price in the retrieved context.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# ---------------------------------------------------------
# 1. Procedural Data Generation (FIXED)
# ---------------------------------------------------------
# Invece di puro rumore a 384 dimensioni, creiamo un "Segnale di Qualità" forte
# in modo che la Cosine Similarity trovi *davvero* case simili!

n_samples = 1000
# Valore intrinseco della casa (es. da 0.1 "baracca" a 1.0 "villa lusso")
intrinsic_quality = np.random.uniform(0.1, 1.0, n_samples)

# Generiamo embeddings dove la caratteristica principale è la qualità reale, più rumore
embeddings = np.zeros((n_samples, 384), dtype=np.float32)
embeddings[:, 0] = intrinsic_quality * 10.0  # Segnale forte
embeddings += np.random.normal(0, 0.5, (n_samples, 384)) # Rumore statistico

# Normalizziamo i vettori (come fanno Ollama o OpenAI)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# --- ERA 1: 2015 (Training Data) ---
prices_2015 = intrinsic_quality * 100_000

# --- ERA 2: 2026 (Test Data / Massive Inflation) ---
prices_2026 = intrinsic_quality * 1_000_000

X_train = embeddings[:800]
y_train_abs = prices_2015[:800]

X_test = embeddings[800:]
y_test_abs = prices_2026[800:]

# ---------------------------------------------------------
# 2. The RAG Retrieval Engine
# ---------------------------------------------------------
def retrieve_context(query_emb, database_embs, database_prices, top_k=3):
    """Simulates a Vector Database (like FAISS/Chroma)"""
    similarities = cosine_similarity(query_emb.reshape(1, -1), database_embs)[0]
    top_indices = np.argsort(similarities)[-top_k:]
    retrieved_prices = database_prices[top_indices]

    # The Dynamic North Star: Max price in the retrieved context
    local_max_price = np.max(retrieved_prices)
    return local_max_price

# ---------------------------------------------------------
# 3. Prepare Relational Targets (Anchored to Local Context)
# ---------------------------------------------------------
print("Building Relational Targets using 2015 RAG Context...")
y_train_rel = np.zeros_like(y_train_abs)

for i in range(len(X_train)):
    # Retrieve from the rest of the training set
    db_mask = np.arange(len(X_train)) != i
    local_max = retrieve_context(X_train[i], X_train[db_mask], y_train_abs[db_mask])

    # Target is dimensionless [0, 1] relative to its specific neighbors
    y_train_rel[i] = y_train_abs[i] / local_max

# ---------------------------------------------------------
# 4. Train Models (on 2015 Data)
# ---------------------------------------------------------
print("\nTraining on 2015 Economy (Low Prices)...")

abs_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', max_iter=500, random_state=42)

rel_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', max_iter=500, random_state=42)

abs_model.fit(X_train, y_train_abs)
rel_model.fit(X_train, y_train_rel)

# ---------------------------------------------------------
# 5. ZERO-SHOT INFERENCE ON 2026 DATA (RAG Augmented)
# ---------------------------------------------------------
print("\nTesting on 2026 Economy (10x Inflation - Unseen Scale)...")

abs_predictions = []
rel_predictions = []

for i in range(len(X_test)):
    query = X_test[i].reshape(1, -1)

    # Absolute Model: Blindly predicts based on 2015 memory
    pred_abs = abs_model.predict(query)[0]
    abs_predictions.append(pred_abs)

    # Relational Model: Uses RAG to fetch current 2026 market context
    # It retrieves from the TEST set (simulating a live, updated database)
    db_mask = np.arange(len(X_test)) != i
    current_market_max = retrieve_context(query[0], X_test[db_mask], y_test_abs[db_mask])

    # Predicts the ratio, then scales by today's market reality
    pred_ratio = rel_model.predict(query)[0]
    pred_rel_actual = pred_ratio * current_market_max
    rel_predictions.append(pred_rel_actual)

# ---------------------------------------------------------
# 6. Results & Evaluation
# ---------------------------------------------------------
abs_predictions = np.array(abs_predictions)
rel_predictions = np.array(rel_predictions)

abs_rmse = np.sqrt(mean_squared_error(y_test_abs, abs_predictions))
rel_rmse = np.sqrt(mean_squared_error(y_test_abs, rel_predictions))

print("\n" + "="*60)
print("📈 RAG SIMULATION: 10x MARKET INFLATION (Zero-Shot Transfer)")
print("="*60)
print(f"Absolute Model RMSE:   ${abs_rmse:,.0f} (Stuck in 2015)")
print(f"Relational RAG RMSE:   ${rel_rmse:,.0f} (Perfect Adaptation)")
print(f"Improvement:           {abs_rmse/rel_rmse:,.1f}x better")
print("="*60)
