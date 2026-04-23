import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from backend.app.dataset.builder import build_dataset, load_dataset
from backend.app.models.train import train_model
from backend.app.core.config import DATA_DIR

BUILD = True  # ← False après le premier run pour réutiliser X.npy/y.npy

if __name__ == "__main__":

    # ── 1. Supprimer l'ancien dataset et reconstruire
    if BUILD:
        for f in ["X.npy", "y.npy"]:
            path = os.path.join(DATA_DIR, "datasets", f)
            if os.path.exists(path):
                os.remove(path)
                print(f"🗑️  Supprimé {f}")
        print("🔨 Construction du dataset depuis data/processed/...")
        X, y = build_dataset()
    else:
        print("📂 Chargement dataset existant...")
        X, y = load_dataset()

    print(f"📊 X={X.shape}, classes={set(y)}")

    # ── 2. Split 80% train / 20% test
    # stratify=y garantit que chaque classe est représentée
    # proportionnellement dans train et test
    from collections import Counter
    counts = Counter(y)
    small_classes = [c for c, n in counts.items() if n < 2]
    if small_classes:
        print(f"⚠️  Classes trop petites retirées : {small_classes}")
        mask = np.array([label not in small_classes for label in y])
        X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)  
    print(f"   Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # ── 3. Entraîner
    print("\n🤖 Entraînement RandomForest...")
    model, scaler = train_model(X_train, y_train)

    # ── 4. Évaluation sur le test set
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy : {acc*100:.1f}%")
    print("\n📋 Rapport par classe :")
    print(classification_report(y_test, y_pred))

    # ── 5. Taux d'apprentissage sur train (pour détecter overfitting)
    y_train_pred = model.predict(scaler.transform(X_train))
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"📈 Accuracy train : {train_acc*100:.1f}%")
    if train_acc - acc > 0.15:
        print("⚠️  Écart train/test > 15% — possible overfitting")
        print("   → Essaie d'augmenter les données ou réduire n_estimators")
    else:
        print("✅ Pas d'overfitting détecté")