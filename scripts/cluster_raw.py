"""
cluster_raw.py — Clustering K-Means sur fichiers .wav bruts
Usage:
    python scripts/cluster_raw.py            # K automatique
    python scripts/cluster_raw.py --k 6     # forcer K
    python scripts/cluster_raw.py --dry-run # simuler sans copier
"""

from glob import glob
import os, sys, shutil, argparse, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
K_MIN, K_MAX = 2, 10


def extract(file_path):
    """Lit le .wav directement (bypass load_audio qui attend FastAPI UploadFile)."""
    from backend.app.dsp.filter import low_pass_filter
    from backend.app.dsp.segmentation import split_signal
    from backend.app.features.spectral import extract_features

    signal, sr = librosa.load(file_path, sr=22050, mono=True)
    segments = split_signal(low_pass_filter(signal), sr)
    arrays = [list(extract_features(seg, sr).values()) for seg in segments]
    result = np.mean(arrays, axis=0)
    result = np.nan_to_num(result, nan=0.0)  # silence → 0 partout, valide pour K-Means psq j'avais des NaN dans centroid et bandwidth et des div par zéro dans zcr mais ça marche mieux que de les ignorer. À revoir si tu trouves une meilleure solution. et je pense que ça peut être un bon point de départ pour faire du data augmentation en ajoutant du bruit ou en modifiant la vitesse et en extrayant les features de ces versions augmentées, ça pourrait aider à rendre le modèle plus robuste.
    return result


# def find_k(X):
#     """Elbow method — choisit le meilleur K automatiquement."""
#     inertias = []
#     for k in range(K_MIN, K_MAX + 1):
#         inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)
#     deltas = np.diff(inertias)
#     return K_MIN + int(np.argmax(np.abs(np.diff(deltas))) + 1)
# méthode plus robuste que la simple elbow method qui peut être trompeuse si les inertias ne forment pas un coude clair, cette méthode normalise les inertias et mesure la distance de chaque point à la ligne droite formée entre K_MIN et K_MAX, le point avec la plus grande distance est considéré comme le coude optimal.
def find_k(X):
    inertias = []
    for k in range(K_MIN, K_MAX + 1):
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)
    
    # Normaliser et trouver le vrai coude
    inertias = np.array(inertias)
    norm = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    ks = np.arange(K_MIN, K_MAX + 1)
    # Distance de chaque point à la ligne droite K_MIN→K_MAX
    line = np.linspace(norm[0], norm[-1], len(norm))
    distances = np.abs(norm - line)
    optimal_k = ks[np.argmax(distances)]
    print(f"   Inertias: {[f'{v:.0f}' for v in inertias]}")
    return int(optimal_k)
# Alternative : silhouette score, mais plus lent à calculer, surtout pour K-Means qui n'est pas optimisé pour ça. La méthode du coude est généralement suffisante pour une première approche de clustering non supervisé, surtout si les données sont bien séparées. Si tu veux vraiment faire du fine-tuning, tu peux comparer les résultats de plusieurs méthodes (elbow, silhouette, gap statistic) et voir si elles convergent vers le même K optimal.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 1. Lister les .wav à la racine de raw/
    files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR)
             if f.lower().endswith(".wav") and os.path.isfile(os.path.join(RAW_DIR, f))]
    print(f"🎵 {len(files)} fichiers trouvés")

    # 2. Extraire les features
    X, valid = [], []
    for i, fp in enumerate(files):
        try:
            X.append(extract(fp))
            valid.append(fp)
            if (i+1) % 50 == 0: print(f"   {i+1}/{len(files)}...")
        except Exception as e:
            print(f"   ⚠️  Ignoré {os.path.basename(fp)}: {e}")

    if not X:
        print("❌ Aucune feature extraite — vérifie tes imports DSP.")
        sys.exit(1)

    X = StandardScaler().fit_transform(np.array(X))
    print(f"✅ Features extraites : {len(valid)} fichiers")

    # 3. Trouver K
    k = args.k or find_k(X)
    print(f"📌 K = {k}")

    # 4. Clustering
    labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)

    # 5. Résumé
    for cid in range(k):
        print(f"   cluster_{cid:02d} → {np.sum(labels == cid)} fichiers")

    if args.dry_run:
        print("⚠️  dry-run : aucun fichier copié.") ; return

    for old in glob(os.path.join(RAW_DIR, "cluster_*")):
        shutil.rmtree(old)  # ← nettoie avant de recréer
    # 6. Copier dans data/raw/cluster_XX/
    for fp, cid in zip(valid, labels):
        dest = os.path.join(RAW_DIR, f"cluster_{cid:02d}")
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(fp, os.path.join(dest, os.path.basename(fp)))

    print("✅ Fichiers copiés — renomme les dossiers avec les vrais labels !")


if __name__ == "__main__":
    main()