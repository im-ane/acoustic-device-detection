"""
cluster_raw.py — Clustering K-Means sur fichiers .wav bruts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PIPELINE COMPLET :
    .wav bruts
        → extraction features (ton pipeline DSP)
        → normalisation (StandardScaler)
        → réduction dimensions (PCA)          ← NOUVEAU
        → recherche K optimal (Silhouette)    ← NOUVEAU
        → K-Means clustering
        → copie dans cluster_XX/

POURQUOI PCA ?
    Tes features (dominant_freq, energy, centroid, bandwidth, zcr, rms)
    sont souvent corrélées entre elles. Ex: energy et rms mesurent presque
    la même chose. PCA élimine ces redondances et garde seulement les
    "axes d'information pure". K-Means performe mieux dans un espace
    réduit et non-corrélé.

POURQUOI SILHOUETTE plutôt qu'ELBOW ?
    Elbow mesure l'inertie interne (compacité des clusters).
    Silhouette mesure les deux : compacité ET séparation entre clusters.
    Score de -1 à 1 — plus c'est proche de 1, mieux les clusters sont
    séparés. C'est beaucoup plus fiable pour des signaux acoustiques.

Usage:
    python scripts/cluster_raw.py            # K automatique (silhouette)
    python scripts/cluster_raw.py --k 5     # forcer K
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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
K_MIN, K_MAX = 2, 10


def extract(file_path):
    """
    Lit le .wav directement (bypass load_audio qui attend FastAPI UploadFile).
    Applique tout ton pipeline DSP : filtre → segmentation → features.
    Retourne un vecteur moyen sur tous les segments du fichier.
    nan_to_num gère les fichiers silencieux (division par zéro → 0).
    """
    from backend.app.dsp.filter import low_pass_filter
    from backend.app.dsp.segmentation import split_signal
    from backend.app.features.spectral import extract_features

    signal, sr = librosa.load(file_path, sr=22050, mono=True)
    segments = split_signal(low_pass_filter(signal), sr)
    arrays = [list(extract_features(seg, sr).values()) for seg in segments]
    result = np.mean(arrays, axis=0)
    return np.nan_to_num(result, nan=0.0)  # silence → vecteur de zéros


def apply_pca(X):
    """
    Réduit les dimensions avec PCA en gardant 95% de la variance.

    Exemple concret avec tes 6 features :
        Avant PCA : [dominant_freq, energy, centroid, bandwidth, zcr, rms]
        → energy et rms sont très corrélés (mesurent tous les deux l'intensité)
        → PCA les fusionne en 1 seul axe "intensité"
        Après PCA : 4-5 composantes indépendantes au lieu de 6 corrélées

    n_components=0.95 signifie "garde assez de composantes pour expliquer
    95% de la variance totale des données" — automatique, pas besoin de
    deviner combien de dimensions garder.
    """
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)
    n_original = X.shape[1]
    n_reduit   = X_pca.shape[1]
    variance   = sum(pca.explained_variance_ratio_) * 100
    print(f"   PCA : {n_original} features → {n_reduit} composantes "
          f"({variance:.1f}% variance conservée)")
    return X_pca


def find_k(X):
    """
    Trouve le K optimal via le Silhouette Score.

    Comment ça marche :
        Pour chaque fichier audio, le silhouette score mesure :
        - a = distance moyenne aux autres fichiers du MÊME cluster
        - b = distance moyenne aux fichiers du cluster LE PLUS PROCHE
        - score = (b - a) / max(a, b)

        score proche de  1 → le fichier est bien dans son cluster
        score proche de  0 → le fichier est à la frontière
        score proche de -1 → le fichier est mal classé

    On choisit le K qui maximise le score moyen sur tous les fichiers.
    """
    print(f"\n🔍 Recherche du K optimal (silhouette, K={K_MIN}→{K_MAX})...")
    best_k, best_score = K_MIN, -1

    for k in range(K_MIN, K_MAX + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score  = silhouette_score(X, labels)
        marker = " ← meilleur" if score > best_score else ""
        print(f"   K={k:2d}  silhouette={score:.4f}{marker}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"\n✅ K optimal : {best_k}  (silhouette={best_score:.4f})")
    return best_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",       type=int, default=None,
                        help="Forcer K (sinon auto via silhouette)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simuler sans copier les fichiers")
    args = parser.parse_args()

    # ── 1. Lister les .wav à la racine de raw/ (pas dans les sous-dossiers)
    files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR)
             if f.lower().endswith(".wav") and os.path.isfile(os.path.join(RAW_DIR, f))]
    print(f"🎵 {len(files)} fichiers .wav trouvés")

    # ── 2. Extraire les features via ton pipeline DSP
    print("⚙️  Extraction des features...")
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

    # ── 3. Normaliser — obligatoire avant PCA et K-Means
    # StandardScaler centre chaque feature (moyenne=0, écart-type=1)
    # Sans ça, dominant_freq (en Hz, valeurs ~1000) écraserait zcr (valeurs ~0.1)
    X = StandardScaler().fit_transform(np.array(X))
    print(f"✅ {len(valid)} fichiers normalisés")

    # ── 4. PCA — réduire et décorréler les features
    X = apply_pca(X)

    # ── 5. Trouver K ou utiliser celui forcé
    if args.k:
        k = args.k
        print(f"\n📌 K forcé à {k}")
    else:
        k = find_k(X)
    print(f"📌 K = {k}")

    # ── 6. K-Means final avec le K choisi
    # n_init=20 : lance K-Means 20 fois avec des centroïdes initiaux différents
    # et garde le meilleur résultat — évite les minima locaux
    labels = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=20).fit_predict(X)

    # ── 7. Résumé
    print(f"\n📦 Résultat (K={k}) :")
    for cid in range(k):
        print(f"   cluster_{cid:02d} → {np.sum(labels == cid)} fichiers")

    if args.dry_run:
        print("\n⚠️  dry-run : aucun fichier copié.")
        return

    # ── 8. Nettoyer anciens clusters puis copier
    for old in glob(os.path.join(RAW_DIR, "cluster_*")):
        shutil.rmtree(old)

    for fp, cid in zip(valid, labels):
        dest = os.path.join(RAW_DIR, f"cluster_{cid:02d}")
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(fp, os.path.join(dest, os.path.basename(fp)))

    print("\n✅ Fichiers copiés dans leurs clusters.")
    print("👉 Ouvre chaque dossier cluster_XX, écoute quelques fichiers")
    print("   et renomme le dossier avec le vrai label (ex: engine, silence...)")


if __name__ == "__main__":
    main()