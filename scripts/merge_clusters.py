"""
merge_clusters.py — Fusionne automatiquement les clusters avec les labels FreeSound
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMENT ÇA MARCHE :
    1. Calcule le vecteur moyen de features pour chaque cluster (cluster_00...)
    2. Calcule le vecteur moyen de features pour chaque label FreeSound (pc_fan...)
    3. Compare chaque cluster à chaque label → distance cosine
    4. Assigne le label le plus proche à chaque cluster
    5. Copie tout dans data/processed/label/

    Distance cosine mesure l'angle entre deux vecteurs de features.
    0 = identiques, 1 = complètement différents.
    C'est plus fiable qu'euclidienne pour des features audio normalisées.

RÉSULTAT :
    data/processed/
        pc_fan/       ← fichiers FreeSound + clusters assignés
        motor/
        silence/
        ...

Usage:
    python scripts/merge_clusters.py             # fusion automatique
    python scripts/merge_clusters.py --dry-run   # voir sans copier
"""

import os, sys, shutil, argparse
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_distances
from backend.app.dsp.filter import low_pass_filter
from backend.app.dsp.segmentation import split_signal
from backend.app.features.spectral import extract_features

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
THRESHOLD     = 0.3   # distance max pour accepter l'assignation (0=parfait, 1=différent)
                      # si distance > THRESHOLD → cluster mis dans "unknown"


def extract_file(file_path):
    """Extrait un vecteur de features d'un fichier .wav."""
    signal, sr = librosa.load(str(file_path), sr=22050, mono=True)
    segments = split_signal(low_pass_filter(signal), sr)
    arrays = [list(extract_features(seg, sr).values()) for seg in segments]
    result = np.mean(arrays, axis=0)
    return np.nan_to_num(result, nan=0.0)


def mean_vector(folder):
    """Calcule le vecteur moyen de tous les .wav d'un dossier."""
    files = list(Path(folder).glob("*.wav"))
    if not files:
        return None, []
    vectors = []
    for f in files:
        try:
            vectors.append(extract_file(f))
        except:
            pass
    return np.mean(vectors, axis=0) if vectors else None, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Distance max cosine (défaut: {THRESHOLD})")
    args = parser.parse_args()

    # ── 1. Identifier clusters et labels dans data/raw/
    all_dirs   = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    clusters   = [d for d in all_dirs if d.name.startswith("cluster_")]
    label_dirs = [d for d in all_dirs if not d.name.startswith("cluster_")]

    if not clusters:
        print("❌ Aucun dossier cluster_XX trouvé dans data/raw/")
        sys.exit(1)
    if not label_dirs:
        print("❌ Aucun dossier labelisé trouvé dans data/raw/")
        sys.exit(1)

    print(f"📂 {len(clusters)} clusters  +  {len(label_dirs)} labels FreeSound")

    # ── 2. Calculer les vecteurs moyens des labels FreeSound
    print("\n⚙️  Calcul des vecteurs moyens labels...")
    label_vectors = {}
    label_files   = {}
    for d in label_dirs:
        vec, files = mean_vector(d)
        if vec is not None:
            label_vectors[d.name] = vec
            label_files[d.name]   = files
            print(f"   {d.name:20s} → {len(files)} fichiers")

    # ── 3. Calculer les vecteurs moyens des clusters
    print("\n⚙️  Calcul des vecteurs moyens clusters...")
    cluster_vectors = {}
    cluster_files   = {}
    for d in clusters:
        vec, files = mean_vector(d)
        if vec is not None:
            cluster_vectors[d.name] = vec
            cluster_files[d.name]   = files
            print(f"   {d.name:20s} → {len(files)} fichiers")

    if not label_vectors or not cluster_vectors:
        print("❌ Pas assez de données pour comparer.")
        sys.exit(1)

    # ── 4. Comparer chaque cluster à chaque label (distance cosine)
    print("\n📊 Assignation clusters → labels :")
    print(f"   {'Cluster':15s}  {'Label assigné':20s}  {'Distance':10s}  {'Statut'}")
    print("   " + "─" * 60)

    assignments = {}   # cluster_name → label_name
    label_names = list(label_vectors.keys())
    label_matrix = np.array([label_vectors[l] for l in label_names])

    for cluster_name, cvec in cluster_vectors.items():
        distances = cosine_distances([cvec], label_matrix)[0]
        best_idx  = np.argmin(distances)
        best_label = label_names[best_idx]
        best_dist  = distances[best_idx]

        if best_dist <= args.threshold:
            assignments[cluster_name] = best_label
            status = "✅"
        else:
            assignments[cluster_name] = "unknown"
            status = f"⚠️  trop loin (>{args.threshold})"

        print(f"   {cluster_name:15s}  {best_label:20s}  {best_dist:.4f}      {status}")

    # ── 5. Résumé avant copie
    print(f"\n📦 Résultat fusion → data/processed/")
    all_labels = set(assignments.values()) | set(label_files.keys())
    for label in sorted(all_labels):
        n_freesound = len(label_files.get(label, []))
        n_cluster   = sum(len(cluster_files[c]) for c, l in assignments.items() if l == label)
        print(f"   {label:20s} → {n_freesound} FreeSound  +  {n_cluster} cluster")

    if args.dry_run:
        print("\n⚠️  dry-run : aucun fichier copié.")
        return

    # ── 6. Copier dans data/processed/
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Copier les fichiers FreeSound labelisés
    for label, files in label_files.items():
        dest = PROCESSED_DIR / label
        dest.mkdir(exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)

    # Copier les fichiers des clusters assignés
    for cluster_name, label in assignments.items():
        dest = PROCESSED_DIR / label
        dest.mkdir(exist_ok=True)
        for f in cluster_files.get(cluster_name, []):
            shutil.copy2(f, dest / f.name)

    print(f"\n✅ Fusion terminée dans data/processed/")
    print("👉 Lance ensuite : python scripts/train_model.py")


if __name__ == "__main__":
    main()