"""
download_dataset.py — Télécharge des sons depuis FreeSound API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT : Ne jamais mettre le token directement dans le code.
Utilise un fichier .env à la racine du projet :
    FREESOUND_TOKEN=ta_clé_ici

Usage:
    python scripts/download_dataset.py
"""

import os, sys, time, requests
from pathlib import Path

# ── TOKEN — lu depuis .env ou variable d'environnement
TOKEN = os.getenv("FREESOUND_TOKEN")
if not TOKEN:
    # Fallback : lire depuis .env manuellement
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FREESOUND_TOKEN="):
                TOKEN = line.split("=", 1)[1].strip()
if not TOKEN:
    print("❌ Token manquant — crée un fichier .env à la racine avec :")
    print("   FREESOUND_TOKEN=ta_clé_ici")
    sys.exit(1)

# ── CONFIG
RAW_DIR        = Path(__file__).parent.parent / "data" / "raw"
SOUNDS_PER_CLASS = 100      # nombre de sons par device
MIN_DURATION   = 3.0        # secondes minimum
MAX_DURATION   = 30.0       # secondes maximum

# ── CLASSES À TÉLÉCHARGER
# clé = nom du dossier (futur label), valeur = requête FreeSound
# CLASSES = {
#     # "pc_fan"      : "computer fan noise",
#     # "motor"       : "electric motor sound",
#     # "reactor"     : "industrial reactor hum",
#     # "ventilateur": "fan ventilation blower noise loop",
#     # "server"      : "server room datacenter",
#     # "television"  : "television static noise",
#     # "telephone"   : "telephone ringing",
#     # "silence"     : "room tone silence ambience",
#     # Dans download_dataset.py change les requêtes et augmente SOUNDS_PER_CLASS = 100

#     # "pc_fan"      : "computer fan noise hum",
#     "motor"       : "electric motor hum continuous",
#     "reactor"     : "industrial reactor hum drone",
#     "ventilateur" : "ventilation fan blower continuous",
#     "server"      : "server room datacenter hum",
#     "television"  : "television hum buzz electrical",
#     "telephone"   : "telephone device hum electrical buzz",  
#     "silence"     : "room ambience quiet indoor",

# }

CLASSES = {
    "motor"    : "electric motor hum continuous",
    "silence"  : "room ambience quiet indoor hum",
    "telephone": "telephone device electrical hum buzz",
}

BASE_URL = "https://freesound.org/apiv2"
HEADERS  = {"Authorization": f"Token {TOKEN}"}


def search(query, page=1):
    """Cherche des sons sur FreeSound."""
    r = requests.get(f"{BASE_URL}/search/text/", headers=HEADERS, params={
        "query"       : query,
        "fields"      : "id,name,duration,previews,license",
        "filter"      : f"duration:[{MIN_DURATION} TO {MAX_DURATION}]",
        "page_size"   : 15,
        "page"        : page,
    })
    r.raise_for_status()
    return r.json()


def download(sound_id, dest_path):
    """Télécharge le preview HQ d'un son (.mp3 → converti en .wav)."""
    # Récupère les infos du son
    r = requests.get(f"{BASE_URL}/sounds/{sound_id}/", headers=HEADERS)
    r.raise_for_status()
    info = r.json()

    # Utilise le preview HQ (mp3) — pas besoin de licence pour les previews
    preview_url = info["previews"]["preview-hq-mp3"]
    r = requests.get(preview_url, headers=HEADERS)
    r.raise_for_status()

    # Sauvegarde en .mp3 d'abord
    mp3_path = dest_path.with_suffix(".mp3")
    mp3_path.write_bytes(r.content)

    # Convertit en .wav avec ffmpeg
    os.system(f'ffmpeg -y -i "{mp3_path}" -ar 22050 -ac 1 "{dest_path}" -loglevel quiet')
    mp3_path.unlink()  # supprime le .mp3


def main():
    print("🎵 Téléchargement du dataset FreeSound")
    print(f"   {len(CLASSES)} classes × ~{SOUNDS_PER_CLASS} sons\n")

    for label, query in CLASSES.items():
        dest_dir = RAW_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Compte les fichiers déjà téléchargés
        existing = list(dest_dir.glob("*.wav"))
        if len(existing) >= SOUNDS_PER_CLASS:
            print(f"✅ {label} — déjà {len(existing)} fichiers, ignoré")
            continue

        print(f"📥 {label} ({query})...")
        downloaded = len(existing)
        page = 1

        while downloaded < SOUNDS_PER_CLASS:
            try:
                results = search(query, page=page)
            except Exception as e:
                print(f"   ⚠️  Erreur recherche: {e}")
                break

            if not results.get("results"):
                break

            for sound in results["results"]:
                if downloaded >= SOUNDS_PER_CLASS:
                    break
                sound_id  = sound["id"]
                dest_path = dest_dir / f"{label}_{sound_id}.wav"

                if dest_path.exists():
                    continue

                try:
                    download(sound_id, dest_path)
                    downloaded += 1
                    print(f"   [{downloaded}/{SOUNDS_PER_CLASS}] {sound['name'][:50]}")
                    time.sleep(0.3)  # respecte le rate limit FreeSound
                except Exception as e:
                    print(f"   ⚠️  Ignoré {sound_id}: {e}")

            page += 1
            if not results.get("next"):
                break

        print(f"   ✅ {downloaded} fichiers dans data/raw/{label}/\n")

    print("🎉 Téléchargement terminé !")

if __name__ == "__main__":
    main()