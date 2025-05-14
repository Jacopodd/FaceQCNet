import os
import shutil

# === CONFIGURAZIONE ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_DIRS = [
    os.path.join(BASE_DIR, "..", "data", "quality_images"),
    os.path.join(BASE_DIR, "..", "data", "attributes_detected")
]

def clean_directory(directory):
    deleted = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                deleted += 1
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                deleted += 1
        except Exception as e:
            print(f"⚠️ Errore eliminando {file_path}: {e}")

    # Ricrea il file .gitkeep
    gitkeep_path = os.path.join(directory, ".gitkeep")
    open(gitkeep_path, 'w').close()

    return deleted

if __name__ == "__main__":
    for folder in TARGET_DIRS:
        print(f"🧹 Pulizia di: {folder}")
        count = clean_directory(folder)
        print(f"✅ {count} elementi rimossi e creato '.gitkeep' in '{folder}'\n")
