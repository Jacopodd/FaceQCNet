import torch
from torchvision import transforms
from PIL import Image
import os
from iresnet import iresnet100
from tqdm import tqdm
import gdown

# === CONFIGURAZIONE ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "magface_epoch_00025.pth")
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "synthetic")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "quality_images")
TXT_PATH = os.path.join(OUTPUT_DIR, "quality_scores.txt")

# === CONTROLLO E DOWNLOAD MODELLO ===
if not os.path.exists(MODEL_PATH):
    print("[↓] Modello non trovato. Avvio download da Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    file_id = "1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H"  # Sostituisci con il vero ID se cambia
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("[✔] Modello già presente, proseguo.")

# === CARICAMENTO MODELLO ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = iresnet100(pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=device)
full_state_dict = checkpoint['state_dict']

filtered_dict = {k.replace('features.module.', ''): v
                 for k, v in full_state_dict.items()
                 if k.startswith('features.module.')}

model.load_state_dict(filtered_dict, strict=False)
model.to(device)
model.eval()

print("[🚀] Modello caricato e pronto all'uso su", device)


# === PREPROCESSING ===
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

def evaluate_image_quality(image_path):
    input_tensor = load_image(image_path)
    with torch.no_grad():
        feat = model(input_tensor)
        quality_score = torch.norm(feat, p=2, dim=1).item()
    return quality_score

# === INFERENZA ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Leggi i file già processati
    already_processed = set()
    if os.path.exists(TXT_PATH):
        with open(TXT_PATH, "r") as f:
            for line in f:
                name_score = line.strip().split(" - ")[0]
                already_processed.add(name_score)

    results = []
    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc="📊 Valutazione immagini", unit="img"):
        if img_file in already_processed:
            continue  # già elaborata

        img_path = os.path.join(INPUT_DIR, img_file)
        try:
            score = evaluate_image_quality(img_path)
            if score >= 20:
                output_path = os.path.join(OUTPUT_DIR, img_file)
                if not os.path.exists(output_path):  # evita sovrascrittura
                    img = Image.open(img_path).convert('RGB')
                    img.save(output_path)

                results.append(f"{img_file} - {score:.2f}")
        except Exception as e:
            print(f"⚠️ Errore con immagine {img_file}: {e}")

    if results:
        with open(TXT_PATH, "a") as f:
            for r in results:
                f.write(r + "\n")

    print(f"✅ Elaborazione completata: {len(results)} nuove immagini salvate in '{OUTPUT_DIR}'")
    print(f"📄 File di log aggiornato: {TXT_PATH}")
