import torch
from torchvision import transforms
from PIL import Image
import os
from iresnet import iresnet100
from tqdm import tqdm  # <-- barra di avanzamento

# === CONFIGURAZIONE ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "magface_epoch_00025.pth")
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "synthetic")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "quality_images")
TXT_PATH = os.path.join(OUTPUT_DIR, "quality_scores.txt")

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

# === INFERENZA MULTIPLA ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    # Calcola l'indice iniziale basandosi sui file già presenti
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.jpg')]
    existing_indices = []
    for fname in existing_files:
        try:
            base = os.path.splitext(fname)[0]
            existing_indices.append(int(base))
        except ValueError:
            continue
    index = max(existing_indices, default=0) + 1

    # Scorri tutte le immagini di INPUT_DIR
    image_files = [f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc="📊 Valutazione immagini", unit="img"):
        img_path = os.path.join(INPUT_DIR, img_file)
        try:
            score = evaluate_image_quality(img_path)
            if score >= 20:
                new_name = f"{index}.jpg"
                output_path = os.path.join(OUTPUT_DIR, new_name)

                img = Image.open(img_path).convert('RGB')
                img.save(output_path, "JPEG")

                results.append(f"{new_name} - {score:.2f}")
                index += 1
        except Exception as e:
            print(f"⚠️ Errore con immagine {img_file}: {e}")

    with open(TXT_PATH, "w") as f:
        f.write("\n".join(results))

    print(f"✅ Elaborazione completata: {len(results)} immagini salvate in '{OUTPUT_DIR}'")
    print(f"📄 File di log creato: {TXT_PATH}")
