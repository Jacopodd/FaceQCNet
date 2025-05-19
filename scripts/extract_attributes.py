# import sys
# import os
# import torch
# import facer
# from tqdm import tqdm
#
# # === CONFIGURAZIONE ===
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "quality_images")
# OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "attributes_detected")
#
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# face_detector = facer.face_detector("retinaface/mobilenet", device=device)
# face_attr = facer.face_attr("farl/celeba/224", device=device)
# labels = face_attr.labels
# hair_colors = ['Gray_Hair', 'Brown_Hair', 'Black_Hair', 'Blond_Hair']
#
# # Contatori individuali
# hair_counts = {
#     'Gray_Hair': 0,
#     'Brown_Hair': 0,
#     'Black_Hair': 0,
#     'Blond_Hair': 0
# }
#
# def analyze_image(image_path):
#     try:
#         image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)
#
#         with torch.inference_mode():
#             faces = face_detector(image)
#             faces = face_attr(image, faces)
#
#         if len(faces["attrs"]) == 0:
#             return "Nessun volto rilevato."
#
#         face1_attrs = faces["attrs"][0]
#         lines = []
#
#         lines.append("COLORE CAPELLI:")
#         for color in hair_colors:
#             idx = labels.index(color)
#             prob = face1_attrs[idx].item()
#             if prob > 0.5:
#                 hair_counts[color] += 1
#                 lines.append(f"- {color.replace('_', ' ')} ({prob:.2f})")
#
#         lines.append("")
#         lines.append("Attributi rilevati:")
#         for prob, label in zip(face1_attrs, labels):
#             if prob > 0.5:
#                 lines.append(f"{label}: {prob.item():.2f}")
#
#         return "\n".join(lines)
#
#     except Exception as e:
#         return f"Errore nell'elaborazione: {e}"
#
# # === ANALISI MULTIPLA ===
# if __name__ == "__main__":
#     image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#
#     for img_file in tqdm(image_files, desc="Analizzando immagini"):
#         img_path = os.path.join(INPUT_DIR, img_file)
#         output_file = os.path.join(OUTPUT_DIR, os.path.splitext(img_file)[0] + ".txt")
#
#         result_text = analyze_image(img_path)
#
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(result_text)
#
#     print(f"Analisi completata: {len(image_files)} immagini elaborate.")
#     print(f"File di output salvati in '{OUTPUT_DIR}'\n")
#     print("Conteggio colori capelli rilevati:")
#     for color in hair_colors:
#         print(f"- {color.replace('_', ' ')}: {hair_counts[color]}")


import sys
import os
import torch
import facer
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms

# === CONFIGURAZIONE ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "quality_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "attributes_detected")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = facer.face_detector("retinaface/mobilenet", device=device)
face_attr = facer.face_attr("farl/celeba/224", device=device)
labels = face_attr.labels
hair_colors = ['Gray_Hair', 'Brown_Hair', 'Black_Hair', 'Blond_Hair']

# === FairFace Setup ===
races = ['White', 'Black', 'Latino_Hispanic', 'East_Asian', 'Southeast_Asian', 'Indian', 'Middle_Eastern']
genders = ['Male', 'Female']
ages = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

fairface_model = models.resnet34()
fairface_model.fc = torch.nn.Linear(fairface_model.fc.in_features, 18)
fairface_model.load_state_dict(torch.load("../models/res34_fair_align_multi_7_20190809.pt", map_location='cpu'))
fairface_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Contatori
hair_counts = {color: 0 for color in hair_colors}

# === FUNZIONE ANALISI ===
def analyze_image(image_path):
    output_lines = []

    try:
        # Facer
        image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)

        with torch.inference_mode():
            faces = face_detector(image)
            faces = face_attr(image, faces)

        if len(faces["attrs"]) == 0:
            return "Nessun volto rilevato."

        face1_attrs = faces["attrs"][0]
        output_lines.append("COLORE CAPELLI:")
        for color in hair_colors:
            idx = labels.index(color)
            prob = face1_attrs[idx].item()
            if prob > 0.5:
                hair_counts[color] += 1
                output_lines.append(f"- {color.replace('_', ' ')} ({prob:.2f})")

        output_lines.append("\nAttributi rilevati:")
        for prob, label in zip(face1_attrs, labels):
            if prob > 0.5:
                output_lines.append(f"{label}: {prob.item():.2f}")

    except Exception as e:
        output_lines.append(f"Errore nell'elaborazione Facer: {e}")

    # === FairFace (aggiunta alla fine)
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = fairface_model(input_tensor)[0]
            race = races[torch.argmax(output[0:7])]
            gender = genders[torch.argmax(output[7:9])]
            age = ages[torch.argmax(output[9:18])]

        output_lines.append("\n--- FairFace Prediction ---")
        output_lines.append(f"Razza stimata: {race}")
        output_lines.append(f"Genere stimato: {gender}")
        output_lines.append(f"Età stimata: {age}")

    except Exception as e:
        output_lines.append(f"Errore FairFace: {e}")

    return "\n".join(output_lines)

# === ANALISI MULTIPLA ===
if __name__ == "__main__":
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc="Analizzando immagini"):
        img_path = os.path.join(INPUT_DIR, img_file)
        output_file = os.path.join(OUTPUT_DIR, os.path.splitext(img_file)[0] + ".txt")

        result_text = analyze_image(img_path)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)

    print(f"Analisi completata: {len(image_files)} immagini elaborate.")
    print(f"File di output salvati in '{OUTPUT_DIR}'\n")
    print("Conteggio colori capelli rilevati:")
    for color in hair_colors:
        print(f"- {color.replace('_', ' ')}: {hair_counts[color]}")
