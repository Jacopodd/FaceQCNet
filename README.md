# FaceQCNet

**FaceQCNet** è un framework modulare per l'analisi avanzata della qualità e degli attributi facciali soft in immagini statiche. Combina pipeline di deep learning basate su **MagFace**, **Facer** (FaRL) e **FairFace** per effettuare inferenze affidabili e demograficamente sensibili, utili per scenari di fairness audit, bias mitigation e soft-biometric profiling.

---

## Obiettivo
FaceQCNet è uno strumento che ci aiuta a scegliere solo le immagini sintetiche nitide e realistiche, 
per essere usate in un progetto di intelligenza artificiale. 
Dopo aver selezionato le immagini migliori, analizza il volto di ciascuna per capire alcune caratteristiche, 
come il colore dei capelli, l’etnia, il genere e l’età. In questo modo possiamo sapere esattamente che tipo di 
persone stiamo rappresentando con queste immagini artificiali. 
Tutto questo ci serve per allenare meglio un modello di intelligenza artificiale, chiamato Slim-CNN, così che non 
sia ingiusto verso alcune categorie di persone (ad esempio donne o persone di pelle scura).

FaceQCNet è un modulo di prefiltraggio e annotazione semantica volto alla selezione di immagini sintetiche ad alta qualità, 
destinate alla mitigazione del bias nei modelli di riconoscimento di attributi facciali soft. 
Le immagini vengono selezionate solo se superano una soglia di qualità visiva (quality score ≥ 20), calcolata tramite il modello MagFace. 
Successivamente, ciascuna immagine viene analizzata mediante due pipeline:
- **Facer** per l’estrazione di attributi binari secondo lo schema CelebA (es. colore dei capelli, presenza di occhiali, sorriso);
- **FairFace** per la stima di attributi demografici sensibili: etnia, genere ed età.

Questa doppia annotazione consente di costruire un dataset sintetico demograficamente controllato. 
Il dataset risultante sarà impiegato per il riaddestramento della rete Slim-CNN, con l’obiettivo di migliorare l’equità delle predizioni, 
soprattutto nei confronti di gruppi demografici sottorappresentati nei dataset reali.

---


## 🧬 Architettura del Sistema

FaceQCNet è composto da tre pipeline interconnesse:

### 1. **MagFace Quality Scoring**
- Architettura: `iresnet100`
- Metodo: magnitudo del deep feature embedding (L2-norm)
- Soglia di accettazione predefinita: `||f||₂ ≥ 20.0`
- Uscita: immagini filtrate ad alta qualità
- Fonte: [MagFace – IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace)

### 2. **Facer Attribute Recognition**
- Backbone: `retinaface/mobilenet` (face detection)
- Classificatore: `farl/celeba/224` (multi-label attribute prediction)
- Dataset di riferimento: CelebA
- Uscita: probabilità binarie per 40+ attributi soft
- Fonte: [FacePerceiver/facer](https://github.com/FacePerceiver/facer)

### 3. **FairFace Demographic Classification**
- Modello: `resnet34` fine-tuned per task multi-classe
- Output: classificazione [etnia (7), genere (2), età (9 classi)]
- Dataset: FairFace, bilanciato per razza/genere
- Fonte: [dchen236/FairFace](https://github.com/dchen236/FairFace)

---

## 🛠 Requisiti e Setup

### Dipendenze principali

- `torch`, `torchvision`, `facer`
- `Pillow`, `tqdm`, `gdown`

### Installazione

```bash
git clone https://github.com/tuo-username/faceqcnet.git
cd faceqcnet
pip install -r requirements.txt
```

### Scaricamento modelli

- `magface_epoch_00025.pth` sarà scaricato automaticamente via `gdown`
- `res34_fair_align_multi_7_20190809.pt` va posto in `models/`

---

## 📈 Pipeline di Inferenza

### 1. Filtraggio qualità volto con MagFace

```bash
python script/quality_check.py
```

**Input:** `data/synthetic/`  
**Output:** `data/quality_images/` + `quality_scores.txt`

- Pre-processing: resize(112×112), normalize [-1, 1]
- Threshold decision based on L2-norm of embedding

### 2. Estrazione attributi (Facer + FairFace)

```bash
python script/extract_attributes.py
```

**Input:** `data/quality_images/`  
**Output:** `.txt` descrittivi in `data/attributes_detected/`

- Facer → parsing colore capelli, sorriso, trucco, barba, occhiali...
- FairFace → classificazione demografica robusta (età, genere, etnia)

---

## 📂 Organizzazione del Repository

```
faceqcnet/
│
├── models/                     # Contiene pesi pre-addestrati
├── data/
│   ├── synthetic/              # Immagini non valutate
│   ├── quality_images/        # Output filtrato da MagFace
│   └── attributes_detected/   # File testuali con predizioni
├── script/                       # Codice sorgente
│   ├── quality_check.py
│   └── extract_attributes.py
│   └── clean_outputs.py       # Pulisce le directory data/quality_images e attributes_detected
├── README.md
└── NOTICE
```

---

## 📤 Output Esemplificativo

```
COLORE CAPELLI:
- Black Hair (0.96)
- Blond Hair (0.54)

Attributi rilevati:
Smiling: 0.91
Wearing_Lipstick: 0.88
Heavy_Makeup: 0.65
Eyeglasses: 0.12

--- FairFace Prediction ---
Razza stimata: Southeast_Asian
Genere stimato: Female
Età stimata: 30-39
```

---

## 📜 Licenza

**FaceQCNet** è distribuito sotto licenza **Apache License 2.0**.

Componenti terze incluse:

- **MagFace** – Apache 2.0
- **Facer** – MIT License
- **FairFace** – Creative Commons Attribution 4.0 (CC BY 4.0)

Consulta il file [`NOTICE`](NOTICE) per dettagli completi sulle attribuzioni.

---

## 📚 Citazioni Originali

Se utilizzi questo framework in pubblicazioni accademiche, cita i seguenti lavori:

- MagFace: *CVPR 2021 – Meng et al.*
- Facer: *FaceXFormer/FaRL – 2023-2024*
- FairFace: *CVPR 2020 – K. Joumaa & D. Chen*

---

## 📩 Contatti

Jacopo de Dominicis  
MITIGAZIONE BIAS DEMOGRAFICO – Fondamenti di Visione e Biometria 2025  
Email: jacopodedominicisdeveloper@gmail.com

