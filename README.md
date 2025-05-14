[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

# FaceQCNet: Face Quality-Control Network

**FaceQCNet** è una parte di un progetto accademico sviluppato nell’ambito dell’iniziativa **FVAB** (Fondamenti di Visione Artificiale e Biometria) presso l’Università degli Studi di Salerno.  
Il progetto ha come obiettivo la **mitigazione del bias demografico** nei modelli di riconoscimento di attributi facciali soft (come genere, età, sorriso, colore capelli, ecc.), con particolare attenzione alle disuguaglianze tra gruppi etnici, di genere e di età.

FaceQCNet rappresenta la prima fase di una pipeline completa finalizzata al **riaddestramento del modello [Slim-CNN](https://github.com/gtamba/pytorch-slim-cnn)**, utilizzando immagini sintetiche di alta qualità, accuratamente filtrate e annotate.  
Questi dati bilanciati sono fondamentali per costruire un dataset *synthetic-aware* ed equo, utile per valutazioni comparative e mitigazione dei bias nei modelli biometrici.

---

## Tecnologie principali

- **[MagFace](https://github.com/IrvingMeng/MagFace)**  
  Utilizzato per valutare la **qualità delle immagini facciali** tramite la magnitudine dell'embedding.  
  Solo le immagini con qualità ≥ 20 vengono ammesse alla fase successiva di analisi.

- **[Facer](https://github.com/FacePerceiver/facer)**  
  Framework transformer-based per l’**estrazione automatica di attributi facciali soft**, preaddestrato su CelebA.  
  Fornisce una rappresentazione dettagliata del volto includendo attributi come colore capelli, età, genere, trucco, sorriso e altri 40+ soft labels.

---

## Obiettivo operativo

1. **Valutare automaticamente** la qualità di immagini sintetiche generate (es. da StyleGAN3 o Stable Diffusion)
2. **Scartare immagini di bassa qualità**
3. **Annotare automaticamente** le immagini valide con attributi CelebA via Facer
4. **Generare dataset annotati e bilanciati** da usare per il riaddestramento di Slim-CNN
5. **Mitigare il bias demografico** nei sistemi di analisi facciale soft

---

## Requisiti
Installa tutte le dipendenze con:

```bash
pip install -r requirements.txt
```

**Nota**: il pacchetto facer viene installato direttamente da GitHub.

---

## Come usare
### 1. Preparare le immagini sintetiche
Posiziona le immagini da analizzare in:

```bash
data/synthetic/
```

### 2. Valutare la qualità (MagFace)
Esegui:
```bash
python scripts/quality_check.py
```

- Salverà le immagini con punteggio ≥ 20 in `data/quality_images/`
- Crea anche un file `quality_scores.txt` con i punteggi ottenuti

### 3. Estrarre attributi facciali (Facer)
Esegui:

```bash
python scripts/extract_attributes.py
```

- Analizza tutte le immagini in `quality_images/`
- Per ogni immagine genera un file `.txt` in `attributes_detected/`
- Gli attributi includono colore dei capelli e tutti i soft attributes di CelebA

--- 

## Obiettivo
Questo progetto è parte dell’iniziativa **FVAB** per la mitigazione del bias demografico nei modelli di riconoscimento facciale soft.
L’obiettivo è garantire che solo immagini di qualità sufficiente vengano utilizzate per analisi di attributi, riducendo distorsioni dovute a qualità scarsa o squilibrata

---

## 📜 Licenza

Questo progetto è distribuito sotto licenza **Apache License 2.0**.  
Puoi liberamente utilizzarlo, modificarlo e ridistribuirlo, a patto di rispettare i termini della licenza.

Consulta il file [`LICENSE`](./LICENSE) per il testo completo.

---

### 📎 Terze parti

FaceQCNet integra codice e modelli provenienti da progetti open source:

- **[MagFace](https://github.com/IrvingMeng/MagFace)** – Licenza Apache 2.0  
- **[Facer](https://github.com/FacePerceiver/facer)** – Licenza MIT

Per maggiori dettagli, consulta il file [`NOTICE`](./NOTICE) incluso nel repository.
