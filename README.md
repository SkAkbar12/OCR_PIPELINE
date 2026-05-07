# 🌿 Palm Leaf OCR using CNN + BiLSTM + CTC

A Deep Learning based OCR (Optical Character Recognition) pipeline for recognizing and transcribing ancient Palm Leaf Manuscripts using custom preprocessing, CNN feature extraction, BiLSTM sequence learning, and CTC decoding.

This project focuses on manuscript digitization and low-resource OCR experimentation for historical document analysis.

---

# 🚀 Features

* PDF to image conversion
* Manual image preprocessing
* Projection-based line segmentation
* CNN feature extraction
* BiLSTM sequence modeling
* CTC decoding
* OCR training and validation pipeline
* Custom convolution implementation

---

# 🧠 Technologies Used

* Python
* OpenCV
* NumPy
* PyTorch
* Matplotlib
* PyMuPDF (fitz)

---

# 📂 Project Structure

```bash
OCR_PIPELINE/
│
├── transcriptor_iitt.py
├── README.md
├── requirements.txt
├── labels.txt
└── sample_pages/
```

---

# ⚙️ Installation & Running

## 1️⃣ Clone Repository

```bash
git clone https://github.com/SkAkbar12/OCR_PIPELINE.git
```

---

## 2️⃣ Move into Project Directory

```bash
cd OCR_PIPELINE
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the OCR Pipeline

### Windows

```bash
python transcriptor_iitt.py
```

### Linux / macOS

```bash
python3 transcriptor_iitt.py
```

---

# 📜 Dataset Information

Currently, the model is trained only on:

* Page 6
* Page 7
* Page 8

English transliterations/translations of Palm Leaf Manuscripts.

The transcription labels are provided inside:

```bash
labels.txt
```

---

# ✍️ Important Notes

⚠️ The model is currently trained on only 3 manuscript pages.

For better OCR accuracy and improved generalization:

* Add more annotated manuscript pages
* Expand labels inside `labels.txt`
* Train on at least **50–100 pages** for significantly better results

---

# 🔤 Custom Transcription Experiment

You can experiment with:

* Sanskrit transliteration
* Sanskrit → English phonetic mapping
* Custom character vocabularies
* Multilingual OCR training

Update corresponding annotations inside:

```bash
labels.txt
```

---

# 🔬 OCR Pipeline Overview

## 1. PDF → Image Conversion

Converts manuscript PDF pages into images.

## 2. Image Preprocessing

Includes:

* Grayscale conversion
* Manual convolution filtering
* Histogram equalization
* Binary thresholding

## 3. Line Segmentation

Uses horizontal projection profiling for text line extraction.

## 4. Feature Extraction

Applies custom CNN-style convolution and pooling operations.

## 5. Sequence Learning

BiLSTM learns sequential dependencies from extracted features.

## 6. CTC Decoding

Performs alignment-free OCR prediction.

---

# 📈 Future Improvements

* Transformer-based OCR
* Sanskrit character support
* Attention mechanisms
* Larger manuscript datasets
* Better segmentation models
* Convolutional Autoencoders
* Data augmentation

---

# ⭐ Acknowledgment

This project was developed as part of research experimentation in OCR, Deep Learning, and historical manuscript digitization using Artificial Intelligence.
