# Detection-and-Suppression-of-Sensitive-Content-Exposure-to-Children-Using-Social-Network-Analytics

### **Author:** Sailesh  
**Model:** Fine-tuned FLAN-T5 (LoRA)  
**Goal:** Detect and prevent the spread of **sensitive / harmful content** in communities with children.

---

## 🧠 Overview

This project fine-tunes a lightweight **Flan-T5-Base** model using **LoRA adapters** to classify text as:

- **SENSITIVE**
- **SAFE**

It integrates with a **Social Network Analytics (SNA)** pipeline (using `NetworkX`) to:

- Score user behavior risk
- Identify repeat offenders
- **Block / flag** nodes that spread harmful content

---

## 📁 Folder Structure
```
Project/
├── flan_t5_sna_adapter/ # Fine-tuned LoRA adapter (model + tokenizer)
├── sna_predict_llm.py # Inference + node-level risk scoring
├── sft_data/
│ ├── train.jsonl # Weakly labeled training data
│ └── val.jsonl # Validation data
├── SNA.ipynb # Full training + evaluation notebook
└── README.md # Project documentation
```

---

## 🧩 Components

### **1️⃣ `sna_predict_llm.py`**
Handles:
- Loading **LoRA fine-tuned Flan-T5**
- Predicting **SENSITIVE** vs **SAFE**
- Returning prediction probabilities
- Calculating **node-level risk scores** using SNA features (e.g., PageRank, strikes)

---

### **2️⃣ `flan_t5_sna_adapter/`**
Contains:
- `adapter_config.json`
- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`

This is the **LoRA adapter** attached to Flan-T5-Base.

---

### **3️⃣ `SNA.ipynb`**
Includes:
- Dataset processing & weak labeling
- Fine-tuning (PEFT + HuggingFace Transformers)
- Model evaluation (Accuracy, Precision, Recall, F1)
- Saving the trained LoRA adapter

---

## ⚙️ Setup Instructions

### **1️⃣ Install Dependencies**
> GPU recommended

```bash
pip install torch transformers peft accelerate datasets evaluate sentencepiece networkx

flan_t5_sna_adapter/
├── config.json
├── adapter_config.json
├── pytorch_model.bin
└── tokenizer.json
