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

Quick Prediction Test
from sna_predict_llm import predict_text_prob

text = "This post promotes violent or adult content."
p_sens, p_safe = predict_text_prob(text)

print(f"SENSITIVE: {p_sens:.3f}  |  SAFE: {p_safe:.3f}")

Node Risk Score Example
from sna_predict_llm import compute_node_risk

graph_feats = {"user42": {"pagerank": 0.12}}
recent_post_risks = [0.88, 0.72, 0.94]
strike_count = 2

risk = compute_node_risk("user42", recent_post_risks, graph_feats, strike_count)
print("Node Risk Score:", risk)

Integration (NetworkX SNA Pipeline)
from sna_predict_llm import predict_text_prob, compute_node_risk

for node, post in posts.items():
    p_sens, _ = predict_text_prob(post)
    
    node_strikes[node] = node_strikes.get(node, 0) + (1 if p_sens > 0.7 else 0)
    
    risk = compute_node_risk(node, [p_sens], graph_feats, node_strikes[node])

    if risk >= 1.2:
        print(f"🚫 Blocking node {node} (risk={risk:.2f})")


Posts ≥ 0.7 probability → Flagged

Cumulative risk ≥ 1.2 → Node blocked

Future Work

Replace weak labels with manually labeled dataset

Add multilingual support (Flan-T5-XL)

Deploy as REST API using FastAPI

Accelerate inference via int8 quantization
