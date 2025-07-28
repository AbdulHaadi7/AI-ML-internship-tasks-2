# 📰 News Topic Classifier Using BERT

## 📌 Objective

The objective of this task is to fine-tune a pre-trained transformer model, specifically **BERT (bert-base-uncased)**, to classify news headlines into their respective topic categories using the **AG News Dataset**. The task covers the complete pipeline of training, evaluation, and deployment of a transformer-based text classification model.

---

## ⚙️ Methodology / Approach

### 🔹 1. Dataset
- **AG News Dataset** is used, which is available directly from the [🤗 Hugging Face Datasets](https://huggingface.co/datasets/ag_news) library.
- The dataset contains news headlines classified into 4 categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

### 🔹 2. Data Preprocessing
- Headlines are tokenized using the **BERT tokenizer (`bert-base-uncased`)**.
- Dataset is split into training and validation sets using `train_test_split`.
- Padding and truncation are applied to ensure uniform input lengths.

### 🔹 3. Model Fine-Tuning
- `AutoModelForSequenceClassification` is used to load a BERT model with 4 output labels.
- The model is fine-tuned using the **Hugging Face `Trainer` API**.
- Key training hyperparameters:
  - Learning rate: `2e-5`
  - Batch size: `16`
  - Epochs: `3`
  - Evaluation strategy: `epoch`

### 🔹 4. Evaluation
- Evaluation metrics:
  - **Accuracy**
  - **F1-Score (weighted)**
- Metrics are computed using the `evaluate` library.

### 🔹 5. Deployment
- The trained model can be deployed using **Gradio** for interactive predictions via a simple web interface.

---

## 📈 Key Results / Observations

- ✅ **High performance** on classification with:
  - **Accuracy:** ~94% (on validation set)
  - **F1-Score:** ~94% (weighted average)
- ✅ **Efficient training** on a relatively small dataset using transfer learning.
- ✅ **Model generalizes well**, handling various headline structures.
- 🚀 **Successfully integrated** into a minimal deployment interface (Streamlit/Gradio) for real-time inference.

---

## 🧠 Skills Gained

- ✅ Practical use of **Transformers** for NLP classification.
- ✅ Experience with **Transfer Learning** and **Fine-Tuning** pre-trained models.
- ✅ Hands-on application of **evaluation metrics** like Accuracy and F1-score.
- ✅ Building and deploying **interactive ML applications** using Streamlit or Gradio.

---
