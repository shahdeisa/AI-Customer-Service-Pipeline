# AI Customer Support & Chatbot Pipeline

An end-to-end modular Natural Language Processing (NLP) pipeline built with Python and Jupyter Notebooks designed for automated customer service workflows.

---

## Project Structure & Architecture

The project is split into four core experimental modules, designed to be executed sequentially or integrated into a unified pipeline:

| Module | Description | Key Tech / Models | Status |
| :--- | :--- | :--- | :--- |
| **1. Language Detection** | Identifies the input text language across 20+ supported languages. | Char + Word TF-IDF, Logistic Regression (`papluca/language-identification`) | Clean & Functional |
| **2. Sentiment Analysis** | Classifies customer emotional tone into 3 distinct categories. | DistilBERT (`dair-ai/emotion`), Hugging Face Trainer | Fixed & Integrated |
| **3. Intent Classification** | Maps customer messages to specific support intents. | TF-IDF, Logistic Regression (Bitext customer-support dataset) | Trained & Functional |
| **4. RAG Pipeline** | Retrieves relevant contextual data to ground responses. | MiniLM Embeddings, FAISS Vector Search | Modular Toy / Expandable |

---

## Key Features

* **Google Drive Integration:** Designed to save and load model checkpoints and datasets directly to/from Google Drive (`MODELS_DIR` & `DATACACHE_DIR`) for persistent, resumable training sessions.
* **Modular Design:** Independent notebooks allow isolation, testing, and debugging of each separate NLP task (Language, Sentiment, Intent, and Retrieval).
* **Evaluation Metrics:** Includes standard performance reporting (`classification_report`, accuracy, F1-scores) for classification modules.

---

## Getting Started

1. Open the notebooks in **Google Colab** (recommended to utilize GPU for transformer training).
2. Mount your Google Drive to enable model persistence:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
