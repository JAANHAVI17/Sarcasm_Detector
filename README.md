# 🎭 Sarcasm Detector AI
### *An NLP-powered Deep Learning application for satirical content classification*

---

## 🚀 Live Demo
**[View Live Application →](https://sarcasm-detector-ai.streamlit.app/)**

---

## 🧠 Project Overview

This project applies **Deep Learning and Natural Language Processing** to solve a real-world text classification problem — distinguishing genuine news headlines from satirical ones.

At its core, the model uses a **Long Short-Term Memory (LSTM)** network, a type of Recurrent Neural Network (RNN) well-suited for sequential text data. LSTMs capture long-range linguistic dependencies, making them highly effective for understanding context and tone in language.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | LSTM (Recurrent Neural Network) |
| Web Framework | Streamlit |
| Deployment | Streamlit Cloud (PaaS) |
| Version Control | Git & GitHub |

---

## ✨ Key Features

- **Real-Time Inference** — Accepts free-text input and returns an instant sarcasm prediction with a confidence score.
- **End-to-End NLP Pipeline** — Implements tokenization and sequence padding for consistent, production-ready text preprocessing.
- **Custom UI** — Responsive Streamlit interface styled with custom CSS (Glassmorphism design pattern).
- **Cloud Deployed** — Live and accessible via Streamlit Cloud with zero setup required for end users.

---

## 📂 Project Structure

```
sarcasm-detector/
├── app.py               # Streamlit application entry point
├── sarcasm_model.h5     # Trained LSTM model (serialized)
├── tokenizer.pickle     # Fitted tokenizer for inference consistency
├── requirements.txt     # Project dependencies
└── .gitignore           # Excludes datasets and cache files
```


## 💡 What I Learned

- Designing and training LSTM networks for sequence classification tasks
- Building a full ML pipeline from data preprocessing to model deployment
- Deploying a machine learning app to the cloud using Streamlit Cloud
- Applying NLP techniques (tokenization, padding) for real-world inference

---

*Developed by [Janhavi Mestry](https://github.com/Janhavi-Mestry) & [Shreya Gharat](https://github.com/Shreya-Gharat)*
