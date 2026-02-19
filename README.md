# 🎭 Sarcasm Detector AI
A sophisticated Deep Learning web application that uses Natural Language Processing (NLP) to distinguish between genuine news headlines and satirical content.

## 🚀 Live Demo
**[Check out the Live App here!](https://sarcasm-detector-ai.streamlit.app/)**

## 🧠 Project Overview
This project leverages a **Long Short-Term Memory (LSTM)** neural network—a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies. This makes it ideal for analyzing the linguistic patterns and subtle context required to detect sarcasm in text.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow & Keras (LSTM)
* **Web Framework:** Streamlit (Custom UI with Glassmorphism)
* **Cloud Platform:** Streamlit Cloud (PaaS)
* **Version Control:** Git & GitHub

## ✨ Key Features
* **Real-Time Inference:** Instant sarcasm detection with a confidence score.
* **Modern UI:** Responsive design using custom CSS for a sleek, glassmorphic look.
* **Robust NLP Pipeline:** Implements professional text preprocessing including tokenization and sequence padding.

## 📂 Project Structure
* `app.py`: The main Streamlit dashboard script.
* `sarcasm_model.h5`: The trained LSTM model file.
* `tokenizer.pickle`: Saved tokenizer for consistent text processing.
* `requirements.txt`: Environment dependencies for cloud deployment.
* `.gitignore`: Configured to ignore cache files and large datasets.

## 🚀 Installation & Local Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Shreya-Gharat/sarcasm-detector.git](https://github.com/Shreya-Gharat/sarcasm-detector.git)
   cd sarcasm-detector
   
2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run app.py
