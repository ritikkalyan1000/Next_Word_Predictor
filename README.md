# Next Word Predictor (LSTM)

A deep learning NLP project that uses **stacked LSTMs** to predict the next word in a given text sequence.  
Built with **TensorFlow/Keras**, this project includes tokenization, sequence padding, model training, and text generation.

---

## 📌 Features
- **Keras Tokenizer** for vocabulary building and text preprocessing.
- **Embedding layer** to learn dense vector representations of words.
- **Stacked LSTMs** for sequence modeling:
  - First LSTM with `return_sequences=True`
  - Second LSTM outputs the final prediction
- **Greedy decoding** and **temperature sampling** for generating varied text.
- Supports custom datasets for training.

---

## 🛠 Technologies Used
- Python 3  
- TensorFlow / Keras  
- NumPy  
- scikit-learn  
- Matplotlib (for training visualizations)

---

## 🚀 How to Run

1. **Clone this repository**
```bash
git clone https://github.com/ritikkalyan1000/Next_Word_Predictor.git
cd Next_Word_Predictor
