# Next-Word Predictor (LSTM)

This project trains an LSTM-based language model to predict the next word given a text prompt.

## Features
- Tokenization with Keras `Tokenizer`
- Embedding layer with 100-d vectors
- Stacked LSTMs (first with `return_sequences=True`)
- Greedy and temperature-based sampling for text generation

## Quickstart

```bash
pip install -r requirements.txt
```

Train / run in the notebook:
1. Open `next_word_predictor_LSTM_clean.ipynb`
2. Run the preprocessing cells (tokenize, pad)
3. Train the model
4. Generate text using greedy or temperature sampling

## Important Implementation Notes
- Use `Embedding(input_dim=len(token.word_index)+1, ...)`
- When stacking LSTMs, set `return_sequences=True` on the first one
- For generation, use `token.index_word` to map ids back to words
- Wrap input texts in a list: `texts_to_sequences([text])`
- Keep the seed text **outside** the generation loop

## Example: Generation Loop

```python
text = "prediction"
for _ in range(10):
    seq = token.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")
    probs = model.predict(padded, verbose=0)[0]
    next_id = int(np.argmax(probs))  # or temperature sampling
    if next_id == 0: break
    next_word = token.index_word.get(next_id)
    if not next_word: break
    text += " " + next_word
print(text)
```

## License
MIT
