# RNN-Based YouTube Comment Sentiment Analyzer

A deep learning application that classifies Hinglish YouTube comments as **Positive or Negative** using recurrent neural networks and PyTorch.





## 📌 Project Overview

This project explores sentiment analysis using recurrent neural networks (RNNs) and transfer learning for text classification.

The model development process began with sentiment analysis on the IMDB movie-review dataset. The learned model was then fine-tuned on a Hinglish YouTube comments dataset to adapt it to the language patterns and sentiment expressions found in YouTube comments.

The final task is binary sentiment classification, where comments are categorized as Positive or Negative.

The project demonstrates a practical workflow involving text preprocessing, vocabulary construction, sequence encoding, neural network training, and domain-specific fine-tuning.

## ✨ Key Features

- **Sentiment Classification:** Predicts Positive or Negative sentiment for a text comment.
- **IMDB Initial Training:** Uses movie-review sentiment data in the initial model-training stage.
- **Hinglish Fine-Tuning:** Adapts the model to Hinglish YouTube comments.
- **Text Preprocessing:** Converts raw text into a format suitable for neural network processing.
- **Vocabulary Construction:** Builds a vocabulary and maps tokens to numerical indices.
- **Sequence Encoding:** Converts comments into fixed-length numerical sequences.
- **Padding and Unknown Tokens:** Handles varying comment lengths and words that are not present in the vocabulary.
- **PyTorch Implementation:** Uses PyTorch for neural network training and inference.

## 🗂️ Dataset

### Stage 1: IMDB Movie Reviews

The initial stage uses the IMDB movie-review sentiment dataset to develop a model for sentiment classification.

### Stage 2: Hinglish YouTube Comments

The model is subsequently fine-tuned using labeled Hinglish YouTube comments.

The original dataset contains three sentiment labels:

- Positive
- Negative
- Neutral

For the final binary classification task, Neutral samples were removed, leaving Positive and Negative comments.

### Hinglish Dataset Summary

| Attribute | Details |
|---|---|
| Data source | Hinglish YouTube comments |
| Original labels | Positive, Negative, Neutral |
| Final labels | Positive, Negative |
| Samples after filtering | 2,420 |
| Vocabulary size | 6,927 tokens |
| Maximum sequence length | 200 tokens |
| Positive samples reported | 1,427 |

## ⚙️ Project Workflow

1. **Initial Model Training:** Train the sentiment model using the IMDB movie-review dataset.
2. **Load Hinglish Dataset:** Read labeled YouTube comments.
3. **Filter Labels:** Remove Neutral samples to create a binary classification dataset.
4. **Preprocess Text:** Clean and prepare comments for tokenization.
5. **Construct Vocabulary:** Map recognized tokens to integer indices and handle unknown tokens.
6. **Encode Sequences:** Convert comments into numerical sequences.
7. **Pad and Truncate:** Standardize sequences to a maximum length of 200 tokens.
8. **Fine-Tune the Model:** Continue training using the Hinglish YouTube comments dataset.
9. **Evaluate Predictions:** Assess sentiment classification performance using evaluation data.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Machine Learning Concepts:** Recurrent Neural Networks, transfer learning, fine-tuning, binary classification
- **NLP Concepts:** Text preprocessing, tokenization, vocabulary construction, sequence encoding, padding and truncation

## 🧠 What I Learned

- How recurrent neural networks can process sequential text data.
- How to represent text numerically for neural network training.
- How initial training and fine-tuning can be used to adapt a model to a different text domain.
- Why preprocessing and vocabulary handling matter in NLP.
- How sentiment classification datasets are prepared for binary classification.
- The importance of evaluating a model after adapting it to a new dataset.

## 📉 Limitations

This is a learning-oriented project and is not yet intended for production use. The model may struggle with sarcasm, spelling variations, informal Hinglish expressions, and context-dependent sentiment.

Fine-tuning does not automatically guarantee better performance. The final model needs to be evaluated on held-out Hinglish YouTube comments to assess how well it generalizes.

No YouTube-model accuracy score is claimed here because the verified result has not been established.

## 🚀 Future Improvements

- Compare RNN, LSTM, and GRU architectures using consistent evaluation conditions.
- Tune learning rate, batch size, hidden dimensions, and training epochs.
- Monitor training and validation loss to identify overfitting or underfitting.
- Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Improve handling of Hinglish spellings, slang, and mixed-language expressions.
- Compare the fine-tuned model with a model trained only on Hinglish data.
- Improve the application interface and deployment experience.

## 🎯 Project Goal

The goal of this project is to gain practical experience in deep learning, NLP, sentiment classification, and adapting a neural network to a target text domain using PyTorch.

**Author:** Aishwari Bambale

**GitHub Profile:** [aishwari26](https://github.com/aishwari26)
