# Next Word Predictor 

## Introduction

This Google Colab notebook presents a Next Word Predictor model using Natural Language Processing (NLP) techniques. The model is built to predict the next word in a given sentence or phrase. The notebook utilizes the following libraries and tools:

- **Spacy**: Used for text processing and tokenization.
- **en_core_web_sm**: A pre-trained English language model for NLP tasks.
- **Tokenizer**: From Keras, used for text tokenization.
- **NumPy**: Used for numerical operations.
- **Keras**: A deep learning framework used for building and training the model.
- **Sequential Model**: A type of Keras model used for building sequential neural networks.
- **Dense, LSTM, and Embedding Layers**: Keras layers for building the predictive model.
- **to_categorical**: A Keras utility for one-hot encoding labels.
- **pickle**: Used for saving and loading model weights and data.
- **load_model**: A Keras function for loading pre-trained models.
- **pad_sequences**: Used for padding sequences to a fixed length.
- **Random**: Used for generating random numbers for text generation.

## Data

The data used for training and testing the Next Word Predictor model is sourced from the "whale2.txt" file. This text file contains the text data necessary for training the language model.

## Model Architecture

The Next Word Predictor model is built using a neural network architecture with two layers of LSTM (Long Short-Term Memory) cells. LSTM layers are known for their ability to capture sequential patterns in text data, making them suitable for text generation tasks.

## Model Evaluation

After training, the model is evaluated using metrics such as loss and accuracy. In this case, the model achieved a loss of 43.36% and an accuracy of 90.21%, indicating that it can effectively predict the next word in a given text sequence.

## Using the Notebook

To use this notebook:

1. Ensure you have access to Google Colab and the required libraries mentioned above.
2. Upload the "whale2.txt" file to your Google Colab workspace or specify the correct path to the data file.
3. Run the notebook cells in order, following the instructions and code comments.
4. Once the model is trained and evaluated, you can use it to predict the next word in a text sequence.

Feel free to experiment with different text data and model parameters to enhance the next word prediction accuracy.

Enjoy exploring the world of Natural Language Processing with this Next Word Predictor!
