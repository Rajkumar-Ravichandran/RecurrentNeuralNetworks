# Sentiment Analysis using Bidirectional LSTM on IMDB Dataset

**Overview**

This project implements a sentiment analysis model using a Bidirectional Long Short-Term Memory (Bi-LSTM) network to classify movie reviews as positive or negative. The model is trained on the IMDB dataset, a widely-used benchmark for sentiment analysis tasks. The main goal is to predict whether a given review is positive or negative based on the text.

**Dataset**

The IMDB dataset consists of 50,000 movie reviews, equally split between positive and negative reviews. Each review is encoded as a sequence of integers, where each integer represents a word in the vocabulary. In this project, the vocabulary size is limited to the top 10,000 most frequent words to reduce model complexity and training time.

**Key Features:**

- Training Set: 25,000 reviews (50% positive, 50% negative)
- Test Set: 25,000 reviews (50% positive, 50% negative)
- Review Length: Reviews are padded to a fixed length of 200 words to ensure uniformity in input size.

**Model Architecture**

The model consists of the following layers:

- Embedding Layer: This converts the input words into dense vectors of fixed size (128). Each word in the review is mapped to a vector from the embedding matrix.
- Bidirectional LSTM: The LSTM layer processes the input text, capturing long-term dependencies from both forward and backward directions. This improves the model's understanding of context in the reviews.
- Global Max Pooling: This layer pools the most important features from the output of the LSTM layer, enhancing Recurrent Neural Networks using Bidirectional LSTM (BI-LSTM) Networks
- Dropout Layer: This regularization technique prevents overfitting by randomly setting a fraction of input units to 0 during training.
- Dense Layer with Sigmoid Activation: The final layer produces a binary output (0 or 1), where 0 represents a negative sentiment and 1 represents a positive sentiment.

**Training Process**

The model is compiled using:

- Loss Function: Binary Cross-Entropy, appropriate for binary classification tasks.
- Optimizer: Adam, an efficient gradient-based optimization algorithm.
- Metrics: Accuracy, to track the performance of the model during training.
- The model is trained with a batch size of 32 and for 12 epochs. Early stopping is employed to prevent overfitting, and the best model is saved based on validation loss during training.

**Improvements and Optimizations**

Several optimizations are made to improve both the performance and efficiency of the model:

- Data Loading: tf.data is used for more efficient data pipeline handling, which improves training speed.
- Early Stopping: Training halts when validation loss does not improve for 3 consecutive epochs, preventing overfitting.
- Model Checkpointing: The best model based on validation loss is saved during training.
- Global Max Pooling: This layer is added after the LSTM to capture the most informative features and reduce the risk of overfitting.

**Results**

After training, the model’s performance is evaluated on the test set. The training history is plotted to visualize the loss and accuracy over epochs, providing insight into the model's learning process.

**Requirements**

1. TensorFlow 2.x
2. NumPy
3. Matplotlib

To install the required packages, you can use:
pip install tensorflow numpy matplotlib

**How to Run**

- Clone this repository.
- Ensure the IMDB dataset is available (TensorFlow automatically downloads it).
- Run the Python script to train the model.

**Accuracy Improvement using Bidirectional LSTM**

![image](https://github.com/user-attachments/assets/821a5599-9172-4dda-9f66-b0dad623a2bf)

