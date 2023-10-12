# eeg-emotion-classification

Project: Training RNN model to predict emotions based on EEG datasets

The following notebook details an approach to classify emotions from EEG (Electroencephalogram) data. We will leverage the potential of neural networks, particularly Recurrent Neural Networks (RNNs), to make this classification. The steps we'll undertake include:

Applying the Fast Fourier Transform (FFT) on the raw EEG data to convert the data from the time domain to the frequency domain.
Preprocessing the labeled data to suit the requirements of machine learning algorithms.
Implementing and training a Recurrent Neural Network using TensorFlow and Keras.
Evaluating the trained model on test data.
Using the model to predict emotions on new, unseen data.
