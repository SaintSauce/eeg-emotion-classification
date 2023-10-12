import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Load EEG data
data = pd.read_csv("./dataset/emotions.csv")

# Plotting Fourier Transform of the data (usage > show how AI can be useful)
plt.figure(figsize = (16, 10))
sample = data.loc[0, 'fft_0_b' : 'fft_749_b']
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
plt.show()

# Display label distribution
print(data['label'].value_counts())

# Label mapping
label_dict = {'NEGATIVE' : 0, 'NEUTRAL' : 1, 'POSITIVE' : 2}

# Preprocess input data for training
def preprocess_data(df):
    df = df.copy()                                          # ensures original dataframe not modified
    df['label'] = df['label'].replace(label_dict)           # label encoding (label -> num. in {0, 1, 2})
    y = df['label'].copy()                                  # label vector y -> only contains label column
    X = df.drop('label', axis = 1).copy()                   # feature matrix X -> every columns except label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(data)

# RNN model (using Keras)
inputs = tf.keras.Input(shape=(X_train.shape[1],))                      # num. of input features (columns) in data

""" 
    expects 3D input, but RNN, so every 
    data sets -> new feature (2D -> 3D)
"""

expand_dims = tf.expand_dims(inputs, axis = 2)                          # each feature acts as timestep w one feature (RNN)
gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)      # GRU -> 256 units (neurons) -> output > 3D tensor
flatten = tf.keras.layers.Flatten()(gru)                                # GRU layer output is 3D -> need 2D
outputs = tf.keras.layers.Dense(3, activation="softmax")(flatten)       # 3 neurons (number of classes)
model = tf.keras.Model(inputs = inputs, outputs = outputs)              # constructs the model

# Verify model structure
print(model.summary())

# Compile model
model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Hyperparameters
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50
)

accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(accuracy * 100))

# Test model for Confusion matrix
y_predict = np.array(list(map(lambda x : np.argmax(x), model.predict(X_test))))

# Confusion matrix
cm = confusion_matrix(y_test, y_predict)
clr = classification_report(y_test, y_predict, target_names=label_dict.keys())
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap="Blues")
plt.xticks(np.arange(3) + 0.5, label_dict.keys())
plt.yticks(np.arange(3) + 0.5, label_dict.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("Classification Report:\n----------------------\n", clr)

# Second test for ROC Curves
y_prob = model.predict(X_test)
y_predict = np.array(list(map(lambda x : np.argmax(x), y_prob)))

# ROC Curves
n_classes = 3
y_score = model.predict(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.figure(figsize=(8, 6))
lw = 2
colors = ['red', 'green', 'blue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for multi-class classification')
plt.legend(loc="lower right")
plt.show()