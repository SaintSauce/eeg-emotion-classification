import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv("./dataset/emotions.csv")

sample = data.loc[0, 'fft_0_b' : 'fft_749_b']

plt.figure(figsize = (16, 10))
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
#plt.show()

print(data['label'].value_counts())

label_dict = {'NEGATIVE' : 0, 'NEUTRAL' : 1, 'POSITIVE' : 2}

#preprocessing
def preproc(df):
    df = df.copy()
    df['label'] = df['label'].replace(label_dict)
    y = df['label'].copy()
    X = df.drop('label', axis = 1).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=50)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preproc(data)
print(X_train)

#model
inputs = tf.keras.Input(shape=(X_train.shape[1],))
expand_dims = tf.expand_dims(inputs, axis = 2)
gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)
flatten = tf.keras.layers.Flatten()(gru)
outputs = tf.keras.layers.Dense(3, activation="softmax")(flatten)
model = tf.keras.Model(inputs = inputs, outputs = outputs)
print(model.summary())

model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )]
)

model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

y_predict = np.array(list(map(lambda x : np.argmax(x), model.predict(X_test))))
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