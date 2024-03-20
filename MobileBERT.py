import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import MobileBertTokenizer, TFMobileBertForSequenceClassification

data = pd.read_csv('train.csv')
print(data.head())

tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
model = TFMobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=3)

# Convert text data to sequences
input_ids = tokenizer.batch_encode_plus(
    data['text'].tolist(),
    padding='longest',
    max_length=256,
    truncation=True,
    return_tensors='tf'
).input_ids

# Create labels
le = LabelEncoder()
labels = le.fit_transform(data['Y'])

# Convert labels to one-hot-encoded format
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=3)

# Split data into training and testing sets
total_size = len(input_ids)
train_size = int(total_size * 0.8)

input_ids_train = input_ids[:train_size]
input_ids_test = input_ids[train_size:]

one_hot_labels_train = one_hot_labels[:train_size]
one_hot_labels_test = one_hot_labels[train_size:]

# Convert the one-hot-encoded labels back to integer labels
integer_labels_train = np.argmax(one_hot_labels_train, axis=1)
integer_labels_test = np.argmax(one_hot_labels_test, axis=1)

# Convert labels back to their original values
integer_labels_train = le.inverse_transform(integer_labels_train)
integer_labels_test = le.inverse_transform(integer_labels_test)

# Compiling the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 32

# Train the model
history = model.fit(input_ids_train, one_hot_labels_train, validation_data=(input_ids_test, one_hot_labels_test), epochs=5, batch_size=batch_size)

# Make predictions on the testing set
predictions = model.predict(input_ids_test)

# Convert predictions to integer labels
predicted_labels = np.argmax(predictions, axis=1)

# Convert labels back to their original values
predicted_labels = le.inverse_transform(predicted_labels)

# Plot the confusion matrix
cm = confusion_matrix(integer_labels_test, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()
