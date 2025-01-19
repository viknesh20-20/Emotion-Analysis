import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# Function to clean the column names by stripping extra spaces and converting to lowercase
def clean_column_names(df):
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df.columns = df.columns.str.lower()  # Convert to lowercase
    return df


# Function to check if 'text' column exists
def check_column(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataframe")
    return df


# Load and clean each CSV file
file1 = pd.read_csv('data/goemotions_1.csv')
file2 = pd.read_csv('data/goemotions_2.csv')
file3 = pd.read_csv('data/goemotions_3.csv')

# Clean the column names
file1 = clean_column_names(file1)
file2 = clean_column_names(file2)
file3 = clean_column_names(file3)

# Check if the 'text' column exists in each file
file1 = check_column(file1, 'text')  # Adjust column name if needed
file2 = check_column(file2, 'text')  # Adjust column name if needed
file3 = check_column(file3, 'text')  # Adjust column name if needed

# Combine the DataFrames
df = pd.concat([file1, file2, file3], ignore_index=True)

# Display the first few rows of the combined DataFrame to verify
print(df.head())

# Extract the features (X) and labels (y)
X = df['text'].values  # Assuming the text column is named 'text'

# Select only the emotion columns (drop unnecessary columns like 'author', 'subreddit', etc.)
emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                   'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                   'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                   'remorse', 'sadness', 'surprise', 'neutral']
y = df[emotion_columns].values  # Only use the emotion columns

# Ensure the labels are binary (0 or 1) - In case any label column has 'True'/'False' or other non-numeric values
y = np.where(y == True, 1, 0)  # Convert any 'True' values to 1 and 'False' values to 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad the text data
tokenizer = Tokenizer(num_words=10000)  # Limit to the top 10,000 words
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

maxlen = 50  # Maximum length of sequences
X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=maxlen)

# Ensure X_train_pad is of type np.ndarray
X_train_pad = np.array(X_train_pad, dtype=np.int32)
X_test_pad = np.array(X_test_pad, dtype=np.int32)

# Define a model using the Functional API
input_layer = layers.Input(shape=(maxlen,))
embedding_layer = layers.Embedding(input_dim=10000, output_dim=128)(input_layer)
lstm_layer = layers.LSTM(128)(embedding_layer)
dropout_layer = layers.Dropout(0.5)(lstm_layer)
dense_layer = layers.Dense(64, activation='relu')(dropout_layer)
output_layer = layers.Dense(y.shape[1], activation='sigmoid')(dense_layer)

model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Save the model
model.save('emotion_model.h5')

# Save the tokenizer for future use
import pickle

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer have been saved successfully!")
