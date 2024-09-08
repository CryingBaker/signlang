from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import TensorBoard
import os
import numpy as np

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            try:
                res = np.load(file_path, allow_pickle=True)
                if res is None or res.size == 0 or res.shape == ():  # Check if the data was not captured properly
                    print(f"Data not captured properly in file: {file_path}. Skipping.")
                    break
                if res.shape != (63,):  # Ensure the shape is consistent
                    print(f"Inconsistent shape in file: {file_path}, shape: {res.shape}. Skipping.")
                    break
                print(f"Loaded file: {file_path}, shape: {res.shape}")  # Print the dimensions of the loaded file
                window.append(res)
            except FileNotFoundError:
                print(f"File not found: {file_path}. Skipping.")
                break  # Exit the loop if a file is missing
        if len(window) == sequence_length:  # Ensure the window is complete before adding
            sequences.append(window)
            labels.append(label_map[action])

# Ensure sequences and labels are not empty
if not sequences or not labels:
    raise ValueError("No complete sequences found. Please check your data.")

# Convert sequences to a consistent shape and type
X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model
model = Sequential()
model.add(Input(shape=(sequence_length, X.shape[2])))  # Use Input layer as the first layer
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Corrected this line

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')