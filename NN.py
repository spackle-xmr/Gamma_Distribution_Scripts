import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def split_and_sort(lst):
    result = []
    for i in range(0, len(lst), 16):
        chunk = lst[i:i+16]
        sorted_chunk = sorted(chunk)
        result.append(sorted_chunk)
    return result

def read_file_to_list(filename):
    lines = []
    skippeddata = 0
    with open(filename, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line == '0':
                skippeddata += 1
                break
            if clean_line:  # skip empty lines
                try:
                    lines.append(int(clean_line)) # Append valid integers to lines
                except ValueError:
                    skippeddata += 1
        print('Skipped Invalid Data Entries: ', skippeddata)
        return lines

tf.keras.backend.clear_session() # Refresh tensorflow session

# Get Data
newnumbers = read_file_to_list('new.txt')
oldnumbers = read_file_to_list('old.txt')

# Initialize Network Data
ringsize = 16
samplesize = min((len(newnumbers) // ringsize), (len(oldnumbers) // ringsize)) # Number of Complete Rings With 50/50 Split Between Source Files
X=np.zeros((samplesize,ringsize))
y=np.zeros(samplesize)

# Log Normalize Data
for i in range(samplesize * ringsize):
    newnumbers[i] = np.log(newnumbers[i])
    oldnumbers[i] = np.log(oldnumbers[i])

# Chunk Data Into Rings With Members Ordered By Age
new2d = split_and_sort(newnumbers) 
old2d = split_and_sort(oldnumbers)

# Only Include Complete Rings
new2d = new2d[:samplesize]
old2d = old2d[:samplesize]

# Combine New and Old Into Complete Dataset
Xunshuff=np.zeros((2 * samplesize, ringsize))
yunshuff=np.zeros(2 * samplesize)
for i in range(samplesize):
    Xunshuff[i] = old2d[i]
    yunshuff[i] = 0
    Xunshuff[samplesize+i] = new2d[i]
    yunshuff[samplesize+i] = 1
X = np.array(Xunshuff)
y = np.array(yunshuff)

# Split Data Into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13643) 

# Standardize Dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Neural Net
model = keras.Sequential([
    keras.layers.Dense(128, activation='mish', input_shape=(ringsize,)),
    keras.layers.Dense(128, activation='mish'),
    keras.layers.Dense(64, activation='mish'),
    keras.layers.Dense(32, activation='mish'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and Test
model.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=True, validation_data=(X_test, y_test)) # Train
loss, accuracy = model.evaluate(X_test, y_test) # Test

# Display Results
print(f'Test Accuracy: {accuracy:.4f}')
y_prediction = (model.predict(X_test) > 0.5).astype(int) # Predict
print('Confusion Matrix', confusion_matrix(y_test, y_prediction , normalize='pred'))
