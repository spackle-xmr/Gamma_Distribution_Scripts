import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def chunk_list(lst, n=16):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def read_file_to_list(filename):
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line == '0':
                print('Removing 0 entry')
                break
            if clean_line:  # skip empty lines
                try:
                    lines.append(int(clean_line)) # Append valid integers to lines
                except ValueError:
                    print(f"Ignoring invalid data on line: {clean_line}")
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

# Chunk Data Into Rings
new2d = chunk_list(newnumbers, ringsize) 
old2d = chunk_list(oldnumbers, ringsize)

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

# Shuffle Dataset
paired = list(zip(Xunshuff, yunshuff)) # Pair elements from both lists
random.shuffle(paired) # Shuffle the pairs
Xlist, ylist = map(list, zip(*paired)) # Unpack back into separate lists
X = np.array(Xlist)
y = np.array(ylist)

# Split Data Into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Standardize Dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Neural Net
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(ringsize,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and Test
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)) # Train
loss, accuracy = model.evaluate(X_test, y_test) # Test

# Display Results
print(f'Test Accuracy: {accuracy:.4f}')
y_prediction = (model.predict(X_test) > 0.5).astype(int) # Predict
print('Confusion Matrix', confusion_matrix(y_test, y_prediction , normalize='pred'))
