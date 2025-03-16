import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

tf.keras.backend.clear_session() # Refresh tensorflow session

#Initialize
samplesize = 125000 # Number of Transactions
ringsize = 16
X=np.zeros((samplesize,ringsize))
y=np.zeros(samplesize)

def chunk_list(lst, n=16):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# Get Data
newnumbers = []
with open('42mix-decoys-2M.txt', 'r') as f:
    for line in f:
        clean_line = line.strip()
        if clean_line:  # skip empty lines
            try:
                newnumbers.append(int(clean_line))
            except ValueError:
                print(f"Ignoring invalid data on line: {clean_line}")

oldnumbers = []
with open('status-quo-decoy-draws-2M.txt', 'r') as f:
    for line in f:
        clean_line = line.strip()
        if clean_line:  # skip empty lines
            try:
                oldnumbers.append(int(clean_line))
            except ValueError:
                print(f"Ignoring invalid data on line: {clean_line}")

#Convert int to ln(int)
for i in range(len(oldnumbers)):
    oldnumbers[i] = np.log(oldnumbers[i])

for i in range(len(newnumbers)):
    newnumbers[i] = np.log(newnumbers[i])
    
new2d = chunk_list(newnumbers)
old2d = chunk_list(oldnumbers)

old2d.pop()
new2d.pop()

Xunshuff=np.zeros((2*samplesize,ringsize))
yunshuff=np.zeros(2*samplesize)

for i in range(len(old2d)):
    Xunshuff[i] = old2d[i]
    yunshuff[i] = 0
for i in range(len(new2d)):
    Xunshuff[len(old2d)+i] = new2d[i]
    yunshuff[len(old2d)+i] = 1

#shuffle X and y
unshuff_X2d = Xunshuff[:-2]
yunshuff = yunshuff[:-2]

Xunshuff = unshuff_X2d.tolist()

# Pair elements from both lists
paired = list(zip(Xunshuff, yunshuff))

# Shuffle the pairs
random.shuffle(paired)

# Unpack back into separate lists
Xlist, ylist = map(list, zip(*paired))

X = np.array(Xlist)
y = np.array(ylist)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=75614) # Split the data

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create ANN
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(ringsize,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='sgd', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)) # Training

loss, accuracy = model.evaluate(X_test, y_test) # Test

#Predict
y_prediction = model.predict(X_test)
for i in range(len(y_prediction)):
    if y_prediction[i] > 0.5:
        y_prediction[i] = 1
    else:
        y_prediction[i] = 0

#Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(y_test, y_prediction , normalize='pred')
print(f'Test Accuracy: {accuracy:.4f}')
print('Confusion Matrix')
print(result)

