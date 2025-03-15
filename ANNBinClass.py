import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.keras.backend.clear_session() # Refresh tensorflow session

#Initialize
samplesize = 1000000 # Number of Transactions
ringsize = 16 # Size of ring signature
X=np.zeros((samplesize,ringsize))
y=np.zeros(samplesize)

# Generate Data
for i in range(samplesize):
    selector = np.random.randint(2) #Split 50-50 between old and new distributions
    if selector == 0:
        X[i] = np.random.gamma(19.28,(1/1.61), ringsize) # Draw 16 values from old DSA
        y[i] = 0
        replacer=random.sample(range(0,16),1)
        X[i][replacer] = np.random.gamma(4.315,(1/0.3751)) # Replace 1 random value with OSPEAD to represent user
    else:
        # All new
        X[i] = np.random.gamma(9.952,1.191, ringsize) # Draw 16 values from new DSA
        y[i] = 1
        
        '''
        # Part New
        X[i] = np.random.gamma(19.28,(1/1.61), ringsize)
        y[i] = 1
        partial = 16 #Number of decoys to draw from new distribution
        replacer=random.sample(range(0,16),partial)
        for j in range(partial):
            X[i][replacer[j]] = np.random.gamma(4.315,(1/0.3751))
        '''

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create ANN
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(ringsize,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)) # Training

loss, accuracy = model.evaluate(X_test, y_test) # Test
print(f'Test Accuracy: {accuracy:.4f}')
