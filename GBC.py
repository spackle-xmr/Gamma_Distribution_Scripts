import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pickle


def chunk_list(lst, n=16):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337) 

# Standardize Dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GradientBoostingClassifier(
    n_estimators=325,          # Number of boosting stages (trees)
    learning_rate=0.1,         # Step size shrinkage to prevent overfitting
    max_depth=5,               # Maximum depth of individual trees
    min_samples_split=5,       # Minimum samples required to split a node
    random_state=1337            # Seed for reproducibility
)

# Train and Test
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open('GBC.pkl', 'wb') as f:
    pickle.dump(model, f)
