import random
import numpy as np
import scipy.stats as stats
from scipy.stats import gamma

#Initialize
dataset = []
with open('42mix-decoys-2M.txt', 'r') as f:
    for line in f:
        clean_line = line.strip()
        if clean_line:  # skip empty lines
            try:
                dataset.append(int(clean_line))
            except ValueError:
                print(f"Ignoring invalid data on line: {clean_line}")

for i in range(len(dataset)):
    dataset[i] = np.log(dataset[i])
new_list = [x for x in dataset if x != 0]

#Fit distribution
print(gamma.fit(new_list, floc=0))
