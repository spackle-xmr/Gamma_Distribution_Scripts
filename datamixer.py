import numpy as np
import random
import matplotlib.pyplot as plt

def chunk_list(lst, n=16):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

#Create mixed dataset, some members from new most from old
newnumbers = []
with open('new-decoy-draws.txt', 'r') as f:
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


newslice = list(newnumbers)
mixed = list(oldnumbers)

for i in range(len(newslice)):
    mixed.append(newslice[i]) # Add slice to old
random.shuffle(mixed) # Shuffle order

mixslice = mixed[:2000000]

# Save data
with open('mix-decoys-2M.txt', 'w') as f:
    for line in mixslice:
        f.write(f"{line}\n")

# Sanity Check Plotting
old2d = chunk_list(oldnumbers)
mix2d = chunk_list(mixslice)

plt.xscale('log')
plt.yscale('log')

bins = np.logspace(np.log10(0.1), np.log10(10**9), num=40)

plt.hist(mixslice, bins=bins)
plt.hist(oldnumbers, bins=bins)
