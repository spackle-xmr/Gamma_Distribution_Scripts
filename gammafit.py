import random
import numpy as np
import scipy.stats as stats
from scipy.stats import gamma

#Initialize
samplesize = 5000000 # Number of samples, 5M
x=np.zeros(samplesize)
for i in range(samplesize):
    selector = np.random.randint(160)
    if selector < 42:
        x[i] = np.random.gamma(4.315,(1/0.3751))
    else:
        x[i] = np.random.gamma(19.28,(1/1.61))

#Fit distribution
print(gamma.fit(x, floc=0))
