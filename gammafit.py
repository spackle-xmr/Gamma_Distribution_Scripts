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
printy = gamma.fit(x, floc=0)
print(printy)
#4 gives 9.13717, 1.2944
#4.2 gives 9.952, 1.191 with ANN Bin Class 77% 
#5 gives 8.27, 1.426 
#5.2 gives 8.948, 1.321
