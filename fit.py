import numpy as np
import scipy

rng = np.random.default_rng(np.random.PCG64(1))
noise = rng.normal(loc=0, scale=0, size=2000)

mean, std = scipy.stats.norm.fit(noise)
print(mean, std)
