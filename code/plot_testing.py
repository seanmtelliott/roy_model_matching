import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[0:1:.01, 0:1:.01]
rv = multivariate_normal([0, 0], [[1, 0.25], [0.25, 1]])
data = np.dstack((x, y))
z = np.exp(rv.pdf(data))
z2 = rv.pdf(data)
plt.contourf(x, y, z)
plt.close()
