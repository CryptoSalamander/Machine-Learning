"""Softmax."""
import numpy as np
scores = [3.0, 1.0, 0.2]

def softmax(x):
    """Conpute softmax values for x."""
    pass # TODO : Compute and return softmax()
print(softmax(scores))

import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x,np.ones_like(x),0.2*np.ones_like(x)])

plt.plot(x,softmax(scores),linewidth=2)
plt.show()

