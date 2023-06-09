# Drawing graphics...


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(1, 101, 1)

y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1)
plt.xlabel("Angle")
plt.ylabel("Sinus")
plt.title("Sinusoid")

plt.plot(x, y2)
plt.grid(True)

plt.show()