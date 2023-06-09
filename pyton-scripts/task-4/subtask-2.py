# Drawing graphics...


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(1, 101, 1)

y1 = np.sin(x)
y2 = np.cos(x)

plt.subplot(1, 2, 1)
plt.plot(x, y1)
plt.xlabel("Angle")
plt.ylabel("Sinus")
plt.title("Sinusoid")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, y2)
plt.xlabel("Angle")
plt.ylabel("Cosinus")
plt.title("Cosinusoid")
plt.grid(True)

plt.show()