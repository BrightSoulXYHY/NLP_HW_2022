from cProfile import label
import matplotlib.pyplot as plt
import numpy as np



data = np.loadtxt("perplexity.csv",delimiter=",")

plt.plot(data,label="perplexity")
plt.xlabel("iter")
plt.title("perplexity")
plt.show()