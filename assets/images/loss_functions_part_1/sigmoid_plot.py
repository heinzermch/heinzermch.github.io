import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
plt.ylim(top=2, bottom=-2)
plt.xlim(left=-5, right=5)

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))


ax.set_title('Sigmoid')

x = np.arange(-5, 5, 0.01)
ppl.plot(ax, x, sigmoid(x), label='sigmoid(x)', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')
plt.axvline(x=0, ymin=-2, ymax=2, linewidth=0.5, color='black')

ppl.legend(ax)

fig.savefig('sigmoid_plot.png')
