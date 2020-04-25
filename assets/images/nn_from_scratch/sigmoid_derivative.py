import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

fig, ax = plt.subplots(1)
plt.ylim(top=1, bottom=-0.5)
plt.xlim(left=-5, right=5)

ax.set_title('The Sigmoid And Its Derivative')

x = np.arange(-5, 5, 0.01)
ppl.plot(ax, x, sigmoid(x), label='sigmoid(x)', linewidth=1.0)
ppl.plot(ax, x, dsigmoid(x), label='sigmoid(x)*(1-sigmoid(x))', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')


ppl.legend(ax)

fig.savefig('sigmoid_derivative.png')
