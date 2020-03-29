import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
plt.ylim(top=4, bottom=-2)
plt.xlim(left=-2, right=4)


def array_wise_max(x):
    r = np.zeros(x.shape, dtype=np.float)
    for i in range(len(x)):
        x_i = x[i]
        r[i] = max([0.0, x_i])
    return r


ax.set_title('Softplus and ReLU')

x = np.arange(-2, 4, 0.01)
ppl.plot(ax, x, array_wise_max(x), label='ReLU(x)', linewidth=1.0)
ppl.plot(ax, x, np.log(1+np.exp(x)), label='Softplus(x)', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')
plt.axvline(x=0, ymin=-2, ymax=2, linewidth=0.5, color='black')

ppl.legend(ax)

fig.savefig('softplus_relu_plot.png')
