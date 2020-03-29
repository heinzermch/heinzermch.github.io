import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
plt.ylim(top=2, bottom=-2)
plt.xlim(left=-2, right=2)

def maxmin(x):
    r = np.zeros(x.shape, dtype=np.float)
    for i in range(len(x)):
        x_i = x[i]
        r[i] = max(0.0, min(1.0, x_i))
    return r


ax.set_title('Maxmin')

x = np.arange(-2, 2, 0.01)
ppl.plot(ax, x, maxmin(x), label='maxmin(x)', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')
plt.axvline(x=0, ymin=-2, ymax=2, linewidth=0.5, color='black')

ppl.legend(ax)

fig.savefig('maxmin_plot.png')
