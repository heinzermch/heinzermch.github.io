import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
plt.ylim(top=5, bottom=-5)
plt.xlim(left=0, right=5)

ax.set_title('+/- log')

x = np.arange(0.01, 5, 0.01)
ppl.plot(ax, x, np.log(x), label='  log(x)', linewidth=1.0)
ppl.plot(ax, x, -1*np.log(x), label='- log(x)', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')


ppl.legend(ax)

fig.savefig('log_plot.png')
