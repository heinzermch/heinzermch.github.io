import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
plt.ylim(top=5, bottom=0)
plt.xlim(left=0, right=1)

ax.set_title('Loss Functions on [0,1]')

x = np.arange(0.01, 1, 0.01)
ppl.plot(ax, x, -1*np.log(x), label='-log(x)', linewidth=1.0)
ppl.plot(ax, x, (1-x)**2, label='x^2', linewidth=1.0)
ppl.plot(ax, x, (1-x), label='|x|', linewidth=1.0)

ppl.legend(ax)

fig.savefig('loss_functions.png')
