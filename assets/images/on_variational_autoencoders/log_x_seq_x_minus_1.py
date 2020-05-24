import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1)
plt.ylim(top=3, bottom=-5)
plt.xlim(left=0, right=3)

ax.set_title('A Visual Proof of log(x) <= x - 1')

x = np.arange(0.01, 3, 0.01)
ppl.plot(ax, x, np.log(x), label='log(x)', linewidth=1.0)
ppl.plot(ax, x, x-1, label='x-1', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')


ppl.legend(ax)

fig.savefig('log_x_seq_x_minus_1.png')
