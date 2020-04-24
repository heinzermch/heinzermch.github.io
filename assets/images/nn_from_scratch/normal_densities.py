import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

def normal_density(x, m=0, std=1):
    return np.exp(-1*((x-m)**2/(2*std**2)))/(np.sqrt(2*np.pi*std**2))


fig, ax = plt.subplots(1)
plt.ylim(top=0.5, bottom=-0.1)
plt.xlim(left=-5, right=5)

ax.set_title('Normal Densities')

x = np.arange(-5, 5, 0.01)
ppl.plot(ax, x, normal_density(x), label='N(0,1)', linewidth=1.0)
ppl.plot(ax, x, normal_density(x, std=2), label='N(0,2)', linewidth=1.0)
ppl.plot(ax, x, normal_density(x, m=2), label='N(2,1)', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')


ppl.legend(ax)

fig.savefig('normal_densities.png')
