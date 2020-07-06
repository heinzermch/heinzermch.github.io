import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

x_max = 1000
fig, ax = plt.subplots(1)
plt.ylim(top=500, bottom=0)
plt.xlim(left=0, right=x_max)

ax.set_title('Pythagorean Means')

def arithmetic_mean(elems):
   return sum(elems)/len(elems)

def harmonic_mean(elems):
   return len(elems) / sum([1/e for e in elems])

def geometric_mean(elems):
   p = 1
   for e in elems:
      p *= e ** (1/len(elems))
   return p


def apply_mean(n_max, mean_func):
   elems = [[i for i in range(1,n+1)] for n in range(1, n_max+1)]
   return np.array([mean_func(e) for e in elems])

x = np.array([n for n in range(1, x_max+1)])
ppl.plot(ax, x, apply_mean(x_max,arithmetic_mean), label='Arithmetic Mean', linewidth=1.0)
ppl.plot(ax, x, apply_mean(x_max, geometric_mean), label='Geometric Mean', linewidth=1.0)
ppl.plot(ax, x, apply_mean(x_max, harmonic_mean), label='Harmonic Mean', linewidth=1.0)

ppl.legend(ax)

fig.savefig('pythagorean_means.png')
