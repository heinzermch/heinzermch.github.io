import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

def f(x: np.ndarray) -> np.ndarray:
    return (x-2)**2 + 1
def gf(x: float) -> float:
    return 2*(x-2)
def step(x: float, gamma: float) -> float:
    return x - gamma*gf(x)
x_left, x_right = -2, 6
y_bottom, y_top = 0, 11

fig, ax = plt.subplots(1)
plt.ylim(top=y_top, bottom=y_bottom)
plt.xlim(left=x_left, right=x_right)

ax.set_title('Gradient Descent on f')

x = np.arange(x_left, x_right, 0.01)
ppl.plot(ax, x, f(x), label='f(x)', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')

x_start = 5
runs = 10
gamma = 0.1
x_n = np.zeros(runs+1)
y_n = np.zeros(runs+1)
x_n[0] = x_start
y_n[0] = f(x_start)
for i in range(1, runs+1):
    x_n[i] = step(x_n[i-1], gamma)
    y_n[i] = f(x_n[i])
ax.scatter(x_n, y_n, facecolor='red', edgecolor='white', label='Gradient Descent Sequence')

ppl.legend(ax)

fig.savefig('gradient_descent.png')
