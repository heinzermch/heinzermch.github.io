import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

x_left, x_right = 0, 30
y_bottom, y_top = 0, 1.05

fig, ax = plt.subplots(1)
plt.ylim(top=y_top, bottom=y_bottom)
plt.xlim(left=x_left, right=x_right)

ax.set_title('Accuracy on 100 images')

x_values = [i for i in range(31)]
y_values = [0.1, 0.26, 0.15, 0.34, 0.31, 0.29, 0.39, 0.28, 0.5, 0.66, 0.66, 0.73, 0.81, 0.79, 0.87, 0.85, 0.9, 0.83, 0.84, 0.76, 0.76, 0.66, 0.86, 0.95, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

x = np.arange(x_left, x_right, 0.01)
ppl.plot(ax, x_values, y_values, label='Accuracy', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')

ppl.legend(ax)

fig.savefig('training_accuracy.png')
