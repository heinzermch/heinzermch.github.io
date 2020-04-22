import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

x_left, x_right = 0, 50
y_bottom, y_top = 0, 1.05

fig, ax = plt.subplots(1)
plt.ylim(top=y_top, bottom=y_bottom)
plt.xlim(left=x_left, right=x_right)

ax.set_title('Accuracy for different losses on 100 images')

x_values = [i for i in range(51)]
accuracy_ce = [0.13, 0.21, 0.2, 0.65, 0.6, 0.71, 0.74, 0.78, 0.81, 0.84, 0.83, 0.87, 0.82, 0.88, 0.84, 0.88, 0.87, 0.9, 0.9, 0.91, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
accuracy_mse = [0.06, 0.11, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.11, 0.12, 0.16, 0.13, 0.06, 0.15, 0.06, 0.23, 0.09, 0.19, 0.06, 0.23, 0.08, 0.27, 0.1, 0.3, 0.12, 0.34, 0.16, 0.37, 0.22, 0.37, 0.29, 0.41, 0.34, 0.42, 0.41, 0.46, 0.45, 0.5, 0.5, 0.52, 0.56, 0.53, 0.59, 0.58, 0.63, 0.61, 0.65, 0.62, 0.66]
accuracy_mses = [0.06, 0.06, 0.06, 0.09, 0.09, 0.09, 0.09, 0.09, 0.07, 0.08, 0.13, 0.12, 0.12, 0.13, 0.13, 0.14, 0.15, 0.15, 0.15, 0.15, 0.16, 0.17, 0.18, 0.18, 0.19, 0.19, 0.19, 0.19, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.23, 0.23, 0.24, 0.24, 0.24, 0.24, 0.24, 0.25, 0.25, 0.25, 0.25, 0.27, 0.27, 0.27, 0.29]


x = np.arange(x_left, x_right, 0.01)
ppl.plot(ax, x_values, accuracy_ce, label='Cross-Entropy', linewidth=1.0)
ppl.plot(ax, x_values, accuracy_mse, label='Mean Squared Error', linewidth=1.0)
ppl.plot(ax, x_values, accuracy_mses, label='Mean Squared Error Softmax', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')
ppl.plot(ax, x, 0*x+1, label="", linewidth=0.5, color='grey')

ppl.legend(ax)

fig.savefig('training_accuracy_multiple_losses.png')
