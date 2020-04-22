import prettyplotlib as ppl
import numpy as np
import matplotlib.pyplot as plt

x_left, x_right = 0, 30
y_bottom, y_top = 0, 3

fig, ax = plt.subplots(1)
plt.ylim(top=y_top, bottom=y_bottom)
plt.xlim(left=x_left, right=x_right)

ax.set_title('Accuracy and Loss on 100 images')

x_values = [i for i in range(31)]
y_accuracy = [0.11, 0.13, 0.18, 0.23, 0.51, 0.48, 0.42, 0.57, 0.3, 0.44, 0.53, 0.57, 0.65, 0.77, 0.76, 0.79, 0.83, 0.86, 0.95, 0.97, 0.97, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0, 1.0]
y_loss = [255.94599793967663, 259.56856873514744, 232.58680051041057, 195.93351013644156, 172.48984386509912, 157.52182404712022, 188.41351862072756, 176.1604602830762, 209.87076037052393, 155.03264323738145, 153.03489382400835, 119.30388766185374, 101.6044473090725, 85.1050499064838, 76.3769126715133, 78.31107009704158, 65.16351518357952, 47.27848575663675, 22.47292661520279, 16.31354013564551, 13.350055798980017, 11.110888464914668, 9.483399068881356, 8.205443376048988, 7.237620707196308, 6.460034860957622, 5.807151326982063, 5.264531663926018, 4.8186642085842974, 4.403044038403941, 4.04214718582512]
y_loss = [l/100 for l in y_loss]

x = np.arange(x_left, x_right, 0.01)
ppl.plot(ax, x_values, y_accuracy, label='Accuracy', linewidth=1.0)
ppl.plot(ax, x_values, y_loss, label='Cross-Entropy Loss * 0.01', linewidth=1.0)
ppl.plot(ax, x, 0*x, label="", linewidth=0.5, color='black')
ppl.plot(ax, x, 0*x+1, label="", linewidth=0.5, color='grey')

ppl.legend(ax)

fig.savefig('training_accuracy_and_loss.png')
