from fastai import datasets
import gzip
import pickle
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
path = datasets.download_data(MNIST_URL, ext='.gz')
with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

mpl.rcParams['image.cmap'] = 'gray'
img = x_train[0]

for i in range(9):
    plt.subplot(330 + 1 + i)
    imgdata = x_train[i]
    imgdata = imgdata.reshape((28,28))
    plt.imshow(imgdata, cmap=plt.get_cmap('gray'))
# save the figure
plt.savefig("mnist_data.png")
