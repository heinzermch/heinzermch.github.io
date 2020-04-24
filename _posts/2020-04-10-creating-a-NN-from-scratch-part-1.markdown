---
layout: post
author: Michael Heinzer
title:  "Creating a Neural Network from scratch - Part I"
description: Explaining and coding the basic concepts of deep learning
date:   2020-04-17 11:00:00 +0530
categories: DeepLearning ComputerVision GradientDescent BatchNorm NeuralNetwork Initialization MatrixMultiplication LinearLayer ReLU
comments: no
published: false
---


The goal of this post is to create a the basic building blocks of a network from scratch. That means without using not using any PyTorch or Tensorflow library functionalities. In the end the code should look as simple as when using one of these libraries. This work is  based on the fast.ai course called deep learning from the foundations which gives an introduction similar to this over multiple courses. It is highly recommended to watch the entire course.

Even though the goal is to write a library, whenever a new concept is introduced we will also explore the accompanying mathematical part.

## Basic concepts and notation

Readers familiar with the basic concepts and notation for neural networks may skip this section.

- **Scalar**: A scalar $$a \in \mathbb{R}$$ is a single number. Usually denoted in lower case.

- **Vector**: A vector $$\mathbf{v} \in \mathbb{R}^n$$ is a collection of $$n$$ scalars, where $$n>0$$ is an integer.

  $$\mathbf{v}=\begin{pmatrix}
   v_{1}  \\
   v_{2}  \\
  \vdots   \\
   v_{n}  \\
  \end{pmatrix}$$

  $$v_i$$ is the $$i$$-th scalar value from the vector. A vector it is denoted in lowercase and bold.

- **Matrix**: A matrix $$A  \in \mathbb{R}^{n \times m}$$ for some integers $$n, m > 0$$ is a collection of $$nm$$ scalars or $$m$$ $$n$$-dimensional vectors arranged as follows:

  $$A=\begin{pmatrix}
   a_{11} & a_{12} & \cdots & a_{1m} \\
   a_{21} & a_{22} & \cdots & a_{2m} \\
  \vdots & \vdots & \ddots & \vdots \\
   a_{n1} & a_{n2} & \cdots & a_{nm} \\
  \end{pmatrix}$$

  $$a_{ij}$$ is a scalar value picked from the matrix at position $$(i, j)$$. By convention the first value in $$n \times m$$ is the number of rows and the second the number of columns. A matrix is usually denoted in uppercase.

- **Transposed Matrix**: Transposing a matrix a basic operation, suppose we have the same matrix $$A \in \mathbb{R}^{n \times m}$$ as above, then $$A^T \in \mathbb{R}^{m \times n}$$ is defined as

  $$ a^T_{ij} = a_{ji} \quad \forall i \in \lbrace 1, \dotsc, m \rbrace, j \in \lbrace 1, \dotsc, n \rbrace$$

  or more visually, we are "mirroring" the matrix along the diagonal:

  $$A^T = \begin{pmatrix}
   a^T_{11} & a^T_{12} & \cdots & a^T_{1n} \\
   a^T_{21} & a^T_{22} & \cdots & a^T_{2n} \\
  \vdots & \vdots & \ddots & \vdots \\
   a^T_{m1} & a^T_{m2} & \cdots & a^T_{mn} \\
  \end{pmatrix} = \begin{pmatrix}
   a_{11} & a_{12} & \cdots & a_{1n} \\
   a_{21} & a_{22} & \cdots & a_{2n} \\
  \vdots & \vdots & \ddots & \vdots \\
   a_{m1} & a_{m2} & \cdots & a_{mn} \\
  \end{pmatrix}$$

  Transposing a matrix has some interesting properties, such as $$A = (A^T)^T$$, but we will not use those here.

- **Row**: When we want to access a row in a matrix $$A  \in \mathbb{R}^{n \times m}$$ we denote the $$i$$-th row as $$\mathbf{a_{i.}}$$

  $$ \mathbf{a_{i.}} = \begin{pmatrix}
   a_{i1} & a_{i2} & \cdots & a_{im} \\
  \end{pmatrix} \qquad i \in \lbrace 1, \dotsc, n \rbrace.$$

  As a row is a vector, we will denote it in bold.

- **Column**: When we want to access a column in a matrix $$A  \in \mathbb{R}^{n \times m}$$ we denote the $$j$$-th row as $$\mathbf{a_{.j}}$$

  $$\mathbf{a_{.j}}=\begin{pmatrix}
   a_{1j}  \\
   a_{2j}  \\
  \vdots   \\
   a_{nj}  \\
  \end{pmatrix} \qquad j \in \lbrace 1, \dotsc, m \rbrace.$$

  As a column is a vector, we will denote it in bold.

- **Dot product**: The dot product between two vectors $$ \mathbf{a}, \mathbf{b} \in \mathbb{R}^n$$ is defined as

  $$\mathbf{a} \cdot \mathbf{b} := \sum_{i=1}^n a_i b_i $$

- **Linear function**: A function $$f : \mathbb{R} \longrightarrow \mathbb{R}$$ is linear if it is of the form

  $$f(x) = ax+b \qquad \forall a,b \in \mathbb{R}$$

  This is for the most simple case where we have a function from a real numbers to the real numbers. But this also applies for more complicated objects such as vectors or matrices.

  "Image of linear function"

- **Mean**: Let $$ n \geq 1$$ and $$x_1, \dotsc\, x_n \in \mathbb{R}$$, the mean of $$x_1, \dotsc\, x_n$$ is

  $$\overline{x} = \frac{1}{n}\sum_{i=1}^n x_i$$

- **Variance**: Let $$ n \geq 1$$, $$x_1, \dotsc\, x_n \in \mathbb{R}$$ and $$\overline{x}$$ the mean. Then the variance of $$x_1, \dotsc\, x_n$$ is defined to be 

  $$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i -\overline{x})^2 $$

- **Normal Distribution**: The normal or Gaussian distribution $$N(\mu, \sigma^2)$$ for $$\mu, \sigma \in \mathbb{R}$$ is defined by the density 

  $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\big(\frac{x-\mu}{\sigma}\big)^2}$$

  we call it the standard normal distribution if $$\mu =0$$ and $$\sigma^2=1$$. Some of the important properties are best seen visually:

  ![Plot of three normal densities](/assets/images/nn_from_scratch/normal_densities.png)
  The density is always centered at the mean $$\mu$$ and symmetric around it. The larger $$\sigma^2$$ the more spread out the density will become. There is $$67\%$$ of the area under the density in between $$-\sigma^2$$ and $$\sigma^2$$. However the property we are going to use is not visible on the graph, if we have two random variables $$X_1$$ and $$X_2$$ which follow normal distributions: $$X_1 \sim N(\mu_1, \sigma_1^2)$$ and $$X_2 \sim N(\mu_2, \sigma_2)$$. Then their sum is normally distributed again, i.e.:

  $$X_1 + X_2 = Y \sim N(\mu_1+\mu_2, \sigma_1^2 + \sigma_2^2)$$

  This is not an obvious and not a common property among random variables.

- **Sigmoid**: Plus derivative of sigmoid

- **Softmax**: $$s:  \mathbb{R}^n \longrightarrow \mathbb{R}^n$$, sometimes we omit the dependence on $$\mathbf{y}$$ for convenience and simply write $$s_i$$.

- **Derivatives**: We will deal with derivatives in multiple parts of this post. If we have a function $$f: \mathbb{R} \longrightarrow \mathbb{R}$$, then we will denote its derivative with respect to the input $$x$$ as

  $$ \frac{\partial f(x)}{\partial x} := \lim_{h \longrightarrow 0} \frac{f(x+h)-f(x)}{h}$$

- **Directional Derivative**: If $$g$$ is a function which takes a vector as input, i.e. $$g: \mathbb{R}^n \longrightarrow \mathbb{R}$$ then we will call its derivative with respect to the input vector $$\mathbf{y}$$ as

  $$ \frac{\partial g(\mathbf{y})}{\partial \mathbf{y}} =  \begin{pmatrix}
 \frac{\partial g(\mathbf{y})}{\partial y_1}\\
   \frac{\partial g(\mathbf{y})}{\partial y_2}\\
\vdots   \\
    \frac{\partial g(\mathbf{y})}{\partial y_n}\\
  \end{pmatrix} $$
  
  This is a slight abuse of more standard mathematical notation such as $$\\nabla g$$ or $$Dg$$.
  
- **Jacobian**: 

- **Gradient**: derivative evaluated at point $$x$$

## Data

We will use the most basic data for this task: MNIST. It consists of 28x28 images which have to be identified as numbers from 0 to 9.

![Plot of the first ten images in MNIST](/assets/images/nn_from_scratch/mnist_data.png)

The main work is done in the fast.ai library, but this is of no concern to us as we want to focus on the neural network part

```python
from fastai import datasets
import gzip
import pickle
import numpy as np
import typing
MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

def load_data() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    y_train, y_valid = index_to_one_hot(y_train), index_to_one_hot(y_valid)
    return x_train, y_train, x_valid, y_valid
```

Where the labels are loaded as a class label, for example class four will just be a $$4$$, but we need one hot encoded vectors, which are of the form $$[0,0,0,1,0,0,0,0,0,0]$$. The following method helps us transform them:

```python
import numpy as np

def index_to_one_hot(index: np.ndarray) -> np.ndarray:
    max_elem, min_elem = np.max(index), np.min(index)
    n, classes = index.size, max_elem + 1 - min_elem
    one_hot = np.zeros((n, classes))
    one_hot[np.arange(n), index] = 1
    return one_hot
```

There is one big assumption in the method, namely that all classes that are present in the data, you can see that in the line where we calculate the number of classes. This might not be suitable for cases where one class is extremely rare. 

## Normalizing Data

So far we have just loaded the raw data from the files, however when feeding data into a neural network it needs to be normalized to have mean zero and standard deviation one. This is very important as the networks are sensitive to the input distribution, failure to do this will lead to unexpected behavior and most likely trainings which will not converge.

```python
import numpy as np

def normalize(x: np.ndarray) -> np.ndarray:
    mean, std = x.mean(), x.std()
    return (x-mean)/std
```

We will see later on why this is important and that the network as a whole needs to follow similar rules with its weights.

## Creating a data-loader

Now we have loaded and normalized the data, but we need a way to access it for training and testing.

```python
import numpy as np
import typing


class DataSet:
    def __init__(x: np.ndarray, y: np.ndarray):
        self.x, self.y = x, y
       
    def __next__() -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    
class DataLoader:
    def __init__():
        x_train, y_train, x_valid, y_valid = load_data()
        x_train, x_valid = normalize(x_train), normalize(x_valid)
        self.x_train, self.y_train = x_train, y_train
        self.x_valid, self.x_valid = x_valid, y_valid
    
    def get_training_data() -> DataSet:
        return DataSet(self.x_train, self.y_train)
    
    def get_validation_data() -> DataSet:
        return DataSet(self.x_valid, self.y_valid)
```

This is based on how PyTorch names its data loading structures, but the functionality is different. The data loader will return us training and validation sets, then we will iterate over them once for one epoch or one validation sequence. Note that for now there is no mini-batching functionality, we will return the entire set at once.

# Forward Pass

## Matrix Multiplication

At the very basis of a lot of operations in neural networks is the matrix multiplication

### Formal Definition

Let $$A \in \mathbb{R}^{n \times m}$$ and $$B \in \mathbb{R}^{m \times l}$$ be two matrices. Then $$C \in \mathbb{R}^{n \times l}, C = A \cdot B$$ is defined as

$$c_{ij} := \sum_{k=1}^m a_{ik}b_{kj} \qquad \text{for all } i \in \lbrace 1, \dotsc, n \rbrace,\: j \in \lbrace 1, \dotsc, l \rbrace.$$

In words, each element of the resulting matrix $$C$$ is the sum of the dot product of a row in $$A$$ times a column in $$B$$. Notice that the case where we multiply a matrix with a vector is just a special case of matrix multiplication with $$l=1$$. We can write the above formula as

$$c_{ij} := a_{i.} b_{.j} \qquad \text{for all } i \in \lbrace 1, \dotsc, n \rbrace,\: j \in \lbrace 1, \dotsc, l \rbrace.$$

These formula is a bit hard to grasp when seen at the first time, let us do a small example which visualizes the process of matrix multiplication.

### A small example

Suppose we want to multiply two matrices $$A \in \mathbb{R}^{4 \times 3}$$ and $$B \in \mathbb{R}^{3 \times 2}$$. The resulting matrix $$C=AB$$ will have the shape $$\mathbb{R}^{4 \times 2}$$. Note that the resulting shape is the first dimension of the first matrix $$A$$ and the second dimension is the second dimension of the second matrix $$B$$.

$$AB = \begin{pmatrix}
 1 & 1 & 1 \\
 2 & 2 & 2 \\
3 & 3 & 3 \\
 4 & 4 & 4  \\
\end{pmatrix} \begin{pmatrix}
 1 & 1  \\
 2 & 2 \\
3 & 3 \\
 \end{pmatrix} = \begin{pmatrix}
 6 & 6  \\
 12 & 12 \\
18 & 18 \\
 24 & 24 \\
\end{pmatrix} = C$$

Even after seeing the complete example, it might still be a bit mysterious how we arrived at the final numbers, after all the formula we defined above was for a single element. Hence lets have a look how we calculate the value of $$c_{31}$$, the element of $$C$$ in the third row and first column. The elements from $$A$$ and $$B$$ which we need for our calculations have been colored in blue and red, the result in green:

$$AB = \begin{pmatrix}
 1 & 1 & 1 \\
 2 & 2 & 2 \\
\color{blue}3 & \color{blue}3 & \color{blue}3 \\
 4 & 4 & 4  \\
\end{pmatrix} \begin{pmatrix}
 \color{red}1 & 1  \\
 \color{red}2 & 2 \\
\color{red}3 & 3 \\
 \end{pmatrix} = \begin{pmatrix}
 6 & 6  \\
 12 & 12 \\
\color{green}{18} & 18 \\
 24 & 24 \\
\end{pmatrix} = C$$

If we now apply the definition we get

$$\color{green}c_{31} = \sum_{k=1}^m \color{blue}a_{3k} \color{red}b_{k1} = \color{blue}3 \cdot  \color{red}1 + \color{blue}3 \cdot  \color{red}2 + \color{blue}3\cdot  \color{red}3 = 3+6+9 = \color{green}{18}$$

Let us now write up this calculations in code.

### In code

A straightforward way would be to simply translate the mathematical definition to code. Assuming that $$A$$ and $$B$$ are NumPy arrays with the proper shapes. We add an assert to check that the dimensions match.

```python
import numpy as np

def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n, m_a = A.shape
    m_b, l = B.shape
    assert m_a == m_b, f"Dimension mismatch for A an B: {m_a} != {m_b}"
    m = m_a
    C = np.zeros((n, l))
    for i in range(n):
        for j in range(l):
            for k in range(m):
                C[i,j] += A[i, k] * B[k, j]
    return C
```



However as this is an highly inefficient way of doing matrix multiplication in python, it is usually best to delegate these computationally heavy calculations to optimized libraries, such as PyTorch or NumPy.

```python
import numpy as np
    
def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A.dot(B)
```

Confusingly the matrix muliplication in NumPy is called dot. Calling `A * B`  will do element wise multiplication, something which is used much more rarely.

## Fully connected layer

A fully connected layer connects every input value with every output value. In most cases the input and output size is fixed by the data. We are looking at MNIST images with digits of size $$28 \times 28$$ as inputs, it is quite natural to denote images as matrices. Hence we can denote them by $$X \in \mathbb{R}^{24 \times 24}$$. Our output is constrained by the number of classes we want to predict. For MNIST there are digits from 0 to 9, or 10 different classes, therefore the output is a vector $$\mathbf{y} \in \mathbb{R}^{10}$$. But which operation will transform an $$24 \times 24$$ matrix to a $$10$$-dimensional vector, connecting every input value to every output value along the way?

The answer is that we need to flatten our input image to a $$784$$-dimensional vector $$\mathbf{x} \in  \mathbb{R}^{784}$$ and apply a matrix multiplication. A flattening operation takes every image row and simply concatenates them after each other. By performing this operation we lose spatial information, however as we connect the input with the output pixel-wise, this does not matter for a fully connected layer (this is not true when applying convolutions).

If we look at a single image and write the input vector as matrix, then it has the shape $$X \in \mathbb{R}^{1 \times 784}$$ and our output needs to be a matrix $$Y \in \mathbb{R}^{1 \times 10}$$. Lets denote the matrix we are looking for as $$W$$, our operation will be $$Y = XW$$. By looking at the definition of the matrix operation we can see that $$n=1$$, $$m=784$$ and $$l=10$$. We can now know the dimensions of $$W$$ to be $$m \times l = 784 \times 10$$, an object in $$\mathbb{R}^{784 \times 10}$$. We can see this layer as a linear projection from the $$784$$-dimensional (input-)space into $$10$$-dimensional (output-)space:

$$f : \mathbb{R}^{784} \longrightarrow \mathbb{R}^{10} \quad \text{ where } \: f(X) := XW$$

Actually, any kind of neural network that does classification can be seen as a projection, of course in general neural networks are not linear. However as linear functions are quite strongly limited how they can represent data, we will try to get rid of that limit next. But first let us write up these insights in code.

```python
import numpy as np

class Linear:
    def __init__(self, input_units: int, output_units: int):
        self.weights = np.random.randn(input_units, output_units)

    def __call__(self, input: np.ndarray):
        return matrix_multiplication(input, self.weights)
```

To create a layer for our MNIST case we would instantiate the class with `linear_layer = Linear(784, 10)` and call it with `linear_layer(x)`. We need to chose initial values for our weights matrix, using zeros only would be a bad choice as our output would be zero no matter the input. As a simple start we will use randomly generated values from a standard normal distribution. This means our matrix has mean zero and standard deviation of one, like our input. Why this is not a great idea will be shown later in the section on initialization.

## Multiple Layers

For starters we can try adding more layers, that means instead of one multiplication we can do multiple ones in a row. For example suppose we first want to reduce to $$50$$ dimensions and only then to $$10$$. This is what we call a hidden layer in neural networks. In our case we would create matrices $$W_1 \in \mathbb{R}^{784 \times 50}$$ and $$W_2 \in \mathbb{R}^{50 \times 10}$$ to represent 

$$f_1 : \mathbb{R}^{784} \longrightarrow \mathbb{R}^{50}  \quad \text{ where } \:  f_1(X) := XW_1$$ 

$$f_2 : \mathbb{R}^{50} \longrightarrow \mathbb{R}^{10}  \quad \text{ where } \: f_2(X) := XW_2$$

And then do a concatenation $$f_2(f_1(X))$$ to reduce the input to our desired size.

### Concatenating linear functions

So does concatenating linear functions help us get non-linear functions? Unfortunately not. To see this consider the simple case of two linear functions $$f(x) = ax +b$$ and $$g(x) = cx + d$$ where $$a,b,c,d \in \mathbb{R}$$, then if we concatenate these functions as suggested before we get

$$ g(f(x)) = g(ax + b) = c(ax+b) + d = cax + cb + d = ex+f \quad e, f \in \mathbb{R}$$.

Which is again a linear function. The same happens when we do matrix multiplication, only the calculations get a bit more tedious because a lot more terms are involved. If we look closely at the equation above we see that we have been missing something else so far: the intercept. 

Not having an intercept means that our function always has to go through the origin, i.e. $$f(0) = 0$$. But it might be that we want to approximate functions which do not have this property, the answer is adding a vector with the output dimension. For the case of $$f_1$$ above, we need $$\mathbf{b_1} \in \mathbb{R}^{50}$$ and add it to the result: $$f_1(X) = XW_1 + \mathbf{b_1}$$. Now we have a complete linear layer. Lets rewrite the class to reflect the change:

```python
import numpy as np

class Linear:
    def __init__(self, input_units: int, output_units: int):
        self.weights = np.random.randn(input_units, output_units)
        self.bias = np.random.rand(output_units)
        
    def __call__(self, input: np.ndarray):
        return matrix_multiplication(input, self.weights) + self.bias
```


### Non-linearities

The key to having more expressive networks is adding a non linear function between linear layers. In the early days of deep learning people used sigmoid functions $$\sigma(x) = \frac{\exp(x)}{1+\exp(x)}$$. However they have the unfortunate property that they will reduce the gradient when the values are not close to zero, this will lead to the so called vanishing gradient problem when training very deep networks. Nowadays it is standard to use a rectified linear unit, or ReLU:

$$g(z) = \max(0, z)$$

The advantage here is that it is very simple to calculate, and the gradient will be unchanged for values above zero, and zero for anything else. It is not differentiable at $$z=0$$, but this is not an issue in practice. The operation gets applied element-wise, that means we apply it for every scalar of the output of the matrix multiplication. The entire network with the above introduced functions $$f_1$$ and $$f_2$$ would now have the form

$$f_2(g(f_1(X))) = f_2(g(W_1X + \mathbf{b_1})) = f_2(\max(0, W_1X) + \max(0,\mathbf{b_1}))$$

Note that we do not apply the non-linearity after the very last layer, why will be explained later on. In code the ReLU layer is quite simple:

```python
import numpy as np

class ReLU():
    def __call__(self, input: np.ndarray):
    	return np.maximum(input, 0)
```

Here we can instantiate the object again with `relu = ReLU()` and call it with `relu(x)`.

## Initialization

So far our we initialized our weight matrices with values drawn from a standard normal distribution. Assume that we want to build a network that is over a hundred layers deep, which is not fairly standard these days. See for example the [ResNet152](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), which is even in the vision library of PyTorch. We can approximate this by doing a matrix multiplication for the same number of times. Let $$W_1, \dotsc, W_{152} \in \mathbb{R}^{512,512}$$ be these $$152$$ matrices. We initialize them randomly by drawing each value from a standard normal distribution $$N(0,1)$$. Furthermore we also take a random input vector $$\mathbf{x} \in \mathbb{R^{512}}$$, where each element is also drawn from a $$N(0,1)$$ distribution. This mimics the input normalization we do for our data.

```python
import numpy as np

x = np.random.randn(512)
np.mean(x)
-0.012377488357691202
np.std(x)
0.9713856958243533
for i in range(152):
    W = np.random.randn(512,512)
    x = W.dot(x)
np.mean(x)
-6.918073644685097e+203
np.std(x)
inf
```

The exact numbers you will get from running this code will vary, but the orders of magnitude will be similar. We should note here that NumPy is more generous with the numerical precision than PyTorch, if you do the same operations there, the result will be a vector full of nan (not a number). Intuitively the results is a bit strange, after all the numbers are equally likely to be positive or negative, and they are all clustered around zero ($$67\%$$ of them will be between $$[-1,1]$$). So why do they explode?

### A mathematical explanation

To get a deeper understanding of what is happening inside these multiplications, let us look at the simpler case of one multiplication in the same setting as above, i.e. $$\mathbf{y} = W_1 \mathbf{x}$$. We are especially interested in the distribution of the elements in the vector $$\mathbf{y}$$, remember we would like them to follow a standard normal distribution $$N(0,1)$$, the same as our input data. This is how one single element $$y_i$$ of $$\mathbf{y}$$ is calculated:

$$y_i = \sum^{n}_{j=1} w_{ij}x_j = w_{i1}x_1 + w_{i2}x_2 + \dotsc + w_{in}x_n $$

Each element in the weight matrix $$W$$ independently follows a standard normal distribution, $$w_{ij} \sim N(0,1)$$. If we ignore the $$x_j$$ terms for the time being, we can see the term as a sum of $$n$$ independent $$N(0,1)$$ variables.

$$y_i =  \sum^n_{j=1} X_j \qquad X_1, \dotsc, X_n \sim N(0,1), X_j \text{ independent }$$

$$y_i$$ is then also a normal variable and by a well known theorem we get that it follows also a normal distribution: $$y_i \sim N(0,n)$$. However the variance is the sum of the individual variances because we have independence. We can transform it back to a standard normal distribution by dividing it by the square root of $$n$$:

$$ \frac{y_i}{\sqrt{n}} = \sum^n_{j=1} \frac{X_j}{\sqrt{n}} \sim N(0,1) $$

Going back to our original problem, the short analysis above suggests we could solve the problem of exploding values by multiplying each weight value $$w_{ij}$$ by $$\frac{1}{\sqrt{n}}$$. Let us define the new normalized weight matrix $$ \hat{W}_1 = \frac{W_1}{\sqrt{n}}$$ where we set $$n=512$$ and get a new vector result vector $$\mathbf{\hat{y}}$$ with values

$$\hat{y_i} = \sum^{512}_{j=1} \hat{w}_{ij}x_j = \sum^{512}_{j=1} \frac{w_{ij}}{\sqrt{512}}x_j$$

Translating this into code, we only need to change one line to make our network more stable:

```python
import numpy as np

x = np.random.randn(512)
np.mean(x)
-0.012377488357691202
np.std(x)
1.032536505693108
for i in range(152):
    W = np.random.randn(512,512) / np.sqrt(512)
    x = W.dot(x)
np.mean(x)
0.009774121752351837
np.std(x)
0.9994743773409233
```

Note that while running this code snippet multiple times, you will get quite diverse values for the standard deviation of $$x$$ at the end. Mine varied from $$0.3$$ to $$2.0$$, however they will remain reasonably small. There is only one flaw in this analysis, so far we have neglected the key part of our network: the non-linearity.

### Taking ReLU into account

How will the result change if we add ReLU layers after each multiplication? If we look at what ReLU does intuitively, it will move all the negative values to zero after every step. This means we will lose all the variation in the negative part of our output after each step, this will surely reduce our overall variation (or variance). This is exactly what we see in our experiment using a ReLU layer:

```python
import numpy as np

x = np.random.randn(512)
relu = ReLU()
for i in range(152):
    W = np.random.randn(512,512) / np.sqrt(512)
    x = relu(W.dot(x))
np.mean(x)
4.5802235212737294e-24
np.std(x)
6.567280682740789e-24
```

The math in this case is not as straightforward as in the previous case, but good insights can be gained from the papers by [Kaiming](https://arxiv.org/abs/1502.01852) and [Xavier](http://proceedings.mlr.press/v9/glorot10a.html) about the initializations with their name. As the intuitive approach suggests, we need a two in our normalization factor to increase the lost variation slightly. The resulting factor is $$\sqrt{\frac{2}{n}}$$ or four our case $$\sqrt{\frac{2}{512}}$$ which we can insert in our code to bring the variance back up:

```python
import numpy as np

x = np.random.randn(512)
relu = ReLU()
for i in range(152):
    W = np.random.randn(512,512) * np.sqrt(2/512)
    x = relu(W.dot(x))
np.mean(x)
0.6635736242280306
np.std(x)
1.0063207498458
```

Note that we introduced a slight bias into the mean, it will generally be around $$0.5$$, but this seems to be less of an issue. All the steps we have outlined above give us a certain guarantee that our network will produce reasonable output distributions at the start, however there is no restriction once we start training. Such restrictions can inserted into the network, the most popular one is called BatchNorm. We will discuss it once we have covered the basics.

## Loss

We now have all the tools to pass images through the network, but we still need to measure the results. The way we commonly do this are loss functions. In a previous [post](https://heinzermch.github.io/posts/on-loss-functions-part-1/) we saw why we generally use cross-entropy for classification and why mean squared error is a bad idea. However for illustration purposes we are going to implement mean squared error first and cross-entropy afterwards. Let us define a general loss function as function taking two vectors and producing a positive scalar as loss:

$$l : \mathbb{R}^n \times \mathbb{R}^n \longrightarrow [0, \infty)$$

Our input will generally be the model output $$\mathbf{y}  \in \mathbb{R}^n$$, and target values $$\mathbf{y_T}  \in \mathbb{R}^n$$. We assume that the target will be represented as one-hot encoded vectors. An example is that we have $$5$$ classes and the true label is $$3$$, this will be represented by a vector of the form $$[0,0,1,0,0]$$, where the third element is one and all others are zero. Note that one-hot encoded vectors are also discrete probability distributions. They fulfill the conditions that all elements are between zero and one and the sum of all elements should be equal to one.

### Mean Squared Error

The most basic case is taking the L2 distance, or the square between the model output and the true label. We do this element wise and sum up the individual distances to get a scalar:

$$l_{MSE}(\mathbf{y}, \mathbf{y_T}) := \frac{1}{n} \sum_{i=1}^n (y_i - y_{Ti})^2$$

Note that we did not restrict the model output $$\mathbf{y}$$ in any way. While our target will always be values which are either zero or one, our model output will not. In general this is a bad idea, one single output could completely dominate the loss by contributing a large value to the loss. However as we initialized our model to have outputs which are almost normally distributed around zero, this should not happen. And indeed this approach, although theoretically unsound, will work surprisingly well. This is a general feature of neural networks, they are surprisingly resistant to small inconsistencies such as these, which can lead to many undetected bugs in a machine learning pipeline. 

In code the entire loss becomes a single line

```python
import numpy as np

class MSE:
    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.mean(np.power(input-target, 2))
```

### Mean Squared Error with Softmax

In order to restrict the model output to the same range as our labels, i.e. make it a probability distribution, we can use a softmax layer. The softmax output for the $$i$$-th element is defined as

$$\text{softmax}(\mathbf{y})_i := \frac{\exp(y_i)}{\sum_{k=1}^n \exp(y_k)} = s_i$$

Note that we will use $$s_i$$ as the output for the loss function to make the equation more concise.

$$l_{MSES}(\mathbf{y}, \mathbf{y_T}) :=\frac{1}{n} \sum_{i=1}^n (s_i - y_{Ti})^2$$

Although this approach sounds more reasonable in practice we will run into issues because the softmax layer will shrink our gradients if the values are not close to zero before the softmax layer. We will get to that in the gradient section.

For now let us have a look at the softmax layer implementation. We don't simply do the calculation mentioned in the formula above, we are subtracting the maximum prediction for each vector first. This is done to for numerical stability and does not have an influence on the probabilities. We are taking advantage of the following softmax property:

$$ \text{softmax}(\mathbf{y})_i = \text{softmax}\bigg(\mathbf{y}-\max_k(y_k)\bigg)_i = \text{softmax} \begin{pmatrix}
 y_1  -\max_k(y_k)\\
 y_2  -\max_k(y_k)\\
\vdots   \\
 y_n  -\max_k(y_k)\\
\end{pmatrix}_i$$

This means the maximum value in the exponential function will be $$0$$, and we do not risk the large values appearing after exponentiating the network output.

```python
import numpy as np

class Softmax:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        input_max = np.max(input, axis=1)[:, None]
        input_exp = np.exp(input - input_max)
        return input_exp / np.sum(input_exp, axis=1)[:, None]
```

Moreover we are also taking advantage of a NumPy functionality called broadcasting. An detailed explanation would be out of the scope of this post, but in this case it helps us resizing the vector containing row wise maxima to a matrix.

The implementation of the MSES loss does not change significantly compared to the MSE. We simply need to create the softmax layer and use it before applying the loss function.

```python
import numpy as np

class MSES:
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        return np.mean(np.power(self.softmax(input)-target, 2))
```

### Cross-Entropy

Cross-entropy is a measure to compare two probability distributions, hence we have to restrict the model input to be a probability distribution as well. This is again done by applying a soft-max layer before the loss.

$$l_{CE}(\mathbf{y}, \mathbf{y_T}) := -\sum_{i=1}^n  y_{Ti} \log(s_i)$$

This expression will be positive as the input to the log will be restricted to the range of $$[0,1]$$ and we multiply all the elements by minus one. As our target vector is one-hot encoded, all the terms of the sum except for the one with the true label will become zero. So if the true label is $$k$$, the loss will reduce to

$$l_{CE}(\mathbf{y}, \mathbf{y_T}) = -\log(s_k)$$

The implementation is similar as with the MSES loss, however we only take the values into account where our target vector is not zero.

```python
import numpy as np

class CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray):
        self.input, self.target = input, target
        input_probabilities = self.softmax(input)
        return np.mean(-1*np.sum(np.log(input_probabilities[target > 0])))
```

We still take the mean because the loss will be summed over multiple predictions, or images, at once. Now we have collected all the ingredients to do a forward pass and evaluate it, next will put them together.

## Model

To put all the layers together and call them sequentially, we create a model class. Its task is to create a prediction `y = model(x)` form some input.

```python
import numpy as np

class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x
```

Also we would like a function to calculate the accuracy of our predictions. We simply chose the index containing the maximum value to be the predicted index (or class). Note that the raw output of the model is not probabilities, however the softmax transformation will not change the ordering of the elements hence there is no need to apply it.

```python
import numpy as np

def calculate_accuracy(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.argmax(output, axis=1) == np.argmax(target, axis=1))
```

Putting it all together is only a few lines, first we load the data and select only the first 100 elements (or images). Next we create the loss function and the model. The model contains three layers, two linear ones and a ReLU in between in the shapes we discussed above.

```python
x_train, y_train, _, _ = load_data()
first_images = x_train[:1000]
first_targets = y_train[:1000]

loss = CrossEntropy()
model = Model([Linear(784, 50), ReLU(), Linear(50, 10)])

y = model(first_images)
loss_value = loss(y, first_targets)
accuracy = calculate_accuracy(y, first_targets)
loss_value, accuracy
2592.4592221138564 0.112
```

Afterwards we run the images through the network and evaluate the predictions on the loss and accuracy function. The results are not great, because we did not train the network yet. The accuracy is at 11%, which is what we would expect when making random guesses, which have on average a 10% change of being right. Next we will look at how to update the network to improve the loss and accuracy.

# Backward Pass

The backward pass consists of calculating the gradient from the loss function, and then passing it backwards through the layers, storing the change we need to make at each of them. Why we can do this so easily and in a relatively isolated form is due to the Backprop algorithm, or the chain rule of partial derivatives. Explaining this in detail would be a topic for another large post. We will store the gradient updates for all the parameters in the network, and apply them in the next step called optimization.

Remember that our loss function was a function from a vector to  positive scalar: $$l : \mathbb{R}^n \times \mathbb{R}^n \longrightarrow [0, \infty)$$. The gradient of a loss function will not map to a scalar anymore, the output will be a vector of the same size as the input, where the values can also be negative:

$$\nabla l : \mathbb{R}^n \times \mathbb{R}^n \longrightarrow \mathbb{R}^n$$

More specifically the gradient will be taken with respect to the input vector $$\mathbf{y}$$ for a target vector $$\mathbf{y_T}$$: 

$$\nabla_{\mathbf{y}} l = \frac{\partial l(\mathbf{y}, \mathbf{y_T})}{\partial \mathbf{y}} = \begin{pmatrix}
 \frac{\partial l(\mathbf{y}, \mathbf{y_T})}{\partial y_1}\\
 \frac{\partial l(\mathbf{y}, \mathbf{y_T})}{\partial y_2}\\
\vdots   \\
 \frac{\partial l(\mathbf{y}, \mathbf{y_T})}{\partial y_n}\\
\end{pmatrix} $$

Each of the expressions inside the vector is a function which maps to a scalar, so we can use the standard partial derivative rules there. So for example if 

$$f:\mathbb{R}^2 \times \mathbb{R}^2 \longrightarrow \mathbb{R}, \: f(x,y) := 2x_1 + 3x_2^2y_1 + y_2 $$

Then the partial derivatives and the gradient with respect to $$x$$ are

$$\frac{\partial f(x,y)}{\partial x_1} = 2 \qquad \frac{\partial f(x,y)}{\partial x_2} = 6x_2y_1$$

$$\nabla_x f(x,y) = \begin{pmatrix} 2\\ 6x_2y_1 \end{pmatrix}$$

Note that the gradient can depend on the input values of both of the vectors (as in example), this means we have to store them for each layer. Unfortunately the calculus becomes a bit more complicated when matrices are involved, such as in the linear layer.

## Gradients and Back-propagation

First we calculate the gradients of various loss functions, and then we pass them as updates through the layers. We will call this the backward pass and implement the corresponding method in each. Generally it is easier to first look at the gradients or partial derivatives in only one dimension with a single example. Then we can generalize to multiple dimensions and finally to multiple examples. Where the last step is generally a simple average of the gradients per example.

### Mean Squared Error

The first loss function has a relatively straightforward update. Let us first suppose we do this for a case where we have one example and a one dimensional output $$y$$, then this reduces to

$$ \frac{\partial l_{MSE}(y,y_T)}{\partial y} = \frac{\partial (y-y_t)^2}{\partial y} = 2 (y-y_t)\frac{\partial y}{\partial y} =2 (y-y_t)$$

If the output $$\mathbf{y}$$ is a vector, then the same rule applies for every output dimension:

$$\frac{\partial l_{MSE}(\mathbf{y}, \mathbf{y_T})}{\partial \mathbf{y}}  = \begin{pmatrix}
 \frac{\partial l_{MSE}(\mathbf{y}, \mathbf{y_T})}{\partial y_1}\\
 \frac{\partial l_{MSE}(\mathbf{y}, \mathbf{y_T})}{\partial y_2}\\
\vdots   \\
 \frac{\partial l_{MSE}(\mathbf{y}, \mathbf{y_T})}{\partial y_n}\\
\end{pmatrix} =\begin{pmatrix}
 \frac{1}{n} \sum_{i=1}^n  \frac{\partial}{\partial y_1} (y_i - y_{Ti})^2\\
 \frac{1}{n} \sum_{i=1}^n  \frac{\partial}{\partial y_2}(y_i - y_{Ti})^2\\
\vdots   \\
 \frac{1}{n} \sum_{i=1}^n  \frac{\partial}{\partial y_n}(y_i - y_{Ti})^2\\
\end{pmatrix} = \begin{pmatrix}
 \frac{2}{n} (y_1  - y_{T1})\\
 \frac{2}{n} (y_2  -y_{T2})\\
\vdots   \\
 \frac{2}{n}(y_n  -y_{Tn})\\
\end{pmatrix} $$ 

All the terms except for one become zero in each line. If we have $$m$$ examples, then we take the average over them for each individual dimension

$$\frac{1}{m}\sum^m_{k=1} \frac{\partial l_{MSE}(\mathbf{y_k}, \mathbf{y_{kT}})}{\partial \mathbf{y}}  = \begin{pmatrix}
\frac{2}{nm}\sum^m_{k=1} y_{k1}  - y_{kT1}\\
 \frac{2}{nm}\sum^m_{k=1} y_{k2}  - y_{kT2}\\
\vdots   \\
 \frac{2}{nm}\sum^m_{k=1} y_{kn}  - y_{kTn}\\
\end{pmatrix} $$ 

The indexes can become quite messy, but the principle should be clear. From now on we will assume that the last step is straightforward and not explicitly mention it.

We need to adapt the forward pass to store the the input and target values. Additionally we add the gradient method which will return us the gradient for the last loss we calculated.

```python
import numpy as np

class MSE:
    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        return np.mean(np.power(input-target, 2))

    def gradient(self) -> np.ndarray:
        return 2.0 * (self.input - self.target) / np.multiply(*self.target.shape)
```

Note that the loss calculation is not strictly necessary to have a gradient. We nevertheless make them dependent by having stored the input and target values as attributes.

### Mean Squared Error with Softmax

Let us now do the same steps with the MSE with softmax. This will add an additonal step to the derivative because we have to take the softmax into account. We start again with the simple case of a one dimensional function with one dimensional input:

$$\begin{align*}
\frac{\partial l_{MSES}(y,y_T)}{\partial y} &= \frac{\partial (\sigma(y)-y_T)^2}{\partial y}  \\
&= 2(\sigma(y)-y_T) \frac{\partial \sigma(y)}{\partial y} \\ 
&= 2(\sigma(y)-y_T)\sigma(y)(1-\sigma(y)) 
\end{align*}$$

The last step is taking the derivative of the sigmoid function and is explained in the introduction. Note that as we normalized the input to a probability, the difference between the label and the output can no longer be larger than one. We see that the term is pretty similar to the the mean squared error but with an additional factor of $$\sigma(y)(1-\sigma(y))$$. This factor will shrink the gradient considerably as the following plot shows.

![Plot of sigmoid and its derivative](/assets/images/nn_from_scratch/sigmoid_derivative.png)

In the best case the gradient will be shrunk by a factor of one fourth. This will slow the learning process down considerably as we will see later on. This factor is one of the reasons why we prefer cross-entropy for classification tasks.

Now on to the multi-dimensional case where our sigmoid transforms into a softmax.

$$\frac{\partial l_{MSES}(\mathbf{y}, \mathbf{y_T})}{\partial \mathbf{y}}  = \begin{pmatrix}
 \frac{1}{n} \sum_{i=1}^n  \frac{\partial}{\partial y_1} (s_i - y_{Ti})^2\\
 \frac{1}{n} \sum_{i=1}^n  \frac{\partial}{\partial y_2}(s_i - y_{Ti})^2\\
\vdots   \\
 \frac{2}{n} \sum_{i=1}^n  \frac{\partial}{\partial y_n}(s_i - y_{Ti})^2\\
\end{pmatrix} = \begin{pmatrix}
 \frac{2}{n} \sum_{i=1}^n  (s_i - y_{Ti})\frac{\partial s_i}{\partial y_1}\\
 \frac{2}{n} \sum_{i=1}^n  (s_i - y_{Ti})\frac{\partial s_i}{\partial y_2}\\
\vdots   \\
 \frac{2}{n} \sum_{i=1}^n  (s_i - y_{Ti}) \frac{\partial s_i}{\partial y_n}\\
\end{pmatrix}  $$ 

Unfortunately this time the sum does not disappear as before, the derivative of the inner function of $$( \cdot )^2$$ does not go away because $$s_i$$ depends on $$y_k$$ for all $$k$$'s. So what is the derivative of the softmax exactly? First we have to note that $$ s : \mathbb{R}^n \longrightarrow \mathbb{R}^n$$ is a function from a vector to a vector, so the derivative would be a matrix, called the Jacobian matrix. However we only deal with $$s_i : \mathbb{R}^n \longrightarrow \mathbb{R}$$, which is essentially the $$i$$-th output of the softmax function $$s$$. As it returns a scalar, we can take the derivative w.r.t. to different input scalars, here for the case $$y_j$$

$$\begin{align*}
\frac{\partial s_i}{\partial y_j} &=  \frac{\partial}{\partial y_j} \bigg(\frac{\exp(y_i)}{\sum^n_{k=1} \exp(y_k)}\bigg) \\
&=  \frac{\big(\frac{\partial}{\partial y_j}(\exp(y_i)\big)\sum^n_{k=1} \exp(y_k)-\exp(y_i) \big(\frac{\partial}{\partial y_j}\sum^n_{k=1} \exp(y_k)\big)}{\big(\sum^n_{k=1} \exp(y_k)\big)^2}\\ 
&=\frac{ \mathbb{1}_{i=j}\exp(y_i)\sum^n_{k=1} \exp(y_k)-\exp(y_i) \exp(y_j)}{\big(\sum^n_{k=1} \exp(y_k)\big)^2}\\
&= \frac{ \mathbb{1}_{i=j}\exp(y_i)}{\sum^n_{k=1} \exp(y_k)} - \frac{ \exp(y_i) \exp(y_j)}{\big(\sum^n_{k=1} \exp(y_k)\big)\big(\sum^n_{k=1} \exp(y_k)\big)}  \\
&= \mathbb{1}_{i=j}s_i - s_i s_j = s_i(\mathbb{1}_{i=j} - s_j)
\end{align*}$$

This calculation can then be inserted in the equations from before, with the twist that we can fix $$j$$ for each row: 

$$\frac{\partial l_{MSES}(\mathbf{y}, \mathbf{y_T})}{\partial \mathbf{y}}  =  \begin{pmatrix}
 \frac{2}{n} \sum_{i=1}^n  (s_i - y_{Ti})s_i(\mathbb{1}_{i=1} - s_1) \\
 \frac{2}{n} \sum_{i=1}^n  (s_i - y_{Ti})s_i(\mathbb{1}_{i=2} - s_2)\\
\vdots   \\
 \frac{2}{n} \sum_{i=1}^n  (s_i - y_{Ti})s_i(\mathbb{1}_{i=n} - s_n)\\
\end{pmatrix} $$ 

The results is much messier for the softmax because every $$s_i$$ term depends on the entire input vector $$\mathbf{y}$$, which means the derivative does not cancel most terms. The above equations have been confirmed to be correct by [Yoshua Bengio himself](https://stats.stackexchange.com/questions/153285/derivative-of-softmax-and-squared-error).

The implementation of the gradient calculation is tricker here because of all these terms appearing in the gradient. With some careful broadcasting we can avoid writing any loops and do all the work fairly efficiently in a few lines:

```python
import numpy as np

class MSES:
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.input, self.target = input, target
        return np.mean(np.power(self.softmax(input)-target, 2))

    def gradient(self) -> np.ndarray:
        softmax = self.softmax(self.input)
        mse_gradient = 2.0 * (softmax - self.target) / np.multiply(*self.input.shape)
        softmax_horizontal, softmax_vertical = softmax[:, :, None], softmax[:, None, :]
        indicator = np.diag([1] * softmax.shape[1])[None, :, :]
        gradient = mse_gradient[:, :, None] * softmax_vertical * (indicator - softmax_horizontal)
        return np.sum(gradient, axis=1)
```

Note that the last line is doing the sum we also see in the equation above. The line before the sum is calculating all the terms in the sum. Before that we are only resizing the arrays among various dimensions to make the final calculation possible

### Cross-Entropy

For cross-entropy we begin again with the easiest case where we have a binary classification problem, so our only parameter is a probability $$p$$ and a label or true probability $$y_T$$. For that case the cross entropy loss is defined as

$$l_{CE}(p, y_T) = -(y_T \log(p) + (1-y_T)\log(1-p))$$

As the label $$y_T$$ is one hot encoded, it will be either zero or one. Which will zero out one of the two terms above. Suppose $$y_T=1$$, then the loss turns into $$-\log(p)$$ with derivative

$$ \frac{\partial l_{CE}(p, 1)}{\partial p} = \frac{\partial \log(p)}{\partial p} = -1/p$$

In our case the probability $$p$$ will be the softmax $$s_i$$. As in the the case with the mean squared error, the equations will become slightly more complicated when we take that into account. The general cross-entropy loss is defined as

$$l_{CE}(\mathbf{y}, \mathbf{y_T}) = -\sum^n_{i=1} y_{Ti}\log(s_i(\mathbf{y}))$$

As always $$\mathbf{y_T}$$ is one hot encoded so only one term will remain, let's call this the correct class $$c \in \lbrace 1, \dotsc n \rbrace$$, hence $$y_{Tc}=1$$. The loss is then reduced to $$l_{CE}(\mathbf{y}, \mathbf{y_T}) = -\log(s_c)$$. The partial derivative with respect to $$y_j$$ will be

$$\begin{align*}
\frac{\partial l_{CE}(\mathbf{y}, \mathbf{y_T})}{\partial y_j} &= 

- \frac{\partial \log(s_c), \mathbf{y_T})}{\partial y_j} \\ 
&= -\frac{1}{s_c} \frac{\partial s_c}{\partial y_j} \\ 
&= -\frac{1}{s_c} s_c(\mathbb{1}_{c=j} - s_j) \\
&= s_j - \mathbb{1}_{c=j}
\end{align*}$$

Or if we write it in the vector notation as for the other losses we have

$$\frac{\partial l_{CE}(\mathbf{y}, \mathbf{y_T})}{\partial \mathbf{y}}  =  \begin{pmatrix}
 s_1 - \mathbb{1}_{c=1} \\
 s_1 - \mathbb{1}_{c=2}\\
\vdots   \\
 s_n - \mathbb{1}_{c=n}\\
\end{pmatrix} $$ 

In other words, it will be $$s_j$$ if $$j \neq c$$ and $$ s_c - 1$$ for the true label $$c$$. That should be fairly simple to implement as our gradient is basically the output of the softmax layer minus one where the ground truth is one.

```python
import numpy as np

class CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, input: np.ndarray, target: np.ndarray):
        self.input, self.target = input, target
        input_probabilities = self.softmax(input)
        return np.mean(-1*np.sum(np.log(input_probabilities[target > 0])))

    def gradient(self) -> np.ndarray:
        gradient = self.softmax(self.input)
        gradient[self.target.astype(np.bool)] -= 1
        return gradient
```

Note that we need to create the softmax layer upon initialization and safe the input and output for the gradient calculations. 

### ReLU 

In case of the non-output layers there is a slightly different procedure to follow. They receive the gradient from the previous layer, which can be either a loss or another non-output layer, store updates for their parameters and pass it on. In the case of the ReLU layer which represents the function $$ReLU(x) = \max(0,x)$$ there are not parameters to adapt. The derivative of the ReLU layer is

$$\frac{\partial ReLU(x)}{\partial x} = \frac{\partial \max(0,x)}{\partial x} =    \begin{cases}1 &x>0 \\ 0 & x \leq 0 \end{cases} $$

Hence we set the gradient to zero where the output was zero and otherwise we multiply by one, which means simply passing the gradient on. As the function is applied element wise we do not need to consider the multi-dimensional case any differently than the one above. Hence if $$\nabla_{\mathbf{y}}$$ is the input gradient vector the output gradient vector $$\nabla_{\mathbf{x}}$$ will be

$$\nabla_{\mathbf{x}}   = \begin{cases}\nabla_{\mathbf{y}} &x>0 \\ 0 & x \leq 0 \end{cases}$$

We can translate that to code in one line:

```python
import numpy as np

class ReLU():
    def __call__(self, input: np.ndarray):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, gradient: np.ndarray):
        gradient[self.output == 0] = 0
        return gradient
```

The shape of the gradient will stay the same for this layer, just as the shape of the input remained the same in the forward pass. This will not be the case for the linear layer.

### Linear Layer

The case for the linear layer is slightly more complex for two reasons:

* The input and output size are not the same
* It contains parameters which need to be updated

Let us have a look at the first topic, suppose that we have an input vector $$\mathbf{x} \in \mathbb{R}^n$$, a weight matrix $$W \in \mathbb{R}^{m \times n}$$ and a bias vector $$\mathbf{b} \in \mathbb{R}^m$$ the output will then be $$\mathbf{y} = W\mathbf{x} + \mathbf{b} \in \mathbb{R}^m$$. Suppose that as for in our case we have $$m < n$$, the input vector is larger than the output vector. Then each element $$y_i, i \in \lbrace 1, \dotsc m \rbrace$$ in $$\mathbf{y}$$ the output vector is obtained by

$$y_i = \sum_{j=1}^n w_{ij} x_j + b_i$$

Note that $$b_i$$ is outside the sum and only added once for the entire term. If we take the derivative w.r.t. to an arbitrary input $$x_k, k \in \lbrace 1, \dotsc, n \rbrace$$, we get

$$ \frac{\partial y_i}{\partial x_k} = \frac{\partial}{\partial x_k} \sum_{j=1}^n w_{ij} x_j + b_i = \sum_{j=1}^n \frac{\partial w_{ij} x_j}{\partial x_k} +  \frac{\partial b_i}{\partial x_k}  = w_{ik} + 0 = w_{ik}$$

What this tells us is that the directional derivative with respect to every input $$x_k$$ will be non-zero in every in output $$y_i$$, where the indexes are $$k \in \lbrace 1, \dotsc, n \rbrace , i \in \lbrace 1, \dotsc m \rbrace$$. Let $$\nabla_{\mathbf{y}} \in \mathbb{R}^m$$ denote the incoming gradient, or the derivative of the loss with respect to this layers output $$\nabla_{\mathbf{y}} = \frac{\partial l}{\partial \mathbf{y}}$$. What we want to calculate is the gradient, or derivative of the input w.r.t. to the loss, i.e. $$\nabla_{\mathbf{x}} = \frac{\partial l}{\partial \mathbf{x}} \in \mathbb{R}^n$$. If we look at a single term in the vector, we can simplify $$\nabla_{x_k}$$ as follows

$$\nabla_{x_k} = \frac{\partial l}{\partial x_k}  = \sum_{j=1}^m  \frac{\partial l}{\partial y_j} \frac{\partial y_j}{\partial x_k} = \sum_{j=1}^m  \frac{\partial l}{\partial y_j} w_{jk} = \sum_{j=1}^m  \nabla_{y_j} w_{jk} = \sum_{j=1}^m  w_{kj}^T \nabla_{y_j}$$

So to propagate the gradient back from the output to the input we multiply by the transposed of weight matrix $$W^T$$. This will change change the dimension as to match the one of the input vector. Hence we get

$$ \nabla_{\mathbf{x}} = W^T \nabla_{\mathbf{y}}$$

So far we have taken care of passing the gradient backwards, but what about the gradient with respect to the the parameters? This is what we will look at in the next section called optimization. For now the implementation of the above calculations is as follows

```python
import numpy as np

class Linear(Module):
    def __init__(self, input_units: int, output_units: int):
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2/512)
        self.bias = np.random.randn(output_units)

    def __call__(self, input: np.ndarray):
        return input.dot(self.weights) + self.bias

    def backward(self, gradient: np.ndarray):
        return gradient.dot(self.weights.transpose())
```

Note that here we do not need the input or output values to calculate the gradient, the weights are all that is needed.

## Optimization

After creating the tools for forward and backwards pass of the neural network we now have all that is necessary for the optimization step. There are many techniques for optimizing a function, however many of them rely on the assumption the the function is convex, which is not the case for neural networks. Optimizing them relies on a the gradient descent method, which in turn relies on a differentiable loss function (which all of our above losses are).

### Gradient Descent

There are many explanations of gradient descent out there, here we will only do a quick review of the very basic optimization step used in this post. Suppose we have a differentiable function $$f : \mathbb{R} \longrightarrow \mathbb{R}$$ and a starting point $$x_0 \in \mathbb{R}$$, we would like to find the point $$x_s \in \mathbb{R}$$ for which $$f(x_s)$$ is minimal, i.e. $$f(x_s) \leq f(x) \: \: \forall x \in \mathbb{R}$$. One way to do this is to find a sequence $$(x_i)_{i \in \mathbb{N}} \in \mathbb{R}$$ which converges to $$x_s$$, or written differently $$x_s$$ is the limit of that sequence:

$$\lim_{i \longrightarrow \infty} x_i = x_s$$

We only have  starting point $$x_0$$, how do we find such a sequence? By doing gradient descent. The steps are iteratively and involve the gradient and a learning rate $$\gamma \in \mathbb{R}^+$$.

$$x_{n+1} = x_n - \gamma \nabla f(x_n)$$

Generally $$\gamma$$ should be small, meaning less than one. The gradient points in the direction of the strongest increase in the function value. So if we go into the other direction then we should go towards a minimum, the minimum of the loss function we defined. To visualize this process, we take a simple example $$f$$ and calculate its gradient:

$$ f(x) := (x-2)^2+1 \qquad \nabla f(x) = 2(x-2)$$

This function will guide us to the point $$x_s = 2$$, which is its only minimum with $$f(x_2) =1$$. Setting the learning rate to $$\gamma = 0.1$$ and the starting point to $$x_0 := 5$$ gives us a sequence of the following values:

![Plot of the first ten gradient descent steps on f](/assets/images/nn_from_scratch/gradient_descent.png)

The above plot shows the first ten steps of the gradient descent algorithm, the points move from right to left, towards the minimum. The red points will cluster pretty quickly around the minimum of $$x_s = 2$$. Gradient descent is guaranteed to converge to the global optima in the case of convex functions, such as in our example.

Unfortunately gradient descent does not give us any guarantee that we will find a global optima in the case of non-convex functions, we might get stuck in a local minimum. In practice there are many ways how we try to reduce the likelihood of ending up in a suboptimal solution, but for our problem here they won't be necessary.

### Linear layer

We can apply the same principle of gradient descent to the problem of a neural network, only now we won't optimize a function $$f$$ but our loss $$l$$. Most of the parts of the network we designed so far are static, except for the linear layer, which has two components which can be optimized:

- A eight matrix $$W$$
- A bias vector $$\mathbf{b}$$ 

For both we need to calculate the gradient with respect to their individual weights to update according to the gradient descent rule. Remember that we have as input the gradient vector $$\nabla_y$$ and the layer operation is defined as follows $$\mathbf{y} = W\mathbf{x} + \mathbf{b} \in \mathbb{R}^m$$. First, let us look at the simpler part of updating the bias vector. For a single bias element $$b_k$$, the derivative of the output with respect to $$b_k$$ is

$$\frac{\partial y_i}{\partial b_k} = \sum_{j=1}^n  \frac{\partial w_{ij} x_j}{\partial b_k} +  \frac{\partial b_i}{\partial b_i} =0 + \mathbb{1}_{i=k} = \mathbb{1}_{i=k}$$

For the gradient with respect to the loss function we have to consider more terms

$$\nabla_{b_{k}} = \frac{\partial l}{\partial b_k} = \sum_{i=1}^m  \frac{\partial l}{\partial y_i} \frac{\partial y_i}{\partial b_k} = \sum_{i=1}^m  \frac{\partial l}{\partial y_i} \mathbb{1}_{i=k} = \frac{\partial l}{\partial y_k}1 =\nabla_{y_{k}}$$

So for each element in the bias vector $$\mathbf{b}$$ the gradient we need is simply the corresponding element in the incoming gradient vector $$\nabla_y$$. If we denote the updated vector by $$\mathbf{b_{\text{new}}}$$, then the gradient update is

$$\mathbf{b_{\text{new}}} = \mathbf{b} - \gamma \nabla_y$$

That was not too difficult! We can do the same for the weight matrix, by looking again first at a single element $$w_{kh}$$ in the matrix. The gradient with respect to the output $$y_i$$ is then

$$ \frac{\partial y_i}{\partial w_{kh}} = \sum_{j=1}^n \frac{\partial w_{ij} x_j}{\partial w_{kh}} +  \frac{\partial b_i}{\partial w_{kh}} = \mathbb{1}_{i=k}\mathbb{1}_{j=h}x_h + 0 = \begin{cases}x_h &i=k \\ 0 & i \neq k \end{cases}$$

So for the $$k$$-th row the gradient depends on $$h$$, otherwise it will be zero. 

$$\begin{align*}
\nabla_{w_{kh}} &=  \frac{\partial l}{\partial x_{kh}} \\
&=  \sum_{i=1}^m \sum_{j=1}^n \frac{\partial l}{\partial y_i} \frac{\partial y_i}{\partial x_{kh}} \\
&= \sum_{i=1}^m \sum_{j=1}^n \frac{\partial l}{\partial y_i} \mathbb{1}_{i=k}\mathbb{1}_{j=h}x_h \\ 
&= \sum_{i=1}^m \frac{\partial l}{\partial y_i} \mathbb{1}_{i=k} \sum_{j=1}^n \mathbb{1}_{j=h}x_h \\ 
&= \sum_{i=1}^m  \frac{\partial l}{\partial y_i} \mathbb{1}_{i=k} x_h \\
&= \frac{\partial l}{\partial y_k} x_h \\
&= \nabla_{y_k} x_h

\end{align*}$$

The double sum simplifies nicely due to the two indicator function which effectively select one term each from the two sums. In simple terms the gradient for the weight $$w_{kh}$$ is the product of the $$h$$-th input element and the $$k$$-th incoming gradient element. How can we write this simple for the entire weight matrix? The answer is to see this as a matrix multiplication, but of two vectors. Remember that $$\nabla_{y} \in \mathbb{R}^m$$, $$\mathbf{x} \in \mathbb{R}^n$$ and $$W \in \mathbb{R}^{m \times n}$$. We can also see the two vectors, $$\nabla_{y}$$ and $$\mathbf{x}$$, as matrix in $$\mathbb{R}^{m \times 1}$$ and $$\mathbb{R}^{n \times 1}$$ respectively. Transposing  the gradient and multiplying the input with it gives us a matrix with the same size as the weight matrix with the right terms:

$$ \nabla_W = x \nabla_y^T \in \mathbb{R}^{m \times n} $$

Once we have the gradient the optimization step is again the generic gradient descent algorithm, for a new weight matrix $$W_{new}$$ we do the update

$$ W_{new} = W - \gamma \nabla_W$$

In the implementation we will call $$\gamma$$ learning rate. Note that we will need to store the input in the forward pass for the gradient update calculation. The gradient calculation is done during the backward pass, the update is applied when we call the `update` method with the learning rate.

```python
import numpy as np

class Linear(Module):
    def __init__(self, input_units: int, output_units: int):
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2/512)
        self.bias = np.random.randn(output_units)
        self.weights_gradient = np.zeros((input_units, output_units))
        self.bias_gradient = np.zeros(output_units)

    def __call__(self, input: np.ndarray):
        self.input = input
        return input.dot(self.weights) + self.bias

    def backward(self, gradient: np.ndarray):
        self.weights_gradient = np.sum(self.input[:, :, None] * gradient[:, None, :], axis=0)
        self.bias_gradient = np.sum(gradient, axis=0)
        return gradient.dot(self.weights.transpose())

    def update(self, learning_rate: float):
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
```

The implementation is slightly different in the sense that we have to deal with multiple examples at once, this is where the sum term in the backward pass comes from. We also do not directly transpose the gradient, but let broadcasting do the job of resizing the vectors for us, this has the same effects but we only need to do an element-wise multiplication.

Finally we have everything to do complete forward, backward and update steps. First we need to update the model class to support the final two steps:

```python
import numpy as np

class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, gradient: np.ndarray):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, learning_rate: float):
        for layer in self.layers:
            layer.update(learning_rate)
```

It will take the gradient from the loss layer as input and propagate it backwards for each term. Then we can do one step each to see if the accuracy improves:

```python
learning_rate, epochs, examples = 0.01, 30, 100
x_train, y_train, x_valid, y_valid = load_data()
multiple_images, multiple_targets = x_train[:examples], y_train[:examples]

loss = CrossEntropy()
model = Model([Linear(784, 50), ReLU(), Linear(50, 10)])

for i in range(epochs):
    y = model(multiple_images)
    loss_value, accuracy = loss(y, multiple_targets), calculate_accuracy(y, multiple_targets)
    gradient = loss.gradient()
    model.backward(gradient)
    model.update(learning_rate)
```

For readability reasons the loss and accuracy values are not printed, but have been plotted on an image. The cross-entropy loss has been reduced by a factor of $$100$$ to fit it into the accuracy plot.

![Plot of training accuracy and cross entropy loss](/assets/images/nn_from_scratch/training_accuracy_and_loss.png)

Overall we see what we would expect, the loss reduces sharply and the accuracy raises to a $$100\%$$. Attentive observers will have seen that we train and test the network on the same images, this is not something one would do to make a serious evaluation. However in this case it is useful to see if the network is capable of learning at all, which it is.

# Tests

Whenever one writes code, it is a good habit to think of ways to validate it. This has many advantages, from discovering bugs to making sure the behavior of code stays the same over time. This is something I personally wish I had learned much earlier, it could have saved me countless hours of wasted research efforts. Additionally it is in general quite hard to detect bugs in neural networks, they are so flexible flaws can hide for quite a while. They might only manifest themselves by a slightly worse performance than expected, but discovering where the issue is in the pipeline will be time consuming.

In this section we will first write individual tests for layers and losses, and then do some quick end-to-end test to see if all the code fits together.

## Unit

There are many ways to test our implementation of these basic building blocks, we could come up with examples and see if the results look as we expect or test against a reference implementation. The latter is what we will do here, specifically we will test results against PyTorch implementations of the same layer.

Because minor imprecisions we will only test if the results are "close enough", i.e. if the do not differ by more than a certain tolerance value. Here is a utility method which test if a tensor is close enough to zero for all elements:

```python
import numpy as np

def assert_tensor_near_zero(tensor: np.ndarray, tol=1e-3) -> None:
    assert np.all(np.abs(tensor) < tol), f"Not zero: {tensor}"
```

There are some more technicalities to take care of, for example to conversion from torch tensors to numpy arrays in an automated fashion, for this there are wrapper classes.

### Testing a loss

Lets have a look how we test the cross-entropy loss. Firstly we create the PyTorch version of it and create a random input and target matrix. Then we test the loss when passing the input through our and the PyTorch version, they need to match by a certain tolerance threshold.

```python
import torch
import numpy as np

def test_cross_entropy() -> None:
    batch_size, features = 3, 5
    torch_cross_entropy = torch.nn.CrossEntropyLoss()
    cross_entropy = ClassificationLossWrapper(CrossEntropy)

    input = torch.randn(batch_size, features, requires_grad=True)
    target = torch.empty(batch_size, dtype=torch.long).random_(features)
    # Forward loss
    loss = cross_entropy(input, target, classes=features)
    torch_loss = torch_cross_entropy(input, target)
    assert_tensor_near_zero(to_numpy(torch_loss) - loss)
    # Check gradient
    torch_loss.backward()
    torch_gradient = input.grad
    gradient = cross_entropy.gradient()
    assert_near_zero_tensor(to_numpy(torch_gradient)-gradient)
```

Secondly we propagate the loss backwards and test if the gradient from both layers are close. Note that in PyTorch the gradient is created in the input variable, and is not a return value as from our loss. 

### Testing a layer

Testing a layer is slightly more complicated for a couple of reasons:

- Layers have parameters which need to be the same (in this case weight and bias)
- Have three functionalities to test: forward pass, backward pass, parameter update
- Need to be given a gradient, which means we have to involve a loss function

Here we test the linear layer, forward and backward pass. To ensure the same parameters we copy the weight and bias the PyTorch layer to ours. After the forward pass we check if the results are the same. To make it a bit more readable the code is split into two parts

```python
import torch
import numpy as np

def test_linear():
    batch_size = 3
    in_features = 768
    out_features = 50
    torch_linear = nn.Linear(in_features=in_features, out_features=out_features)
    linear = LayerWrapper(Linear, input_units=in_features, output_units=out_features)
    input = torch.randn(batch_size, in_features, requires_grad=True)
    
    # Forward pass, to ensure same operation we copy weight and bias
    linear.update_attribute("weight", to_numpy(torch_linear.weight.T))
    linear.update_attribute("bias", to_numpy(torch_linear.bias))
    assert_tensor_near_zero(to_numpy(torch_linear(input)) - linear(input))
```

To test the backward pass we need to use a loss function to create a gradient. Which one we use is not important, it just have to give some non-zero gradient. In this case we will use the MSE. Note that as a intermediate step, we also test that the two calculated losses correspond, this would help us detect problems at an earlier stage. We also check that the gradients which update the parameters are the same, otherwise our layers would not learn in the same way.

```python
    # Backward pass, losses
    torch_loss = nn.MSELoss()
    loss = RegressionLossWrapper(MSE)
    # Backward pass, loss calculation and check
    target = torch.randn(batch_size, out_features)
    torch_loss_output = torch_loss(torch_linear(input), target)
    torch_loss_output.backward()
    loss_output = loss(torch.from_numpy(linear(input)), target)
    assert_tensor_near_zero(loss_output - to_numpy(torch_loss_output))
    
    # Backward pass, gradient pass
    gradient = linear.backward(torch.from_numpy(loss.gradient()))
    torch_gradient = input.grad
    assert_tensor_near_zero(to_numpy(torch_gradient) - gradient)
    # Backward pass, weight gradient
    torch_weight_gradient = torch_linear.weight.grad.T
    weight_gradient = linear.get_attribute("weight_gradient")
    assert_tensor_near_zero(to_numpy(torch_weight_gradient) - weight_gradient)
    # Backward pass, bias gradient
    torch_bias_gradient = torch_linear.bias.grad
    bias_gradient = linear.get_attribute("bias_gradient")
    assert_tensor_near_zero(to_numpy(torch_bias_gradient) - bias_gradient)
```

What is not done at this point is testing the update method, however this is only adding two attributes which we already know are correct: the parameters and their gradient.

## End to end

For a quick end-to-end test we can run three different training runs, with the same network architecture and hyper parameters but a different loss each. For all three the learning rate was constant at $$0.5$$ over the entire training duration. The performance was relatively stable when making the learning rate an order of magnitude larger or smaller. However one could always fine tune it to make specific losses work better.

![Plot of training accuracy for three different losses for 50 epochs](/assets/images/nn_from_scratch/training_accuracy_multiple_losses.png)

# Conclusion

We saw some of the major building blocks of a neural network, layers and losses. For each part there was a theoretical motivation followed by an implementation in Python. The two main phases of neural network training, the forward and backward pass were described for each component. In the end we saw tests to verify the functionality. 

## References

All the code in this post can be found in a cleaned up version in my github repository called [NeuralnetworkFromScratch](https://github.com/heinzermch/NeuralNetworkFromScratch).

A list of resources used to write this post, also useful for further reading:

- [fast.ai course - Part 2: Deep Learning from the Foundations](https://course.fast.ai/part2) for a great introduction in general

  - [Lesson 1 code](https://github.com/fastai/course-v3/blob/master/nbs/dl2/01_matmul.ipynb) for matrix multiplication
  - [Lesson 2 code](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02_fully_connected.ipynb) for forward and backward passes
  - [Lesson 2b code](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb) for initialization

- [Deep Learning](https://www.deeplearningbook.org/) Book by Goodfellow, Bengio and Courville
  
  - [Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) for fully connected layers, relu, back-propagation, MLP training
  
- [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) Wikipedia

- [Matrix transposition](https://en.wikipedia.org/wiki/Transpose) Wikipedia

- [Dot product](https://en.wikipedia.org/wiki/Dot_product) Wikipedia

- [Linear function](https://en.wikipedia.org/wiki/Linear_function) Wikipedia

- [Fully connected layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Fully_connected_layer) Wikipedia

- [Gradient](https://en.wikipedia.org/wiki/Gradient) Wikipedia

- [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) Wikipedia

- [Notes on Weight Initialization for Deep Neural Networks](https://madaan.github.io/init/) for a more detailed explanation on initialization

- [How to initialize deep neural networks?](https://pouannes.github.io/blog/initialization/) for a good summary of the two initialization papers

- [Gradient of a Matrix Matrix multiplication](https://edwardshu.com/posts/matrix-matrix-gradient) blog post on gradient calculations with multiple samples

- [Derivative of the sigmoid function](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x) for a in detail calculation

- [Derivative of the softmax function](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/) for an excellent explanation of the softmax and its derivative

- [Derivative of MSE with softmax](https://stats.stackexchange.com/questions/153285/derivative-of-softmax-and-squared-error) for the calculation of the loss

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) for a Standford class introducing neural networks

  

## Comments

I would be happy to hear about any mistakes or inconsistencies in my post. Other suggestions are of course also welcome. You can either write a comment below or find my email address on the [about page](https://heinzermch.github.io/about/).