---
layout: post
author: Michael Heinzer
title:  "Vectorization and Broadcasting with NumPy"
description: How to use NumPy clearly and efficiently
date:   2020-04-25 00:00:00 +0530
categories: Numpy Vectorization Broadcasting ComputerVision Matrix Tensor Array
comments: no
published: false
---


This post is as applied as it gets for this blog. We will see how to manipulate multi-dimensional arrays or tensors as clean and efficient as possible. Being able to do so is an essential tool for any machine learning practitioner these days, much of what is done in python nowadays would not be possible without libraries such as NumPy, PyTorch and Tensorflow which handle heavy workloads in the background. This is especially true if you are working in computer vision. Images are represented as mutli-dimensional arrays, and we frequently need to pre- and post-process them in an efficient manner in the ML-pipeline. In what follows, we will see the tools which are necessary for these tasks.

## Basic concepts and notation

Readers familiar with the basic concepts and notation for matrices and tensors may skip this section.

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

- **Transposed Matrix**: Transposing a matrix is an operation, suppose we have the same matrix $$A \in \mathbb{R}^{n \times m}$$ as above, then $$A^T \in \mathbb{R}^{m \times n}$$, the transposed matrix of $$A$$ is defined as

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

# Some NumPy Basics

Before we can get into the details of vectorization and broadcasting we need to understand the basics of NumPy, especially its `np.ndarray` class. The power of NumPy lies in its ability to pass on python instructions to much more efficient C implementations, the same applies to mapping data structures to memory. Hence when using NumPy data structures we should only manipulate them with NumPy methods whenever possible.

## The np.ndarray data structure

At its core the `np.ndarray` is simply an array, similar to a list in python. However while in python you can have objects of different types in a list, the `ndarray` is homogeneous. It allows only objects of the same type to be present. Most frequently we will encounter the following types:

- `np.int64`: Also called long in other programming languages, a 64-bit signed integer. The default type for integer typed data.
- `np.int32`: Also called int in other programming languages, a 32-bit signed integer.
- `np.float64`: Double precision float, the default floating point type.
- `np.float32`: Single precision float.
- Many others are supported: `float16` to `float128`, or `int8` to `int64`, `uint8` to `uint64` for unsigned integer, object, string and bool.

NumPy takes great care to allocate the required memory for our arrays as efficiently as possible in the memory, this is something we should keep in when running operations on them. When we transform python lists to a `ndarray` the data type and shape will be inferred automatically. In general a `ndarray` has the following attributes:

- `min`: minimum value in the array
- `max`: maximum value in the array
- `shape`: tuple containing the size of each dimension
- `dtype`: type of the objects in the array, see above for details
- `size`: number of elements in the array

## Generating Data

There are many ways to create a `ndarray` some of the most frequent ones are list here:

### Python or PyTorch to ndarray

We can ask NumPy to transform a list, or a list of lists, to an `ndarray` using the `np.array()` method.

If you are using PyTorch, then there is an efficient way to transform tensors to `ndarray` objects using `tensor.numpy()`. This will not copy the data but instead give you direct access to the same memory space where the tensor in PyTorch is allocated. Hence if you change the NumPy object, the PyTorch object will be changed as well.

### Deterministically filled arrays

We can create an array filled by ones or zeros using `np.ones(shape)` and `np.zeros(shape)`.

Can use `np.arange(size).reshape(shape)` to count from 0 to size-1

### Randomly filled arrays





## Indexing and views



# Vectorization

Operations on a single array only

# Broadcasting

operations between multiple arrays

# Examples

One hot encoding labels

Confusion matrix

Gradient calculation

# Code

ate calculation. While the gradient calculation is done during the backward pass, the update is only applied when we call the `update` method with the learning rate.

```python
import numpy as np

class gradient
```

The implementation is slightly different in the sense that we have to deal with multiple examples at once, this is where the sum term in the backward pass com

# Conclusion

Some special. 

## References

A list of resources used to write this post, also useful for further reading:

- [fast.ai course - Part 2: Deep Learning from the Foundations](https://course.fast.ai/part2) for a great introduction in general

  - [Lesson 1 code](https://github.com/fastai/course-v3/blob/master/nbs/dl2/01_matmul.ipynb) for matrix multiplication
  - [Lesson 2 code](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02_fully_connected.ipynb) for forward and backward passes
  - [Lesson 2b code](https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb) for initialization

- [Deep Learning](https://www.deeplearningbook.org/) book by Goodfellow, Bengio and Courville

  - [Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) for fully connected layers, ReLU, back-propagation, MLP training

- [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) Wikipedia

- [Matrix transposition](https://en.wikipedia.org/wiki/Transpose) Wikipedia

- [Dot product](https://en.wikipedia.org/wiki/Dot_product) Wikipedia

- [Linear function](https://en.wikipedia.org/wiki/Linear_function) Wikipedia

- [Fully connected layer](https://en.wikipedia.org/wiki/Convolutional_neural_network#Fully_connected_layer) Wikipedia

- andford class introducing neural networks

  

## Comments

I would be happy to hear about any mistakes or inconsistencies in my post. Other suggestions are of course also welcome. You can either write a comment below or find my email address on the [about page](https://heinzermch.github.io/about/).