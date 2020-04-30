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


This post is as applied as it gets for this blog. We will see how to manipulate multi-dimensional arrays or tensors as clean and efficient as possible. Being able to do so is an essential tool for any machine learning practitioner these days, much of what is done in python nowadays would not be possible without libraries such as NumPy, PyTorch and TensorFlow which handle heavy workloads in the background. This is especially true if you are working in computer vision. Images are represented as multi-dimensional arrays, and we frequently need to pre- and post-process them in an efficient manner in the ML-pipeline. In what follows, we will see the tools which are necessary for these tasks.

# NumPy Basics

Before we can get into the details of vectorization and broadcasting we need to understand the basics of NumPy, especially its `np.ndarray` class. The power of NumPy lies in its ability to pass on python instructions to much more efficient C implementations, the same applies to mapping data structures to memory. Hence when using NumPy data structures we should only manipulate them with NumPy methods whenever possible.

For all of the following code snippets we will assume that numpy as imported as `np`.

```python
import numpy as np
```



## The np.ndarray data structure

At its core the `np.ndarray` is simply an array, similar to a list in python. However while in python you can have objects of different types in a list, the `ndarray` is homogeneous. It allows only objects of the same type to be present. Most frequently we will encounter the following types:

- `np.int64`: Also called long in other programming languages, a 64-bit signed integer. The default type for integer typed data.
- `np.int32`: Also called int in other programming languages, a 32-bit signed integer.
- `np.float64`: Double precision float, the default floating point type.
- `np.float32`: Single precision float.
- Many others are supported: `float16` to `float128`, or `int8` to `int64`, `uint8` to `uint64` for unsigned integer, object, string and bool.

NumPy takes great care to allocate the required memory for its arrays as efficiently as possible, this is something we should keep in when manipulating them. When we transform python lists to a `ndarray` the data type and shape will be inferred automatically. In general a `ndarray` has the following attributes:

- `min`: minimum value in the array
- `max`: maximum value in the array
- `shape`: tuple containing the size of each dimension
- `dtype`: type of the objects in the array, see above for details
- `size`: number of elements in the array

Here is an example of the debug view of an array of shape `(3,5)` containing elements numbered from `0` to `14`:

![Debug view of a numpy array](/assets/images/on_vectorization_and_broadcasting/debug_view_array.png)

## Generating Data

There are many ways to create a `ndarray` some of the most frequent ones are list here:

### Python or PyTorch to ndarray

We can ask NumPy to transform a list, or a list of lists, to an `ndarray` using the `np.array()` method.

llocated. Hence if you change the NumPy object, the PyTorch object will be changed as well.

```python
python_array = [[i*5+j for j in range(5)] for i in range(3)]
[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
np.array(python_array)
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
```

If you are using PyTorch, then there is an efficient way to transform tensors to `ndarray` objects using `tensor.numpy()`. This will not copy the data but instead give you direct access to the same memory space where the tensor in PyTorch is allocated. Hence if you change the NumPy object, the PyTorch object will be changed as well.

```python
import torch
torch_array = torch.arange(15).reshape(3,5)
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]])
torch_array.numpy()
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
```

### Deterministically filled arrays

We can create an array filled by ones or zeros using `np.ones(shape)` and `np.zeros(shape)`. There is also the option to create an array filled with ones or zeros in the shape of another array, using `np.ones_like(arr)`.

Can use `np.arange(size).reshape(shape)` to count from 0 to size-1 and transform it into the desired size.

```python
size, shape = 15, (3, 5)
np.arange(size).reshape(shape)
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
ones = np.ones(shape)
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
np.zeros_like(ones)
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]
```

### Randomly filled arrays

Arrays filled with random numbers are also possible, we can get random integers by calling `np.random.randint(size, size=size).reshape(shape)`, which will draw integers from o to size-1.

Or numbers drawn from a random normal distribution by using `np.random.randn(*shape)`. Which is unfortunately a small inconsistency compared to the interface of other methods which take a shape parameters.

```python
size, shape = 15, (3, 5)
np.random.randint(size, size=size).reshape(shape)
array([[10, 11,  6,  3,  5],
       [13,  9,  1,  0,  7],
       [ 6,  8, 14, 14,  0]])
np.random.randn(*shape)
array([[-1.44118486,  0.92628989,  0.56191427, -0.78502917, -0.44458898],
       [ 0.07069683,  0.69491737,  1.43222532,  1.31024956, -1.44594812],
       [-0.43396544,  1.07042421, -1.33461508,  0.24587141, -0.40891315]])
```

## Indexing and views

To understand the manipulation of NumPy arrays, we will first need look at how to access parts of the array. Suppose we have an array which is a matrix, i.e. it has two dimensions `(rows, columns)`. Then we can access elements either by using the standard python notation `array[row][col]` or use  `array[row, column]`. The latter version is more concise and what we will use from now on.

For the rest of this section we will assume that we are manipulating this array

```python
size, shape = 15, (3, 5)
array = np.arange(size).reshape(shape)
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
```

### Elements

Accessing and updating elements is simple, we can use the above introduced notation. Although the accesses element is still a NumPy type, it is a copy of the original one. Changing it will not update the array, for that we need to set it the same we we accessed it.

```python
elem = array[1, 2]
7
elem = 14
14
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
array[1, 2] = 15
array([[ 0,  1,  2,  3,  4],
       [ 5,  6, 15,  8,  9],
       [10, 11, 12, 13, 14]])
```

### Slicing or Views

 If we continue with the same example above, we can extract the second row with the slicing operator `array[1,:]`, where `:` means we want to select all the elements in that dimension. This will give us a "view" of the second row, which still points to the same memory space as the original array. Consequently if we manipulate it the original array will be updated to. Selecting a range of elements in an array works the same as in python lists, the last index will not be selected. If you want to make a copy of an array, then you need to specify this explicitly using `array.copy()`.

```python
slice = array[1, :]
array([ 5,  6, 15,  8,  9])
slice[2] = 7
array([ 5,  6, 7,  8,  9])
array
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
slice[1:3]
array([6, 7])
```

### Fancy Indexing

If you would like to select non-contiguous parts of the array, this is also possible with fancy indexing. For example selecting the first and the last row can be done using `array[[0, -1]]`, this will then return a copy of the selected rows. Hence updating it will not change the original array.

```python
fancy_slice = array[[0,-1]]
array([[ 0,  1,  2,  3,  4],
       [10, 11, 12, 13, 14]])
fancy_slice[1,1] = 22
array([[ 0,  1,  2,  3,  4],
       [10, 22, 12, 13, 14]])
array
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
```

Now we have all the tools to talk about our first main goal of this post

# Vectorization

In short, **vectorization is the process of replacing explicit loops with array expressions**. In general, vectorization  will speed up the calculation by one or two orders of magnitudes. A classic example is the calculation of an average of a large matrix. We will do this for the training data set of MNIST, which consists of 50'000 images with a resolution of 28x28. Or in the flattened matrix version it has a shape of `(50000, 784)`.

```python
def classic_average(matrix: np.ndarray) -> float:
    height, width = matrix.shape
    s = 0.0
    for i in range(height):
        for j in range(width):
            s += matrix[i, j]
    return s / (height*width)

x_train = get_mnist_training_images()

classic_average = np.average(x_train)
numpy_average = np.average(x_train)
classic_average, numpy_average
0.13044972207828445 0.13044983
```

Even visually alone NumPy has the clear advantage of being very clear and concise, the larger advantage is however in the speed:

```python
from timeit import timeit
setup = 'from __main__ import classic_average, x_train; import numpy as np'
num = 10
t1 = timeit('classic_average(x_train)', setup=setup, number=num)
t2 = timeit('np.average(x_train)', setup=setup, number=num)
t1/num, t2/num, t1/t2
12.27550071660662 0.012771252996753902 961.1821737245917
```

When we run this simple example ten times, then on average the NumPy version is 961 times faster than the pure python implementation. The speedup is big because we are handling a large data set, but it will be noticeable for smaller tasks too. Maybe you think this task was too far fetched, and its rare that we want to calculate the average of that many images, then consider the next task:

Suppose we have a network which does semantic segmentation, that means it tries to assign each pixel to a class. if the input is an image of resolution `(1280, 720)` and we want to distinguish between 20 classes, then the output will be of shape `(1280, 720, 20)`. The last dimension is then often the probability that class `k` is the true class, and we would like to find the maximum along that dimension to make a prediction. This process is called taking the arg max. We can again do this the naive way or using vectorized calculations:

```python
def tensor_arg_max(tensor: np.ndarray) -> np.ndarray:
    width, height, classes = tensor.shape
    arg_max = np.zeros((width, height), dtype=np.int32)
    for i in range(width):
        for j in range(height):
            max_index, max_value = 0, 0
            for k in range(classes):
                if tensor[i, j, k] > max_value:
                    max_index, max_value = k, tensor[i, j, k]
            arg_max[i, j] = max_index
    return arg_max

shape = (1280, 720, 20)
random_network_output = np.random.random(size=np.prod(shape)).reshape(shape)
python_arg_max = tensor_arg_max(random_network_output)
numpy_arg_max = np.argmax(random_network_output, axis=2)

np.sum(np.abs(python_arg_max - numpy_arg_max))
0
```

This is a common task, and when you look at the number of elements involved `1280 * 720 * 20` which is `18'432'000` while `50000 * 784` is `39'200'000`, you see that this are half as many elements as before! The speedup is comparable and will again be around three orders of magnitude.



Other tasks

- Exponential and logarithmic function applied element wise `np.exp(array)` and `np.log(array)`
- 

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