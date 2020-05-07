---
layout: post
author: Michael Heinzer
title:  "On Variational Autoencoders"
description: An introduction to loss functions for regression and classification, with some mathematical motivation
date:   2020-05-04 11:00:00 +0530
categories: DeepLearning AutoEncoders MultivariateNormal Bayes GenerativeModeling
comments: no
published: False
---
Why look at it.

## Basic concepts and notation

Before we start, let us quickly repeat some basic concepts and their notation. Readers familiar with the topic may skip this section. This is not meant to be an introduction to probability theory or other mathematical concepts, only a quick refresh of what we will need later on.

- **Definition**: If a term on the left hand side of the equation is defined as as the right hand side term, then $$:=$$ will be used. This is similar to setting a variable in programming. As an example we can set $$g(x)$$ to be $$x^2$$ by writing $$g(x) := x^2$$. In mathematics, when writing simply $$=$$ means that the left side implies the right (denoted by $$\Rightarrow$$) and right side the left (denoted by $$\Leftarrow$$), at the same time.

- **Random Variable**: A variable whose values depend on the outcomes of a random phenomenon, we usually denote it by $$X$$ (upper case) and an outcome by $$x$$ (lower case). An example would be a random variable X which represents a coin throw, it can take value zero for head or one for tail.

- **Probability Distribution**: A function $$p$$ associated with a random variable $$X$$, it will tell us how likely an outcome $$x \in X$$ is. In the case of a fair coin, we will have $$p_X(0) = p_X(1) = \frac{1}{2} $$. We usually omit the subscript $$p_X$$ and only write $$p$$ for simplicity.
  If we have an unnormalized probability distribution we will denote it with a hat: $$\hat{P}$$. An unnormalized probability distribution does not need to sum up to one.
  
- **Expectation**: For a random variable $$X$$ the expectation is defined as $$ E(X) := \sum_{x \in X} p(x)  x$$. A weighted average of all the outcomes of a random variable (weighted by the probability). The expectation of the coin throw example is $$E(X) = 0 \cdot p(0) + 1 \cdot p(1) = 0 \cdot \frac{1}{2} + 1 \cdot \frac{1}{2} = \frac{1}{2}$$.

- **(Strictly) Increasing transformation**: a function $$ f : \mathbb{R} \longrightarrow \mathbb{R}$$ is a (strictly) increasing transformation if for all $$x, y \in \mathbb{R}$$ with $$ x \leq y$$ ($$x < y$$) we have that $$f(x) \leq f(y)$$ ($$f(x) < f(y)$$). These transformations have the property that we can apply them without changing the result whenever we care only about the ordering of elements, for example when minimizing a function.

- **Maximum Likelihood Estimation**: For a probability distribution $$p_\theta$$ with parameter $$\theta$$ and data $$ \lbrace x_1, \dotsc, x_n \rbrace$$ we can estimate $$\theta$$ by maximizing the probability over the data: 

  $$\theta = \text{argmax}_{\theta} p_{\theta}(x) = \prod_{i=1}^n p_{\theta}(x_i)$$

  This is called maximum likelihood estimation. Often it is more convenient to maximize the log likelihood instead, and because the log is a strictly increasing transformation the result will not change. The log transformation has the additional benefit that the product becomes a sum: 

  $$\theta = \text{argmax}_{\theta} \log(p_{\theta}(x)) = \sum_{i=1}^n \log(p_{\theta}(x_i))$$

  Which is beneficial when doing numerical optimization.

- **Kullback-Leibler Divergence**:

- **Conditional Probability**: The conditional probability of an event $$A$$ given another event $$B$$ with $$P(B) > 0$$ is

  $$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A,B)}{P(B)}$$

  The last term is simply another way of writing the intersection. What we are saying here is that the probability that an event $$A$$ happens given that $$B$$ has already happened is the probability of both $$A$$ and $$B$$ happening divided by the probability of $$B$$ only. Rewriting the above equations gives us

  $$P(A \cap B) = P(A \mid B)P(B) = P(B \mid A) P(A)$$

- **Law of total probability**: For some event $$A$$ and a partition of the sample space $$  (B_i) , i \in \lbrace 1, \dotsc \rbrace$$, the law of total probability is

  $$P(A) = \sum_{i} P(A \cap B_i) = \sum_{i} P(A|B_i)P(B_i)$$

  If $$C$$ is independent of any $$B_i$$ than we can condition it all on $$C$$ and rewrite the formula as

  $$P(A|C) =  \sum_{i} P(A \mid C \cap B_i) P(B_i) = \sum_{i} P(A| C, B_i)P(B_i)$$

  Note here that the $$B_i$$ need only be countable, but not finite.

- **Bayes Theorem**: For two events $$A$$ and $$B$$ where $$P(B) > 0$$ we have that

  $$ P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}

- 

# Preliminary Concepts

Let us first have a look at some concepts which will show up when exploring Variational Autoencoders

## Autoencoders

Autoencoders are in broad terms a type of unsupervised machine learning. They encode some input into a lower dimensional space, and then decode the compressed data to restore the input. The whole process is done in a way to  minimize information loss. More formally, suppose we have a the following:

- An input space $$\mathbb{R}^n$$ and a latent space $$\mathbb{R}^m$$. Where generally $$m$$ is much smaller than $$n$$, i.e. $$m << n$$.

- Two maps $$g_{\theta_1}: \mathbb{R}^n \longrightarrow  \mathbb{R}^m$$ and $$f_{\theta_2}:  \mathbb{R}^m \longrightarrow  \mathbb{R}^n$$ for some $$\theta_1 \in \Theta_1$$ and $$\theta_2 \in \Theta_2$$. Here we assume that the functions $$g_{\theta_1}$$ and $$f_{\theta_2}$$ are parametrized by $$\theta_1$$ and $$\theta_2$$, they could for example represent the weights of a neural network.

- An objective: want to minimize the difference between a $$x \in \mathbb{R}^n$$ and $$f_{\theta_2}(g_{\theta_1}(x)) \in \mathbb{R}^n $$:

  $$ \text{argmin}_{\theta_1 \in \Theta_1, \theta_2 \in \Theta_2} \mid \mid x - f_{\theta_2}(g_{\theta_1}(x)) \mid \mid^2 $$

   Here we call $$g$$ the encoder and $$f$$ the decoder.

While not necessary, we will assume that $$f$$ and $$g$$ are neural networks and can be trained with gradient descent.

### Applications

There are many applications of auto-encoders, here are a few places where they have been applied successfully:

- Dimensionality Reduction: Learn to represent complex data in lower dimensions, it has been shown to produce better representations than more standard techniques such as PCA. The lower dimensional data can then be more easily used for other tasks such as classification.
- Anomaly Detection: Models learn to store the most common features and will fail to reconstruct anomalous input properly.
- Image Processing: Images can be compressed efficiently into a smaller space $$m << n $$. The same compression technique can be used to denoise images by only reconstructing the important features.
- Machine Translation: Text can be encoded in a lower dimensional space in one language and decoded into another, effectively translating the input.

## Normal Distribution

## Multivariate Normal Distribution

# Variational Autoencoders

At its heart, this post is about generative modeling, or generating data from a specific distribution. This distribution could be quite complex, such as the distribution of all cats, airplanes or faces. Of course we are not limited to images but could model anything that can be expressed as a tensor, however the more complicated the distribution the more difficult it will be to generate examples. For this kind of modeling we do not need any labels as in supervised learning, although they can help in some cases. The most frequently used example for generative modeling right now would be Generative Adversarial Networks, while the approach is somewhat related, there are numerous differences. Most importantly the adversarial learning part is absent from variational autoencoders. The task we are going to learn is how to generate images from the MNIST data set. These are the first ten samples from the MNIST training set

![Plot of the first ten images in MNIST](/assets/images/on_variational_autoencoders/mnist_data.png)

MNIST images consist of pixels with values in $$[0,1]$$ in a $$28 \times 28$$ grid, i.e. they are objects in the space $$[0,1]^{28 \times 28}$$. So if we want to generate an example from that distribution we would have to pick $$784$$ values between zero and one and arrange them meaningfully in a grid of size $$28$$ by $$28$$. This is far from a trivial task, even with a lot of computational power. Computational tools such as NumPy and PyTorch only allow us to sample from one dimension, usually from a well known distribution such as the standard normal or uniform. So is there a way to produce more complex distributions from simple ones? This is what variational autoencoders do, and as the name indicates it involves an autoencoder like structure.

More formally, suppose we have samples $$x \sim X$$ , where $$X$$ is a random variable and we would like to learn its distribution $$P(X)$$. In our case $$X$$ would be a random variable generating MNIST images, so $$X \subset [0,1]^{28 \times 28}$$ and samples $$x \sim X$$ would be images as the ones above. Suppose we can easily generate samples $$z \sim Z$$ from a random variable $$Z \subset \mathbb{R}^m$$. Then we would like to find a transformation $$f_{\theta}$$ which takes a sample $$z \sim Z$$ and transforms it into an image $$x \sim X$$, or formally: find $$f_{\theta} : \mathbb{R}^m \longrightarrow [0,1]^{28 \times 28}$$ such that $$f_{\theta}(z) \sim X$$ for $$z \sim Z$$.

Finding a suitable $$f_{\theta}$$ is approximating a function, something we know how to do well with lots of data, neural networks and gradient descent, if certain conditions are met. However meeting these conditions is not that obvious, and will be what occupies us for the rest of this post.

## The main issue

When we train a supervised model, then the goal is usually pretty simple: we want to maximize its accuracy. We do this by 'punishing' inaccurate predictions and 'rewarding' accurate ones. But how do we produce a signal in an unsupervised setting? We do not have labels, but we have samples from a training set, lets call it $$X_T$$. We want the function to maximize the probability of each sample $$x \in X_T$$. Formally we maximize

$$P(X=x) = \int P(X=x \mid \theta, z) P(Z=z) dz = \int P(X=x \mid f_{\theta}(z)) P(Z=z) dz \qquad \forall x \in X_T$$

Here we are using the law of total probability in its continuous form 

## Creating an objective to optimize for

# Conditional Variational Autoencoders

# Conclusion

What we saw.

## References

A list of resources used to write this post, also useful for further reading:

- [Deep Learning](https://www.deeplearningbook.org/) Book by Goodfellow, Bengio and Courville
  - [Chapter 3](https://www.deeplearningbook.org/contents/prob.html) for Information Theory, softmax and softplus properties
  - [Chapter 5](https://www.deeplearningbook.org/contents/ml.html) for KL-Divergence, Maximum Likelihood Estimation
  - [Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) for Cross-Entropy and sigmoid/softmax discussion
- [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) Wikipedia

## Comments

I would be happy to hear about any mistakes or inconsistencies in my post. Other suggestions are of course also welcome. You can either write a comment below or find my email address on the [about page](https://heinzermch.github.io/about/).