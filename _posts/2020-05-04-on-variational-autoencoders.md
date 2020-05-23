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
  
- **Expectation**: For a random variable $$X$$ the expectation is defined as $$ E[X] := \sum_{x \in X} p(x)  x$$ for a discrete distribution, and $$E[X] = \int p(x) x dx$$ for a continuous distribution . A weighted average of all the outcomes of a random variable (weighted by the probability). The expectation of the coin throw example is $$E[X] = 0 \cdot p(0) + 1 \cdot p(1) = 0 \cdot \frac{1}{2} + 1 \cdot \frac{1}{2} = \frac{1}{2}$$.

- **(Strictly) Increasing transformation**: a function $$ f : \mathbb{R} \longrightarrow \mathbb{R}$$ is a (strictly) increasing transformation if for all $$x, y \in \mathbb{R}$$ with $$ x \leq y$$ ($$x < y$$) we have that $$f(x) \leq f(y)$$ ($$f(x) < f(y)$$). These transformations have the property that we can apply them without changing the result whenever we care only about the ordering of elements, for example when minimizing a function.

- **Maximum Likelihood Estimation**: For a probability distribution $$p_\theta$$ with parameter $$\theta$$ and data $$ \lbrace x_1, \dotsc, x_n \rbrace$$ we can estimate $$\theta$$ by maximizing the probability over the data: 

  $$\theta = \text{argmax}_{\theta} p_{\theta}(x) = \prod_{i=1}^n p_{\theta}(x_i)$$

  This is called maximum likelihood estimation. Often it is more convenient to maximize the log likelihood instead, and because the log is a strictly increasing transformation the result will not change. The log transformation has the additional benefit that the product becomes a sum: 

  $$\theta = \text{argmax}_{\theta} \log(p_{\theta}(x)) = \sum_{i=1}^n \log(p_{\theta}(x_i))$$

  Which is beneficial when doing numerical optimization.

- **Kullback-Leibler Divergence**: The Kullback-Leibler divergence can be defined for both discrete and continuous probability distributions, lets start with the first one. Let $$p$$ and $$q$$ be two discrete probability distributions, i.e. $$0 \leq p(x) \leq 1$$ for all $$x \in X$$ and $$\sum_{x \in X} p(x) = 1$$. The same applies for $$q$$. Then the KL-divergence is defined as

  $$D_{KL} (p \mid \mid q) = \sum_{x \in X} p(x) \log\bigg(\frac{p(x)}{q(x)}\bigg) = \mathbb{E}_{x \sim p(x)}\bigg[\log\bigg(\frac{p(x)}{q(x)} \bigg)\bigg]$$ 

  We can write similarly for $$p$$ and $$q$$ continuous distribution functions, i.e. $$f(x) \geq 0$$ and $$\int f(x)dx = 1$$, the KL-divergence is defined as

  $$D_{KL}(p \mid \mid q) = \int p(x) \log\bigg(\frac{p(x)}{q(x)}\bigg) dx = \mathbb{E}_{x \sim p(x)}\bigg[  \log \bigg(\frac{p(x)}{q(x)}\bigg) \bigg]$$

- **Non-Negativity of Kullback-Leibler**: Later on we will use that $$D_{KL}(p \mid \mid q) \geq 0$$ for any two probability distributions $$p$$ and $$q$$ with the same support. It is not immediately obvious why this is the case, as the logarithm is negative for values in $$(0,1)$$.

  First we need to convince ourselves that $$\log(x) \leq x-1$$ for all $$x > 0$$. How do we know this to be true? One way to see is to look at the derivative of both functions $$\frac{\partial \log(x)}{\partial x} = \frac{1}{x}$$ and $$\frac{\partial (x-1)}{\partial x} = 1$$, and then do a case distinction:

  - For $$x>1$$, we have that $$\frac{1}{x} < 1$$, so the logarithm grows slower than the linear function for positive values.
  - For $$0 < x < 1$$ we have that $$\frac{1}{x} > 1$$, so the logarithm grows faster than the linear function for negative values.
  - For $$x=1$$, we have equality between the two sides as $$log(1) = 0 \leq 1-1 = 0$$. 

  We can also simply compare the two functions visually: 

  ![Plot of the inequality log x is smaller equal x minus 1](/assets/images/on_variational_autoencoders/log_x_seq_x_minus_1.png)

  Now that we have established that fact, we can look at the original objective of proving $$D_{KL}(p \mid \mid q) \geq 0$$, which is the same as $$-D_{KL}(p \mid \mid q) \leq 0$$. Let us start from there, and show for the discrete version of the KL-divergence:

  $$\begin{align} -D_{KL}(p \mid \mid q) &=  -\sum_{x \in X} p(x) \log\bigg(\frac{p(x)}{q(x)}\bigg)\\ 
  &= \sum_{x \in X} p(x) \log\bigg(\frac{q(x)}{p(x)}\bigg) \\
  &\leq \sum_{x \in X} p(x) \bigg(\frac{q(x)}{p(x)}-1\bigg) \\
  &= \sum_{x \in X} q(x) - \sum_{x \in X} p(x) \\ 

  &= 1-1 \\ 

  &= 0
  \end{align}$$

  Hence we have shown that $$D_{KL}(p \mid \mid q) \geq 0$$ using the inequality from above where the $$\leq$$ shows up.

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

  $$ P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

- **Covariance**: The covariance between two random variables $$X$$ and $$Y$$ is defined to be:
  $$Cov(X,Y)=E((X-E(X))(Y-E(Y))$$

- **Variance**: The variance is a special case of the covariance, when $$X=Y$$, then this is called the variance of a random variable $$X$$ and defined as
  $$Var(X) = Cov(X,X) = E((X-E(X))^2)$$

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

## Transforming Functions

Transformation between rectangular and polar coordinates is a bijection, in the space $$\mathbb{R}^2$$. We go from $$(x,y)$$ in $$\mathbb{R} \times \mathbb{R}$$ to $$(r, \theta)$$ in the space $$\mathbb{R} \times (\pi/2, -\pi/2)$$. The two transformations are

$$f(x,y) = (r \cos(\theta), r \sin(\theta)) \qquad g(r, \theta) = (\sqrt{x^2+y^2}, \tan^{-1}(\frac{y}{x}))$$

The image $$B= r(A)$$ then 

$$P((X_1, X_2, \dotsc, X_n) \in A) = \int \dotsc \int_A f(x_1, \dotsc, x_n)dx_1 \dotsc dx_n = \int \dotsc \int_B f(s_1, \dotsc, s_n) \mid J \mid dy_1 \dotsc dy_n = P((Y_1, \dotsc, Y_n) \in B)$$

If the transformation is linear it becomes easier and we get for an invertible matrix $$A \in \mathbb{R}^{n \times n}$$:

$$\mathbf{y} = \begin{pmatrix}
 Y_{1}  \\
\vdots  \\
 Y_{n} 
\end{pmatrix} = A \begin{pmatrix}
 X_{1}  \\
\vdots  \\
 X_{n} 
\end{pmatrix} = A\mathbf{x}$$

$$\mid J \mid \det(A^{-1}) = (\det(A))^{-1} = \frac{1}{\det(A)}$$

That's why we need that $$A$$ is invertible, it is equivalent with $$\det(A) \neq 0$$. The pdf of $$(Y_1, \dotsc, Y_n)$$ becomes

$$g(\mathbf{y}) = g(y_1, \dotsc, y_n) = \frac{1}{\det(A)} f(A^{-1}\mathbf{y})$$

## Normal Distribution

The normal distribution with parameters $$\mu$$ and $$\sigma^2$$ has the probability density function

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \qquad -\infty < x < \infty$$

We usually write $$X \sim N(\mu, \sigma^2)$$. The standard normal distribution means $$N(0,1)$$. We can check that the normal distribution is a probability distribution by verifying 

$$\int_{\infty}^{\infty} f(x) dx = 1$$

This is an interesting proof for many reasons: we can use the function transformation theorem from above, see how to transform variables in an integral and discover a nice idea for a proof. First lets substitute $$z = \frac{x-\mu}{\sigma}$$ in the p.d.f. and define $$I$$:

$$z = \frac{x-\mu}{\sigma} \Rightarrow \frac{dz}{dx} = \frac{d(x-\mu)/\sigma}{dx} = \frac{1}{\sigma} \Leftrightarrow dx = \sigma dz  $$

We apply this now and define $$I$$

$$I:=  \frac{1}{\sqrt{2\pi}\sigma}\int_{\infty}^{\infty}e^{-\frac{1}{2}\big(\frac{x-\mu}{\sigma}\big)^2}  =  \frac{1}{\sqrt{2\pi}\sigma}\int_{\infty}^{\infty}e^{-\frac{1}{2}z^2}dz \sigma = \frac{1}{\sqrt{2\pi}}\int_{\infty}^{\infty}e^{-\frac{1}{2}z^2}dz$$

Note that this is the p.d.f. of the standard normal distribution $$N(0,1)$$. We are not going to prove directly that $$I = 1$$ but equivalently that $$I^2= 1$$. We start by

$$\begin{align} I^2 &= \Big[\frac{1}{\sqrt{2\pi}}\int_{\infty}^{\infty}e^{-\frac{1}{2}x^2}dx\Big] \Big[ \frac{1}{\sqrt{2\pi}}\int_{\infty}^{\infty}e^{-\frac{1}{2}y^2}dy\Big]  \\ 
&= \frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}e^{-\frac{1}{2}(x^2+y^2)}dxdy \\
\end{align}$$

For this we can apply the function transform theorem we have seen above with the functions

$$r = r_1(r, \theta) = \sqrt{x^2+y^2} \text{ and } \theta = r_2(x, y) = \tan^{-1}\frac{y}{x}$$

$$x = s_1(r, \theta) = r \cos(\theta) \text{ and } y = s_2(r, \theta) = r \sin(\theta)$$

$$ J = \begin{pmatrix}
 \frac{\partial s_1}{\partial y_1} & \frac{\partial s_1}{\partial y_2}  \\
 \frac{\partial s_2}{\partial y_1} & \frac{\partial y_2}{y_2} 
\end{pmatrix} =  \begin{pmatrix}
 \frac{\partial r \cos(\theta)}{\partial r} & \frac{\partial r \cos(\theta)}{\partial \theta}  \\
 \frac{\partial r\sin(\theta)}{\partial r} & \frac{\partial r \sin(\theta)}{\partial \theta} 
\end{pmatrix} = \begin{pmatrix}
 \cos(\theta) &  -r \sin(\theta)  \\
 \sin(\theta) & r \cos(\theta) 
\end{pmatrix} $$

$$\det(J) = j_{11} j_{22} - j_{12} j_{21} = \cos(\theta)(r\cos(\theta)) - (-r\sin(\theta))\sin(\theta)=r (\cos(\theta)^2 + \sin(\theta)^2) = r $$

After the transformation we arrive at

$$\begin{align} I^2 &= \frac{1}{2\pi}\int_{0}^{2\pi}\int_{-\infty}^{\infty}e^{-\frac{1}{2}\big((r\cos(\theta))^2+(-r\sin(\theta))^2\big)}rdrd\theta  \\  
&=\frac{1}{2\pi}\int_{0}^{2\pi}\int_{-\infty}^{\infty}re^{-\frac{1}{2}r^2\big(\cos(\theta)^2+\sin(\theta)^2\big)}drd\theta  \\   
&=\frac{1}{2\pi}\int_{0}^{2\pi}\int_{-\infty}^{\infty}re^{-\frac{1}{2}r^2}drd\theta  \\
&=\frac{1}{2\pi}\int_{0}^{2\pi}d\theta\int_{0}^{\infty}re^{-\frac{1}{2}r^2}dr  \\ 
&= \frac{1}{2\pi} [\theta]^{2\pi}_0 \int_{0}^{\infty} re^{-\frac{1}{2}r^2}dr \\
&= \frac{(2\pi-0)}{2\pi} \int_{0}^{\infty} re^{-\frac{1}{2}r^2}dr \\ 
&= \int_{0}^{\infty} re^{-\frac{1}{2}r^2}dr

\end{align}$$

So the problem reduces to showing that the last integral is zero, which we can do by applying another variable transform

$$ s = -\frac{r^2}{2} \quad \Rightarrow \frac{ds}{dr} = \frac{-(r^2)/2}{dr} = -r \Leftrightarrow dr = \frac{ds}{-r}$$

The limits change $$s(\infty) = -\infty$$ and $$s(0) = 0$$. Apply the transform for the final step to show equality to one:

$$\begin{align} I^2 &= \int_{0}^{\infty} re^{-\frac{1}{2}r^2}dr  \\ 
&=-\int_{-\infty}^{0} re^{s}\frac{ds}{-r}dr  \\
&= \int_{-\infty}^{0} e^{s}ds \\
&= [e^s]^0_{-\infty} \\ 
&= 1 - 0 = 1

\end{align}$$ 

Hence we now know that the normal probability density deserves its name.

## Multivariate Normal Distribution

If $$X_1, \dotsc, X_n$$ are i.i.d. standard normal variables, i.e. $$X_i \sim N(0,1)$$ with joint density $$g(\mathbf{x}) = g(x_1, \dotsc, x_n)$$, then by independence their joint distribution factorizes as follows:

$$ g(\mathbf{x}) = g(x_1, \dotsc, x_n) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x_i^2} = \frac{1}{(2\pi)^{\frac{n}{2}}} e^{-\frac{1}{2}\sum_{i=1}^n x_i^2} =  \frac{1}{(2\pi)^{\frac{n}{2}}} e^{-\frac{1}{2} \mathbf{x^T}\mathbf{x}}$$

The vectorized form is hand once we drop the assumption of independence and consider the random vector $$Z = \mu + AX$$ where $$A$$ is an invertible matrix, i.e. $$A \in \mathbb{R}^{n \times n}, \det(A) \neq 0$$, and $$\mu \in \mathbb{R}^n$$ is a vector. They are both deterministic, i.e. their values are fixed. The randomness in $$Z$$ only originates from the random vector $$X$$. The density of $$f(\mathbf{z}) = f(z_1, \dotsc, z_n)$$ is then given by the transformation theorem from before and the fact that $$X=A^{-1}(z-\mu)$$:

$$\begin{align}  f(\mathbf{z}) &= f(z_1, \dotsc, z_n)  \\ 
&= \frac{1}{(2\pi)^{\frac{n}{2}}} \frac{1}{\det(A)}e^{-\frac{1}{2}  X^TX}  \\
&= \frac{1}{(2\pi)^{\frac{n}{2}}} \frac{1}{\det(A)}e^{-\frac{1}{2}  (A^{-1}(z-\mu))^T(A^{-1}(z-\mu))}\\
&= \frac{1}{(2\pi)^{\frac{n}{2}}} \frac{1}{\det(A)}e^{-\frac{1}{2}  (z-\mu)^TA^{-1^T}A^{-1}(z-\mu))} \\ 
&= \frac{1}{(2\pi)^{\frac{n}{2}}} \frac{1}{\det(A)}e^{-\frac{1}{2}  (z-\mu)^TA^{-1^2}(z-\mu))}   
\end{align}$$ 

If we now define $$\Sigma := AA^T$$ then we get the familiar formula

$$f(\mathbf{z}) = f(z_1, \dotsc, z_n) =  \frac{1}{(2\pi)^{\frac{n}{2}}\det(\Sigma^{\frac{1}{2}})} e^{-\frac{1}{2}  (z-\mu)^T\Sigma^{-1}(z-\mu)}$$

This is the multivariate normal density and often abbreviated as $$N(\mu, \Sigma)$$. The expectation is $$\mu$$ as $$E(Z) = E(\mu +  AX) = \mu + 0$$. The matrix $$\Sigma$$ is called covariance matrix as its entries correspond exactly to the definition of the covariance:

$$\begin{align}  Cov(Z,Z) &= E((Z-E(Z))(Z-E(Z))^T)  \\ 
&= E((Z-\mu)(Z-\mu)^T)  \\
&= E((\mu +AX - \mu)(\mu + AX-\mu)^T)\\
&=  E(AX(AX)^T)\\ 
&=  E(AXX^TA^T) \\   
&= AE(XX^T)A^T \\ 
&= AA^T \\  
&= \Sigma   
\end{align}$$ 

Hence the the covariance between individual random variables $$Z_i$$ and $$Z_j$$ are simply entries in the covariance matrix $$\Sigma$$:

$$Cov(Z_i, Z_j) = E((Z_i - \mu_i)(Z_j-\mu_j)) = \Sigma_{ij}$$

# Variational Autoencoders

At its heart, this post is about generative modeling, or generating data from a specific distribution. This distribution could be quite complex, such as the distribution of all cats, airplanes or faces. Of course we are not limited to images but could model anything that can be expressed as a tensor, however the more complicated the distribution the more difficult it will be to generate examples. For this kind of modeling we do not need any labels as in supervised learning, although they can help in some cases. The most frequently used example for generative modeling right now would be Generative Adversarial Networks, while the approach is somewhat related, there are numerous differences. Most importantly the adversarial learning part is absent from variational autoencoders. The task we are going to learn is how to generate images from the MNIST data set. These are the first ten samples from the MNIST training set

![Plot of the first ten images in MNIST](/assets/images/on_variational_autoencoders/mnist_data.png)

MNIST images consist of pixels with values in $$[0,1]$$ in a $$28 \times 28$$ grid, i.e. they are objects in the space $$[0,1]^{28 \times 28}$$. So if we want to generate an example from that distribution we would have to pick $$784$$ values between zero and one and arrange them meaningfully in a grid of size $$28$$ by $$28$$. This is far from a trivial task, even with a lot of computational power. Computational tools such as NumPy and PyTorch only allow us to sample from one dimension, usually from a well known distribution such as the standard normal or uniform. So is there a way to produce more complex distributions from simple ones? This is what variational autoencoders do, and as the name indicates it involves an autoencoder like structure.

More formally, suppose we have samples $$x \sim X$$ , where $$X$$ is a random variable and we would like to learn its distribution $$P(X)$$. In our case $$X$$ would be a random variable generating MNIST images, so $$X \subset [0,1]^{28 \times 28}$$ and samples $$x \sim X$$ would be images as the ones above. Suppose we can easily generate samples $$z \sim Z$$ from a random variable $$Z \subset \mathbb{R}^m$$. Then we would like to find a transformation $$f_{\theta}$$ which takes a sample $$z \sim Z$$ and transforms it into an image $$x \sim X$$, or formally: find $$f_{\theta} : \mathbb{R}^m \longrightarrow [0,1]^{28 \times 28}$$ such that $$f_{\theta}(z) \sim X$$ for $$z \sim Z$$.

Finding a suitable $$f_{\theta}$$ is approximating a function, something we know how to do well with lots of data, neural networks and gradient descent, if certain conditions are met. However meeting these conditions is not that obvious, and will be what occupies us for the rest of this post.

## The main issue

When we train a supervised model, then the goal is usually pretty simple: we want to maximize its accuracy. We do this by 'punishing' inaccurate predictions and 'rewarding' accurate ones. But how do we produce a signal in an unsupervised setting?  The only data we have are random samples $$x_i$$ from a training set 

$$\mathbb{D} = \lbrace x_1, \dotsc, x_n \rbrace$$

We assume they are produced by an unknown process $$X$$ with the underlying or true probability distribution $$p^*(x)$$. We would like to approximate that distribution with a model $$p_{\theta}(x)$$, such that

$$p_{\theta}(x_i) \approx p^*(x_i) \qquad \forall x_i \in \mathbb{D}$$

Approximation is done by learning parameters $$\theta$$ which approximates the true distribution $$p^*(x)$$ as closely as possible. If we assume the data is independently and identically distibuted (i.i.d.)  then the joint distribution factorizes as a product of individual distributions

$$ p_{\theta}(x_1, \dotsc, x_n) = p_{\theta}(x_1) \cdot \dotsc \cdot p_{\theta}(x_n) = \prod_{i=1}^n p_{\theta}(x_i)$$

and the log probability is simply the sum of the individual probabilities

$$\log p_{\theta}(\mathbb{D}) = \log p_{\theta}(x_1, \dotsc, x_n)= \sum_{i=1}^n \log(p_{\theta}(x_i))$$

Maximizing this log likelihood is the same as minimizing the Kullback-Leibler divergence between the data $$p^*(x)$$ and the model distribution $$p_{\theta}(x)$$, we will see that later in more detail. When we do gradient descent we generally do not optimize over the whole data set, but over a randomly selected subset, so called stochastic gradient descent.

We can extend the above model by introducing latent variables, variables which are part of the model but not observed in the data and hence not part of the data set. They are generally denoted by $$z$$. Incorporating this latent variable into our model gives us a joint distribution $$p_{\theta}(x,z)$$ over both variables. If we want to extract the model for only the observed data we can simply integrate out the hidden variable $$z$$:

$$  \begin{equation} p_{\theta}(x) = \int p_{\theta}(x, z)dz \end{equation} $$

This is called the marginal likelihood and is a function of $$\theta$$. If the distribution model $$p_{\theta}(x, z)$$ is represented by a neural network, we call this a deep latent variable model (DLVM). The main problem with maximum likelihood learning in that setting is that the integral is intractable, i.e. it has no analytic solution and can not be approximated efficiently with computations. However the marginal likelihood $$p_{\theta}(x)$$ is also related to the joint distribution via other distributions:

$$p_{\theta}(x,z) = p_{\theta}(z)p_{\theta}(x \mid z)$$

and

$$p_{\theta}(x,z) = p_{\theta}(x)p_{\theta}(z \mid x)$$

However, while $$p_{\theta}(x, z)$$ is traceable to compute, the posterior distribution of the latent variables given the data,  $$p_{\theta}(z \mid x)$$ is also intractable. If however $$p_{\theta}(z \mid x)$$ could be approximated, so could $$p_{\theta}(x)$$. The relationship of course also holds the other way around. This is where variational autoencoders come in, they provide a framework to to optimize deep latent variable models.

In order to make this problem solvable we introduce a model $$q_{\phi}(z \mid x)$$. This model is often called an encoder or inference model, as we are predicting latent variables $$z$$ from data $$x$$. Where $$\phi$$ is called the variational parameters and optimized to approximate

$$ q_{\phi}(z \mid x) \approx p_{\theta}(z \mid x) $$

Having the approximation will help us optimize the marginal likelihood or $$p_{\theta}(x)$$ through the relationship seen above. Specifically we learn a neural net $$g_{\phi} : \mathbb{R}^n \longrightarrow \mathbb{R}^{m \times 2}$$ which will produce the parameters to a multivariate Gaussian distribution $$g_{\phi}(x) = (\mu, \log(\sigma))$$. The probability distribution is then given by 

$$q_{\phi}(z \mid x) = N(z | \mu, \text{diag}(\sigma)) $$

It is important here that we put the elements of $$\sigma$$ produced by the neural network into a diagonal matrix. This means the elements $$z_i$$ are presumed to be independent, i.e. there exists no correlation between them.

Insert image here between the mapping of p(x given z) and q(z given x). Note that this is stochastic embedding, no collusion between encoder and decoder



## Creating an objective to optimize

Remember from before that we want to maximize the log likelihood $$\log(p_{\theta}(x))$$ over the observed data $$\mathbb{D}$$. But as we have seen this is not directly possible, we need to optimize a different objective, called the evidence lower bound (ELBO):

$$ \mathcal{L}_{\phi, \theta}(x) := \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(\frac{p_{\theta}(x,z)}{q_{\phi}(z \mid x)}\bigg)\bigg]$$

For now this is just a definition, an objective whose values depend on the model parameters $$\phi$$ and $$\theta$$ as well as the input $$x \in X_T$$. There is a lot to unpack in this expression, and we will do that just after we saw where it comes from.

### Deriving the ELBO by maximizing the log likelihood

There are multiple ways to deduce the ELBO, here we start with the goal to maximize the log likelihood or probability of a sample $$x \in X_T$$:

$$\begin{align} \log\big(p_{\theta}(x)\big) &= \int q_{\phi}(z \mid x) \log\big(p_{\theta}(x)\big)dz  \\ 
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\big(p_{\theta}(x)\big) \bigg] \\
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(\frac{p_{\theta}(x,z)}{p_{\theta}(z \mid x)}\bigg) \bigg]  \\  
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(\frac{p_{\theta}(x,z)}{q_{\phi}(z \mid x)}\frac{q_{\phi}(z \mid x)}{p_{\theta}(z \mid x)}\bigg) \bigg]\\   
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(\frac{p_{\theta}(x,z)}{q_{\phi}(z \mid x)}\bigg) + \log \bigg(\frac{q_{\phi}(z \mid x)}{p_{\theta}(z \mid x)}\bigg) \bigg] \\
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(\frac{p_{\theta}(x,z)}{q_{\phi}(z \mid x)}\bigg)\bigg] +  \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg(\frac{q_{\phi}(z \mid x)}{p_{\theta}(z \mid x)}\bigg) \bigg] \\   
&=   \mathcal{L}_{\phi, \theta} (x) + D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z \mid x)\big) \\ 
&\geq \mathcal{L}_{\phi, \theta} (x)  
\end{align}$$

In the last line we see why it is called evidence *lower bound*, it bounds the probability of the data from below. We used the fact that the KL-divergence is bigger or equal to zero, i.e. $$D_{KL}(p \mid \mid q) \geq 0 \: \forall p,q$$.

### Deriving the ELBO from posterior distribution

Another way to introduce the ELBO is by trying to find an encoder $$q_{\phi}(z \mid x)$$ which matches the decoder distribution of the latent data $$p_{\theta}(z \mid x)$$, i.e. find parameters $$\phi$$ such that

$$ q_{\phi}(z \mid x) \approx p_{\theta}(z \mid x) $$

Measuring the 'distance' between these two distributions is often done by the KL-divergence, hence we would try to optimize

$$D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z \mid x)\big)$$

Remember that we have no way of computing $$p_{\theta}(z \mid x)$$ directly, so we can not calculate the divergence or any optimization based on it in that form. However, once we start applying Bayes theorem to $$p_{\theta}(z \mid x)$$ we can get to a more manageable objective:

$$\begin{align} D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z \mid x)\big) &= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg(\frac{q_{\phi}(z \mid x)}{p_{\theta}(z \mid x)}\bigg) \bigg]  \\ 
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg(q_{\phi}(z \mid x)\bigg) - \log\bigg(p_{\theta}(z \mid x)\bigg) \bigg] \\
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg(q_{\phi}(z \mid x)\bigg) - \log\bigg(\frac{p_{\theta}(x \mid z) p_{\theta}(z)}{p_{\theta}(x)}\bigg) \bigg]   \\  
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg(q_{\phi}(z \mid x)\bigg) - \log\bigg(p_{\theta}(x \mid z)\bigg) - \log\bigg(p_{\theta}(z)\bigg) + \log\bigg(p_{\theta}(x)\bigg) \bigg] \\   
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg(\frac{q_{\phi}(z \mid x)}{p_{\theta}(z)}\bigg)  - \log\bigg(p_{\theta}(x \mid z)\bigg) + \log\bigg(p_{\theta}(x)\bigg) \bigg]   \\
&=  \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg(\frac{q_{\phi}(z \mid x)}{p_{\theta}(z)}\bigg) \bigg]  - \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(p_{\theta}(x \mid z)\bigg) \bigg]  + \log\big(p_{\theta}(x)\big)\\   
&= D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z)\big)  - \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \big[ \log\big(p_{\theta}(x \mid z)\big) \big]  + \log\big(p_{\theta}(x)\big) 
\end{align}$$

We see that similar terms appear as in the previous derivation, there is the log probability of the data and the same KL-divergence term we removed in the second last step. We can rearrange the terms, by moving those two to the left and multiplying with $$-1$$:

$$\begin{align} \log\big(p_{\theta}(x)\big) - D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z \mid x)\big) &=    \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \big[ \log\big(p_{\theta}(x \mid z)\big) \big] - D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z)\big)\\ 
& \Leftrightarrow  \\
\log\big(p_{\theta}(x)\big) & \geq \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \big[ \log\big(p_{\theta}(x \mid z)\big) \big] - D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z)\big)   \\  
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(p_{\theta}(x \mid z)\bigg) \bigg] + \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ -\log \bigg(\frac{q_{\phi}(z \mid x)}{p_{\theta}(z)}\bigg) \bigg]    \\   
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log\bigg(p_{\theta}(x \mid z)\bigg) + \log \bigg( \frac{ p_{\theta}(z)}{ q_{\phi}(z \mid x)}  \bigg) \bigg]   \\
&= \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg( \frac{ p_{\theta}(x \mid z) p_{\theta}(z)}{ q_{\phi}(z \mid x)}  \bigg) \bigg]  \\   
&=  \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \bigg[ \log \bigg( \frac{ p_{\theta}(x ,z)}{ q_{\phi}(z \mid x)}  \bigg) \bigg]   \\
&=  \mathcal{L}_{\phi, \theta}(x)

\end{align}$$

We end up with the same equation as before! That should not be a surprise, but in the process we had an interesting intermediate step where the inequality is:

$$\log\big(p_{\theta}(x)\big) \geq \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \big[ \log\big(p_{\theta}(x \mid z)\big) \big] - D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z)\big)$$

The right side is an decomposition of the ELBO which looks like an autoencoder:

- Maximizing $$\mathbb{E}_{z \sim q_{\phi}(z \mid x)} \big[ \log\big(p_{\theta}(x \mid z)\big) \big]$$ tells our decoder $$p_{\theta}$$ to produce samples with high likelihood from data which is produced by the encoder $$q_{\phi}$$
- Minimizing $$D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z)\big)$$ tells or encoder to match the latent variable distribution of our choosing.

### Notes on the ELBO

How close the ELBO $$\mathcal{L}_{\phi, \theta}(x)$$ can get to the goal of maximizing the marginal likelihood $$\log\big(p_{\theta}(x)\big)$$ depends on the KL-divergence term

$$ D_{KL} \big(q_{\phi}(z \mid x) \mid \mid p_{\theta}(z \mid x)\big)$$

In theory, if we have a model $$q_{\phi}$$ with high enough capacity, we could exactly replicate the posterior distribution and reduce this term to zero: the ELBO would match the marginal likelihood.

Under the i.i.d. assumption of the training data $$\mathbb{D}$$, ELBO allows us joint optimization over the entire training set as the estimator becomes simply the sum of all the samples

$$ \mathcal{L}_{\phi, \theta}(\mathbb{D}) = \sum_{x \in X_T} \mathcal{L}_{\phi, \theta}(x)$$

This property allows us to do stochastic gradient descent, which is what we will do next!

## Optimization of the ELBO

So far we have seen why the optimization of the ELBO is a good idea, but not how to do it, except for the vague idea of using gradient descent. Now we need to find a way to estimate the gradient with respect to both parameters $$\theta$$ and $$\phi$$.













# Implementing an Variational Autoencoder



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