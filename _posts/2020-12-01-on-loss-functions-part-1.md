---
layout: post
author: michael
title:  "On Loss Functions - Part I"
date:   2020-12-01 11:00:00 +0530
categories: DeepLearning ComputerVision CrossEntropy Optimization Loss Distance LinearRegression
---
# Introduction
When looking at a Deep Learning related project or paper, for me there are three fundamental parts: data, network architecture and loss function. As the title suggests this post will focus on the last part. Loss functions are deeply tied to the task one is trying to solve, and are often used as measures of progress during training. In this post we are going to see where they come from and why we useing the ones we do. The first part will cover the tasks of classification and segmantic segmentation.

# Notions of distance
At the very bottom of loss functions are how we measure distance. In our daily lifes we switch between different measures of distance effortlessly. For example one could ask how far away are you from the next airport? You might measure this in time it takes to drive there, or the distance to the location in kilometers. If you chose to measure it in the latter way, you are most likely using Euclidean distance, more on this in a short while. In Mathematics there is a more general notion of distance on a set $$X$$ ($$X$$ could be anything, numbers, apples, oranges, mathematical objects). A distance $$d$$ on the set $$X$$ is a function

$$d : X \times X \longrightarrow [0, \infty)$$

such that

$$\begin{align*} 
1. \; d(x,y) &\geq  0  & \forall x,y \in X\\ 
2. \; d(x,y) &= 0 \Leftrightarrow x = y & \forall x,y \in X\\
3. \; d(x,y) &= d(y,x)  & \forall x,y \in X \\
4. \; d(x,y) &\leq d(x,z) + d(z,y) & \forall x,y,z \in X
\end{align*}$$

These four conditions have straightforward interpretations. The first one tells us that a distance can never be negative. The second one says if the distance between two elements is zero, then they must be the same. The third one is symmetry, going from A to B is the same as going from B to A. The last one is called triangle inequality, it can never be faster to go from A to C to B than going from A to B directly.

The most well known example of a distance would be the Euclidian distance. In its general form the Euclidean distance between $$x$$ and $$y$$, $$\in \Re^n $$, two n-dimensional real vectors is defined as follows:

$$ d_E(x,y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2 }$$

It is also often called the L2 norm, and denoted by $$\mid\mid \cdot \mid\mid_2$$. In the two dimensional case this reduces to well known formula for calculating the hypothenuse of a triangle:

$$d_E(x,y) = \sqrt{(x_1 - y_1)^2 + (x_2-y_2)^2}$$

The L2 norm is also widely used in machine learning, for example in linear regression. Rember that in linear regressin we have a data matrix $$X \in \Re^{n \times m}$$ ($$n$$ entries for $$m$$ different variables), $$n$$ labels or targets $$y \in \Re^n$$ and a parameter vector $$\beta \in \Re^m$$ which we are trying to find such that $$y = X \beta$$. However once we have $$n > m$$, or more data than variables, there is no unique solution anymore. Hence we try to get as "as close as possible" to that state. And how do we measure closeness? By the squared L2 norm or Euclidean distance! We try to minimize

$$ d_E(X\beta, y)^2 = \mid \mid X \beta - y \mid \mid^2_2$$

Notice that we are trying to minimize the square of the Euclidian distance, this will not change the minimization result because it is a strictly increasing transformation. This distance measure has the advantage that we can easily calculate the optimal values for $$\beta$$ by taking the partial derivative with respect to $$\beta$$ 

$$\begin{align*} 
 \frac{\partial d_E(X \beta, y)^2}{\partial \beta} &=  \frac{\partial((X\beta y)^T(X\beta -y))}{\partial \beta}\\ 
  &= \frac{\partial(\beta^TX^TX\beta - \beta^TX^Ty-y^TX\beta -y^Ty)}{\partial \beta} \\
 &= 2X^TX\beta -X^Ty  -y^TX - 0 \\
&= 2X^TX\beta -2X^Ty
\end{align*}$$

To find the optimal we set the derivative with respect to $$\beta$$ to zero and solve for $$\beta$$
$$\begin{align*}
0 &= 2X^TX\beta -2X^Ty & \Leftrightarrow \\
2X^Ty &= 2X^TX\beta & \Leftrightarrow \\
(X^TX)^{-1}X^Ty &= \beta
\end{align*}$$

Where the last line gives us a unique solution for $$\beta$$, which only valid if the assumptions of linar regression are true.

The Euclidan distance works well for measuring the distance between vectors, but what if you want to predict categories? Say you build a classifier which tells you if an image contains a cat or not.




