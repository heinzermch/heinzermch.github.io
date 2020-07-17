---
layout: post
author: Michael Heinzer
title:  "On Loss Functions - Part II"
description: An introduction to loss functions for semantic segmentation
date:   2020-06-20 11:00:00 +0530
categories: DeepLearning DiceLoss F1Score CrossEntropy HarmonicMean SemanticSegmentation IntersectionOverUnion Precision Recall
comments: no
published: no
---
The second part of this series focuses on losses for semantic segmentation. There is considerably less mathematical motivation for

## Basic concepts and notation

Before we start, let us quickly repeat some basic concepts and their notation. Readers familiar with the topic may skip this section. This is not meant to be an introduction to probability theory or other mathematical concepts, only a quick refresh of what we will need later on.

- **Definition**: If a term on the left hand side of the equation is defined as as the right hand side term, then $$:=$$ will be used. This is similar to setting a variable in programming. As an example we can set $$g(x)$$ to be $$x^2$$ by writing $$g(x) := x^2$$. In mathematics, when writing simply $$=$$ means that the left side implies the right (denoted by $$\Rightarrow$$) and right side the left (denoted by $$\Leftarrow$$), at the same time.

- **Logarithm**: $$\log : (0,\infty) \longrightarrow (-\infty, \infty)$$. One of the basic function in calculus, known as the inverse of the exponential function $$ x = \exp(\log(x)) $$. It has some other properties which we will need later on:
  - $$ \log(xy) = \log(x) + \log(y)$$ and $$\log\big(\frac{x}{y}\big) = \log(x) - log(y)$$
  - For two bases $$b, k$$, we have that $$ \log_b(x) = \frac{\log_k(x)}{\log_k(b)} = C \cdot log_k(x) $$ where $$C = \frac{1}{\log_k(b)}$$ is a constant because it does not depend on $$x$$. This means that a change of basis in the log is only a multiplication by a constant.
  - If not mentioned otherwise, we will assume that $$b=e$$, the natural logarithm.
  - It is always useful to have an idea of how the plot of $$\log(x)$$ looks like.
    ![Plot of log(x) and -log(x)](/assets/images/loss_functions_part_1/log_plot.png)
    There are three cases which are worth noting:
    - Log goes to minus infinity when approaching zero: $$\lim_{x \longrightarrow 0 } \log(x) = -\infty$$
    - one is the only value for which log is equal to zero: $$\log(1) = 0 $$
    - Even if the graph looks like its flattening, it does actually go to infinity:$$ \lim_{x \longrightarrow \infty} \log(x) = \infty $$
  
- **Sigmoid**: $$ \sigma : (-\infty,\infty) \longrightarrow (0,1)$$, is defined as $$ \sigma(x) := \frac{1}{1+\exp(-x)} = \frac{\exp(x)}{1+\exp(x)}$$. It has the property that it maps any value to the open interval $$(0,1)$$, which is very useful if we want to extract a probability from a model.
  - The plot looks as follows:
    ![Plot of sigmoid(x)](/assets/images/loss_functions_part_1/sigmoid_plot.png)
  - Again three cases worth noting:
    - The left limit is 0: $$ \sigma(x)_{x \longrightarrow -\infty } = 0 $$
    - The right limit is 1: $$ \sigma(x)_{x \longrightarrow \infty } = 1 $$
    - At zero we are in the middle of the limits: $$ \sigma(0) = 0.5$$
  - The derivative of the sigmoid is $$\frac{\partial \sigma(x)}{\partial x} =  \sigma(x) (1-\sigma(x))$$
  
- 
  

# Semantic Segmentation

In the previous post on loss functions we were trying to predict a single number in regression, or a single class in classification for an entire image. Computer vision has much more challenging tasks however, and one of those is semantic segmentation.

## Task

The goal here is an extension of classification to the entire image. Instead of predicting a single class on an image level we will try to predict a single class per pixel. The following picture illustrates the task

![Input image and segmented output mask](/assets/images/loss_functions_part_2/semantic_segmentation_task.png)

On the left side we have an input image and on the right side a mask which assigns each pixel to a class. Here we have the classes cat, dog, sofa, background and possibly a window on the upper left side. For illustration purposes the borders have a slightly different shade. What we are not doing in this task is to separate different instances of the same class, if we would have multiple dogs in the same image they would all have the same class or pixel value.

## Challenge

Besides the architectural and computational requirements for this task there is also the problem of balancing the classes. Often there will be images which consist almost entirely of background. A network which would always predict the background class would achieve a high accuracy, we will need new measures of accuracy to deal with this problem. And at least for the binary classification problem, we can address the imbalance with the loss function directly. But first we have to go through some basic concepts of means.

# A detour through different means

It might not be immediately obvious what different definitions of means have to do with semantic segmentation, but at the end of this chapter it should hopefully become clearer. Most readers are probably familiar with the definition of an average $$\frac{1}{n} \sum^n_{i=1} x_i$$ which is one of the three Pythagorean means, which have been studied by the ancient Greeks.

## Pythagorean Means

There are three classical Pythagorean means in mathematics: the arithmetic, geometric and harmonic. Let us have a look at the definition for these three means for data consisting of $$x_1, \dotsc, x_n$$ positive real numbers.

### Arithmetic mean

Most readers are probably familiar with the arithmetic mean which is commonly called mean or average. The definition of the arithmetic mean is

$$ A(x_1, \dotsc, x_n) = \frac{1}{n} \sum^n_{i=1} x_i $$

The arithmetic mean has the disadvantage of not being a robust statistics, it is greatly influenced by outliers. 

### Geometric mean

The geometric mean has its name from the geometric interpretation, in the case of $$n=2$$ it corresponds to the side of a square with area $$x_1 \cdot x_2$$. The general definition for $$n$$ elements is

$$ G(x_1, \dotsc, x_n) = \bigg(\prod^n_{i=1} x_i \bigg)^{\frac{1}{n}}$$

It has the property that a percentage change in one element of the data $$x_i$$ has the same effect on the geometric mean as any other, i.e. it is independent of the absolute value of $$x_i$$. For example if you have two elements which have a large difference in value: $$x_1 = 3, x_2=90$$, then it doesn't matter which one we increase by 10%, the resulting geometric mean will be the same:

$$G(3.3, 90) =\sqrt{3.3\cdot 90} = \sqrt{297}  \text{ and } G(3, 99) = \sqrt{3 \cdot 99} = \sqrt{297}$$

This is easy to see if you remember that increasing a number $$x_i$$ by 10% is the same as multiplying by $$1.1$$, which can be moved around freely inside the factors of the geometric but not the arithmetic mean.

### Harmonic mean

The last of the three Pythagorean means is the harmonic mean, it is defined as

$$ H(x_1, \dotsc, x_n) = \frac{n}{\frac{1}{x_1} + \dotsc + \frac{1}{x_n}} = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}$$

While the arithmetic mean has the property that it is susceptible to large outliers, the harmonic mean has the same tendency for small values. However it is appropriate for situation where we want to compare the average of rates or ratios. 

In the special case where $$n=2$$ we can rewrite it as follows:

$$ H(x_1, x_2) = \frac{2}{\frac{1}{x_1} + \frac{1}{x_2}} = \frac{2}{\frac{1}{x_1} + \frac{1}{x_2}} \frac{x_1 x_2}{x_1 x_2} = \frac{2 x_1 x_2}{x_2 + x_1}$$



### Relationships between the Pythagorean means

For positive numbers $$x_1, \dotsc, x_n$$ where not all numbers are the same, we have that

$$\min(x_1, \dotsc, x_n) \leq H(x_1, \dotsc, x_n) \leq G(x_1, \dotsc, x_n) \leq A(x_1, \dotsc, x_n) \leq \max(x_1, \dotsc, x_n)$$

and equality if and only if all values $$x_1, \dotsc, x_n$$ are the same. We can plot that visually 

![Plot of different means](/assets/images/loss_functions_part_2/pythagorean_means.png)

## Measuring Accuracy

In general, the more complex a tasks is, the more careful we have to be about how we measure performance. In the previous post on loss functions we did not discuss any accuracy measure at all, we implicitly assumed that we just care about predicting the correct class for each image and that was good enough. And this is a good start when doing multi-class classification on balanced data such as MNIST. However if we do pixel-wise classification on imbalanced data, classification accuracy can be very misleading.

Assume we have an image with a resolution of $$1000 \times 1000$$ and two classes: background and foreground. The foreground consists of an object of size $$100 \times 100$$ somewhere in the image. The example is illustrated in the following image with black as background and white as foreground:

![Plot of different means](/assets/images/loss_functions_part_2/classification_problem.png)

If a classifier would only predict background, it would get $$99\%$$ pixel-wise accuracy in this image, yet it is clearly missing what interests us most. This is the case for most semantic segmentation tasks, the background class will often dominate by a large margin. More appropriate would be an average of a class-wise score. Here it would be $$100\%$$ accuracy for the background and $$0\%$$ for the foreground, the average of both would then be $$50\%$$. This is not too far from what is used in practice. However we also have to penalize wrong predictions in our accuracy measure, this is where the intersection over union measure comes in.

### Intersection over Union

Often sets are introduced by using circles or other arbitrary shapes, however here we will focus on their interpretation in images. Suppose we have $$P, G \in \lbrace 0, 1 \rbrace^{3 \times 3}$$ two discrete sets or masks on an $$3 \times 3$$ image:

$$P = \begin{pmatrix}
 0 & 1 & 0 \\
 1 & 1 & 0 \\
0 &0 & 0  \\
\end{pmatrix},

G = \begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 & 1 \\
0 &0 & 0  \\
\end{pmatrix}$$

We now introduce three functions on a more general space consisting of $$n \times m$$ matrices with zero and one as entries:

- A count measure $$\mid \cdot \mid$$ which counts the numbers of pixels which are one:
  $$\begin{align}  \mid \cdot \mid &: \lbrace 0, 1 \rbrace^{n \times m}  \longrightarrow \mathbb{N}  \\ 
  |A| &= \sum^n_{i=1} \sum^m_{j=1} A_{ij}  \end{align}$$

- The intersection operator $$\cap$$ which is a pixel-wise logical AND operator  
  $$ \cap : \lbrace 0, 1 \rbrace^{n \times m} \times  \lbrace 0, 1 \rbrace^{n \times m}  \longrightarrow  \lbrace 0, 1 \rbrace^{n \times m} $$   
  where for $$A, B \in \lbrace 0, 1 \rbrace^{n \times m}$$ we have  
  $$ A \cap B = C \quad\text{ is } \quad A_{ij} \cap B_{ij} = C_{ij} = \begin{cases}
      1& \text{if } A_{ij}=1 \text{ and } B_{ij} = 1 \\
      0,              & \text{otherwise}
  \end{cases}$$

- The union operator $$\cup$$ which is a pixel-wise logical OR operator:  
  $$ \cup : \lbrace 0, 1 \rbrace^{n \times m} \times  \lbrace 0, 1 \rbrace^{n \times m}  \longrightarrow  \lbrace 0, 1 \rbrace^{n \times m} $$  

  where for $$A, B \in \lbrace 0, 1 \rbrace^{n \times m}$$ we have  
  $$ A \cup B = C \quad\text{ is } \quad A_{ij} \cup B_{ij} = C_{ij} = \begin{cases}
      1& \text{if } A_{ij}=1 \text{ or } B_{ij} = 1 \\
      0,              & \text{otherwise}
  \end{cases}$$

If we apply the intersection and union operator to the example matrices $$A$$ and $$B$$ from above, we get

$$P \cap G = \begin{pmatrix}
 0 & 1 & 0 \\
 1 & 1 & 0 \\
0 &0 & 0  \\
\end{pmatrix} \cap \begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 & 1 \\
0 &0 & 0  \\
\end{pmatrix} = 

\begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 &0 \\
0 &0 & 0  \\
\end{pmatrix}$$

$$P \cup G = \begin{pmatrix}
 0 & 1 & 0 \\
 1 & 1 & 0 \\
0 &0 & 0  \\
\end{pmatrix} \cup \begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 & 1 \\
0 &0 & 0  \\
\end{pmatrix} = 

\begin{pmatrix}
 0 & 1 & 0 \\
 1 & 1 &1 \\
0 &0 & 0  \\
\end{pmatrix}$$

Additionally we can apply the count operator to get

$$ \mid P \cap G \mid = 2 \text{ and } \mid P \cup G \mid = 4$$

We now have all we need to define the intersection over union operator for two arbitrary binary sets $$A, B \in  \lbrace 0, 1 \rbrace^{n \times m}$$:

$$\begin{align}  IoU &: \lbrace 0, 1 \rbrace^{n \times m} \times  \lbrace 0, 1 \rbrace^{n \times m} \longrightarrow \mathbb{R}  \\ 
IoU(A,B) &= \frac{\mid A \cap B \mid}{\mid A \cup B \mid} \quad \text{for } \mid A \cup B \mid > 0 \end{align}$$

The IoU has certain noteworthy properties:

- It will always be between zero and one: $$ 0 \leq IoU(A,B) \leq 1$$ with the two extremes reached when
  - We have that $$IoU(A,B) = 0$$ if and only if $$\mid A \cap B \mid = 0$$, i.e. when there is no overlap between the two sets.
  - We have that $$IoU(A,B) = 1$$ if and only if $$\mid A \cap B \mid = \mid A \cup B \mid$$, i.e. when the two sets are equal $$A=B$$.
- For the IoU to be defined, at least one set needs to be non-empty.
- The IoU is scale invariant, it only depends on the proportion of the overlap compared to the surface of their union.
- The IoU is symmetric, i.e. $$IoU(A, B) = IoU(B,A)$$.



### Harmonic mean and $$F_1$$ score

For binary classification problems as in the image above, predictions can be classified in four categories:

- True Positive (TP): Class $$1$$ predicted as $$1$$.
- False Positive (FP): Class $$0$$ predicted as $$1$$.
- True Negative (TN): Class $$0$$ predicted as $$0$$.
- False Negative (FN): Class $$1$$ predicted as $$0$$.

Ideally we would only have true positives and true negatives, but this is rarely the case in practice. Hence we need to take into account the errors our algorithm makes with the false positives and false negatives. This is why these two ratios were invented:

- Precision: 
  - Informally: How many selected items are relevant.
  - Formally: $$\frac{TP}{TP+FP}$$
- Recall: 
  - Informally: How many relevant items are selected.
  - Formally: $$\frac{TP}{TP+FN}$$

They describe how accurate the class one predictions (precision) and how many class one elements we are able to recover (recall). As both are ratios they are best combined using the harmonic mean:

$$H(\text{precision}, \text{recall}) = \frac{2}{\frac{1}{\text{precision}} + \frac{1}{\text{recall}}} = 2 \frac{ \text{precision} \cdot \text{recall}}{\text{precision}+\text{recall}} = F_1$$

This ratio is called the $$F_1$$ score or Dice coefficient.

### Relationship between $$F_1$$ and IoU score

So far we saw why we would rather use IoU than a classic arithmetic mean definition of accuracy for a semantic segmentation task. We also saw a second approach of using a $$F_1$$ score for binary classification problems.  We can relate those two measures when we decompose them in terms of $$TP, FP, TN, FN$$, first lets look at the $$F_1$$ score in these terms:

$$\begin{align*} 
 F_1 &=  2 \frac{ \text{precision} \cdot \text{recall}}{\text{precision}+\text{recall}}\\ 
  &= 2 \frac{\frac{TP}{TP+FP} \cdot \frac{TP}{TP+FN}}{\frac{TP}{TP+FP} + \frac{TP}{TP+FN}} \\
 &=  2\frac{\frac{TP^2}{(TP+FP)(TP+FN)}}{\frac{TP}{TP+FP} + \frac{TP}{TP+FN}}\frac{(TP+FP)(TP+FN)}{(TP+FP)(TP+FN)} \\
&= \frac{2TP^2}{TP(TP+FN) + TP (TP+FP)}  \\  
&= \frac{2TP}{2TP+FP+FN}

\end{align*}$$

Then at the intersection over union for two sets $$P$$ as prediction and $$G$$ as ground truth

$$\begin{align*} 
 IoU(P,G) &=  \frac{\mid P \cap G \mid }{\mid P \cup G \mid}\\ 
  &=  \frac{TP}{TP+FP+FN}\\

\end{align*}$$

While the numerator is clearly a TP, it might be less clear why the denominator is the sum of $$FP$$, $$TP$$ and $$FN$$. We can look at the following image for a visual interpretation of the denominator 

![Intersection over unition and TPs](/assets/images/loss_functions_part_2/TPs_and_intersection_over_union.png)

Or we can go back to the three by three matrix example and insert the definition of TP, FP and FN:

$$P \cup G = \begin{pmatrix}
 0 & 1 & 0 \\
 1 & 1 & 0 \\
0 &0 & 0  \\
\end{pmatrix} \cup \begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 & 1 \\
0 &0 & 0  \\
\end{pmatrix} = 

\begin{pmatrix}
 0 & TP & 0 \\
 FP & TP & FN \\
0 &0 & 0  \\
\end{pmatrix}$$

We simply sum over all the elements that appeared in union, which consists of these three cases. If we write both next to each other the similarities become more clear:

$$\begin{align*} 
 IoU(P,G) &=   \frac{TP}{TP+FP+FN} = \frac{\mid P \cap G\mid}{ \mid P \cup G \mid}\\   
& \\  
F_1(P, G)&= \frac{2TP}{2TP+FP+FN} = \frac{2\mid P \cap G \mid}{\mid P \mid + \mid G \mid} \\

\end{align*}$$

While they are not exactly the same, they are positively correlated and equal at the extremes. Both are bounded by zero from below and one from above.  In general we will always have that $$IoU \leq F_1$$, hence IoU is a more pessimistic measure of accuracy than the $$F_1$$ score.

Why do we care about those two measures? Because in the next chapter we will directly optimize for the $$F_1$$ score by using it as a loss function, while we generally measure accuracy with the IoU score, therefore it is important to have a sense of how these two scores relate to each other.

# Loss Functions

Let us first consider the binary classification case, where all the metrics we developed in the previous chapter apply nicely and we can basically plug in the dice or $$F_1$$ coefficient. For all of the following cases we suppose we have an input image of size $$n \times m$$ image and would like to predict a segmentation map with the same dimensions.

## Two Classes - Dice Loss

For the binary classification case we will have ground truth and prediction data of the following form

- Ground truth: $$G \in \lbrace 0, 1 \rbrace^{n \times m}$$ a label matrix containing elements which are either zero or one.
- Prediction: $$P \in [0,1]^{n \times m}$$ a prediction matrix for which each entry is the probability of the entry belonging to class one.

### The loss function

Suppose we have two classes and need a loss function to optimize for semantic segmentation with strong class imbalance. We define the dice loss $$l_D$$ as function from the label and prediction space to a real number:

$$l_D : \lbrace 0, 1 \rbrace^{n \times m}  \times [0,1]^{n \times m} \longrightarrow \mathbb{R}$$

where

$$l_D(P, G) := \frac{2 \sum^n_{i=1}  \sum^m_{j=1} p_{ij} g_{ij}}{\sum^n_{i=1}  \sum^m_{j=1} p_{ij}^2 + \sum^n_{i=1}  \sum^m_{j=1} g_{ij}^2} = \frac{2 \| P \cdot G \|_1}{\|P\|_2^2 + \| G\|^2_2}$$

Note that we square the denominator element wise. The loss described above is taken from the [V-Net paper](https://arxiv.org/abs/1606.04797), however there are other papers which use it [without](https://arxiv.org/abs/1608.04117) the square. We can adapt the previous example to get an idea of what the loss function does. This time the prediction matrix $$P$$ will contain probabilities instead of labels

$$P = \begin{pmatrix}
 0.12 & 0.83 & 0.23 \\
 0.89 & 0.94 & 0.36 \\
0.23 &0.06 & 0.18  \\
\end{pmatrix},

G = \begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 & 1 \\
0 &0 & 0  \\
\end{pmatrix}$$

Note that if we would threshold $$P$$ by setting everything above $$0.5$$ to one, we would get the same matrix again as before. For the numerator the operation is simply the L1 norm (denoted by $$\| \cdot \|_1$$)  of the dot product between the two matrices:

$$2\| P \cdot G \|_1 = 2\begin{Vmatrix} \begin{pmatrix}
 0.12 & 0.83 & 0.23 \\
 0.89 & 0.94 & 0.36 \\
0.23 &0.06 & 0.18  \\
\end{pmatrix}\cdot \begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 & 1 \\
0 &0 & 0  \\
\end{pmatrix} \end{Vmatrix}_1 = 2\begin{Vmatrix} \begin{pmatrix}
 0 & 0.83 & 0 \\
 0 & 0.94 & 0.36 \\
0 &0 & 0  \\
\end{pmatrix} \end{Vmatrix}_1 = 4.26$$

The denominator is the sum of the squared L2 norm (denoted by $$\| \cdot \|_2$$) of the two matrices:

$$\| P \|_2^2 + \| G \|_2^2 = \begin{Vmatrix} \begin{pmatrix}
 0.12 & 0.83 & 0.23 \\
 0.89 & 0.94 & 0.36 \\
0.23 &0.06 & 0.18  \\
\end{pmatrix} \end{Vmatrix}_2^2 + 

\begin{Vmatrix} \begin{pmatrix}
 0 & 1 & 0 \\
 0 & 1 & 1 \\
0 &0 & 0  \\
\end{pmatrix} \end{Vmatrix}_2^2

 = 2.6651 + 3 =5.6651$$

So overall we have a loss value of $$l_D(P,G) = \frac{4.26}{5.6651} = 0.752$$. Note that for the two edge cases of perfect and zero prediction we have,

- Perfect prediction: $$P=G$$, the loss value is $$l_D(G,G) = 1$$.
- Zero predictions: $$P=0$$, the loss value is $$l_D(0, G) = 0$$.

### The gradient function

If we want to do gradient descent, then we would need a loss of zero for perfect predictions, so it is customary to take $$1-l_D$$ for minimization. The gradient of this loss with respect to one input value $$p_{xy}$$ in the loss function is:

$$\begin{align*} 
 \frac{\partial l_D(P,G)}{\partial p_{xy}} &= 2\frac{ (\frac{\partial}{\partial p_{xy}} \| P \cdot G \|_1 )(\|P\|_2^2 + \| G\|^2_2) - (\frac{\partial}{\partial p_{xy}}\|P\|_2^2 + \frac{\partial}{\partial p_{xy}}\| G\|^2_2) \| P \cdot G \|_1}{(\|P\|_2^2 + \| G\|^2_2)^2} \\ 
  &=2\frac{ (\sum^n_{i=1}  \sum^m_{j=1} \frac{\partial}{\partial p_{xy}}p_{ij} g_{ij} )(\|P\|_2^2 + \| G\|^2_2) - (\sum^n_{i=1}  \sum^m_{j=1} \frac{\partial}{\partial p_{xy}}p_{ij}^2 + 0) \| P \cdot G \|_1}{(\|P\|_2^2 + \| G\|^2_2)^2}  \\
 &= 2\frac{ g_{xy} (\|P\|_2^2 + \| G\|^2_2) - 2p_{xy} \| P \cdot G \|_1}{(\|P\|_2^2 + \| G\|^2_2)^2}  \\
&=  \frac{ 2g_{xy} }{(\|P\|_2^2 + \| G\|^2_2)} - \frac{4p_{xy} \| P \cdot G \|_1}{(\|P\|_2^2 + \| G\|^2_2)^2} 

\end{align*}$$

The interpretation of these two terms is not exactly straightforward. We can see that the first term has only two possible values

- If $$g_{xy} = 1$$: then $$\frac{2}{(\|P\|_2^2 + \| G\|^2_2)}$$
- If $$g_{xy} = 0$$: then it will be zero too.

Note as we have the $$P$$ term in both gradient terms, every pixel will influence the gradient of all other pixels. This makes this loss quite confusing, lets look at the two easiest cases to get an intuition of what is happening. 

Suppose we have a perfect prediction $$P=G$$, then we would like the gradient to be zero because there is nothing to improve. We can do the calculation as follows

$$\begin{align*} 
 \frac{\partial l_D(G,G)}{\partial p_{xy}} &= \frac{ 2g_{xy} }{(\|G\|_2^2 + \| G\|^2_2)} - \frac{ 4g_{xy} \| G \cdot G \|_1}{(\|G\|_2^2 + \| G\|^2_2)^2}  \\ 
  &= \frac{ 2g_{xy} }{2\|G\|_2^2} - \frac{ 4g_{xy} \| G\|_2^2}{(2\|G\|_2^2 )^2}  \\
 &=  \frac{ g_{xy} }{\|G\|_2^2} - \frac{g_{xy}}{\|G\|_2^2} \\
&=  0

\end{align*}$$

That condition seems to be fulfilled. Now for the opposite, what happens if we predict only zeroes, i.e. $$P=0$$? Then we should get the maximal possible penalty in the gradient. A quick calculation shows: 

$$\begin{align*} 
 \frac{\partial l_D(0,G)}{\partial p_{xy}} &= \frac{ 2g_{xy} }{(\|0\|_2^2 + \| G\|^2_2)} - \frac{ 0 \| 0 \cdot G \|_1}{(\|0\|_2^2 + \| G\|^2_2)^2}  \\ 
  &= \frac{ 2g_{xy} }{\|G\|_2^2}  

\end{align*}$$

Which is the maximum penalty possible under this loss function, any non-zero entry in $$P$$ would decrease the gradient term $$\frac{2g_{xy}}{(\|P\|_2^2 + \| G\|^2_2)}$$. Note that that for all pixels where the ground truth $$g_{xy}$$ is zero, the gradient will be zero too. We also see that the gradient is proportional to the number of pixels in class one:

- In the extreme case of only one pixel being of class one, the gradient will be $$1$$.
- In the other extreme case of every pixel being of class one, the gradient will be $$\frac{1}{nm}$$ if the input is of size $$n \times m$$.

Lets have a look at the $$3 \times 3$$ example we looked at earlier. Notice that both $$p_{22}$$ and $$p_{32}$$ are off by $$0.06$$ from their target values one and zero.

$$P = \begin{pmatrix}
 0.12 & 0.83 & 0.23 \\
 0.89 & \mathbf{0.94} & 0.36 \\
0.23 & \mathbf{0.06} & 0.18  \\
\end{pmatrix},

G = \begin{pmatrix}
 0 & 1 & 0 \\
 0 & \mathbf{1} & 1 \\
0 & \mathbf{0} & 0  \\
\end{pmatrix}$$

The gradient for $$p_{22}$$ with label $$g_{22} = 1$$ is

$$\begin{align*} 
 \frac{\partial l_D(P,G)}{\partial p_{22}} &=\frac{ 2 \cdot 1 }{\|P\|_2^2 + \| G\|^2_2} - \frac{4 \cdot 0.94 \cdot\| P \cdot G \|_1}{(\|P\|_2^2 + \| G\|^2_2)^2}   \\ 
  &= \frac{2}{5.665} - \frac{4 \cdot 0.94 \cdot 2.13}{5.665^2}  \\
 &= 0.353 - 0.25 \\
&=  0.103

\end{align*}$$

And the gradient for $$p_{32}$$ is with label $$g_{23}$$ is

$$\begin{align*} 
\frac{\partial l_D(P,G)}{\partial p_{32}} &= \frac{ 2 \cdot 0 }{\|P\|_2^2 + \| G\|^2_2} - \frac{4 \cdot 0.06 \cdot\| P \cdot G \|_1}{(\|P\|_2^2 + \| G\|^2_2)^2}  \\ 
  &=0 - \frac{4 \cdot 0.06 \cdot 2.13}{5.665^2}  \\
 &= - 0.0159 
\end{align*}$$

So we see that the penalty for the incorrect prediction of the class one is much larger. What would happen if $$p_{22}$$ was the only pixel with ground truth class one? Then the loss would become even bigger:

$$P^* = \begin{pmatrix}
 0.12 & 0.83 & 0.23 \\
 0.89 & \mathbf{0.94} & 0.36 \\
0.23 & 0.06 & 0.18  \\
\end{pmatrix},

G^* = \begin{pmatrix}
 0 & 0 & 0 \\
 0 & \mathbf{1} & 0 \\
0 & 0 & 0  \\
\end{pmatrix} \Rightarrow \| G \|_1 = \|G\|_2^2 = 1$$

Recalculating the gradient, we get

$$\begin{align*} 
 \frac{\partial l_D(P^*,G^*)}{\partial p_{22}} &=\frac{ 2 \cdot 1 }{\|P\|_2^2 + \| G\|^2_2} - \frac{4 \cdot 0.94 \cdot\| P \cdot G \|_1}{(\|P\|_2^2 + \| G\|^2_2)^2}   \\ 
  &= \frac{2}{3.665} - \frac{4 \cdot 0.94 \cdot 0.94}{3.665^2}  \\
 &= 0.546 - 0.263 \\
&=  0.283

\end{align*}$$

This is due to the class balancing effect of the Dice score, which tries to give equal importance to both no matter how often they appear.

If we were to use the Dice loss with gradient decent we would have to multiply it the results by minus one and wrap the entire loss in a log function to counteract the gradient shrinking properties of the soft-max layer.

<u>Question</u>: How would you modify $$P$$ and $$G$$ so that $$\frac{\partial l_D(P,G)}{\partial p_{22}} = - \frac{\partial l_D(P,G)}{\partial p_{23}}$$? 

<u>Question</u>: How would you extend this loss to multiple classes?

## Two classes - cross-entropy

While the Dice loss is a nice construct and has useful properties, in reality we almost always use cross-entropy as loss function for semantic segmentation task. Also because the extension to multiple classes is straightforward. The cross-entropy can be applied pixel wise and summed up to create a loss function over an entire prediction map.

### Binary cross-entropy

Previous posts go into much more detail about [why cross-entropy is a good loss function for classification](https://heinzermch.github.io/posts/on-loss-functions-part-1/) and [how to derive the gradient and implement it in NumPy](https://heinzermch.github.io/posts/creating-a-NN-from-scratch-part-1/). Here we simply restate the results from these posts and look at the outcome if we apply it to semantic segmentation. The binary cross-entropy loss for a two class problem with predicted probability $$p$$ and ground truth $$g$$ is:

$$l_{CE}(p, g) = -(g\log(p) + (1-g)\log(1-p))$$

When we apply this over the an entire image, the loss can be defined as a function

$$l_{CE} : \lbrace 0, 1 \rbrace^{n \times m}  \times [0,1]^{n \times m} \longrightarrow \mathbb{R}$$

where

$$l_{CE}(P, G) := - \sum^m_{j=1} \sum^n_{i=1}  g_{ij}\log(p_{ij}) + (1-g_{ij})\log(1-p_{ij})$$

The ground truth label $$g_{ij}$$ acts as indicator function to switch to the term which is relevant. The gradient of prediction $$p_{xy}$$ for $$1 \leq x \leq n$$ and $$1 \leq y leq m$$ is

$$ = $$

$$\begin{align*} 
 \frac{\partial l_D(P,G)}{\partial p_{xy}}&= -\sum^m_{j=1} \sum^n_{i=1}  \frac{\partial}{\partial p_{xy}}g_{ij}\log(p_{ij}) + \frac{\partial}{\partial p_{xy}}(1-g_{ij})\log(1-p_{ij})   \\ 
  &= -g_{xy}\frac{\partial}{\partial p_{xy}}\log(p_{xy}) + (1-g_{xy})\frac{\partial}{\partial p_{xy}}\log(1-p_{xy})    \\
 &= -\frac{g_{xy}}{p_{xy}} - \frac{1-g_{xy}}{1-p_{xy}} \\
&=  -\mathbb{1}_{g_{xy} = 1}\frac{1}{p_{xy}} + \mathbb{1}_{g_{xy}=0}\frac{1}{1-p_{xy}}

\end{align*}$$



### Weighted Cross-Entropy

Have a weight map $$w(x)$$ for each pixel. Use $$\Omega \in \mathbb{Z}^2$$ to index the weight map.

$$ L = \sum_{x \in \Omega} w(x) log(p(x))$$

See U-Net: Convolutional Networks for Biomedical Image Segmentation paper



# Conclusion

In the first part of this post we saw different notions of distance, for points in space and probability distributions. Later we saw what connects  KL-divergence, entropy and cross-entropy. The second part focused on what makes cross-entropy well suited for classification in combination with gradient descent.

## References

A list of resources used to write this post, also useful for further reading:

- [Deep Learning](https://www.deeplearningbook.org/) Book by Goods losses
- [Pythagorean Means](https://en.wikipedia.org/wiki/Pythagorean_means) Wikipedia
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) paper 
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) paper 
- 
- [Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) Paper that introduced the DICE loss for semantic segmentation
- [Blog post](https://www.jeremyjordan.me/semantic-segmentation/) for an overview of semantic segmentation
- 



## Comments

I would be happy to hear about any mistakes or inconsistencies in my post. Other suggestions are of course also welcome. You can either write a comment below or find my email address on the [about page](https://heinzermch.github.io/about/).