---
layout: post
author: Michael Heinzer
title:  "Testing if a sequence is random"
description: None yet
date:   2020-05-04 11:00:00 +0530
categories: DeepLearning AutoEncoders MultivariateNormal Bayes GenerativeModeling
comments: no
published: False
---
Dispute if sequence is random, how can we test that? By using smirnov kolmogorov

## Basic concepts and notation

Before we start, let

# How to compare distributions

Let us first have a look $$P(X=x) = \int P(X=x \mid \theta, z) P(Z=z) dz = \int P(X=x \mid f_{\theta}(z)) P(Z=z) dz \qquad \forall x \in X_T$$

# How to do it for this example

llowing code snippets we will assume that numpy as imported as `np`.

```python
import collections

def kolmogorov_smirnov_statistic_with_uniform(seq: list, min_val: int, max_val: int):
    n = len(seq)
    counter = collections.Counter(seq)
    empirical_distribution = collections.defaultdict(int)
    theoretical_distribution = collections.defaultdict(float)
    cumsum, d_n = 0.0, 0.0
    values = max_val + 1 - min_val
    for i in range(min_val, max_val + 1):
        if i in counter:
            cumsum += counter[i]
        empirical_distribution[i] = cumsum / n
        theoretical_distribution[i] = theoretical_distribution[i - 1] + 1.0 / values
        d_n = max(d_n, abs(empirical_distribution[i] - theoretical_distribution[i]))
    return d_n, n

print(kolmogorov_smirnov_statistic_with_uniform(
    [3, 6, 3, 4, 4, 1, 3, 3, 1, 1, 3, 1, 3, 6, 4, 6, 5, 2, 3, 3, 3, 5, 6, 2, 4, 1, 1, 2, 1, 5, 3, 2, 3, 4, 4, 6, 5, 1,
     5, 4, 5, 6], 1, 6))
[4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 1, 4, 4, 4, 4, 4, 3, 6, 3]
[3, 6, 3, 4, 4, 1, 3, 3, 1, 1, 3]

```



Values found [here](http://www.real-statistics.com/statistics-tables/kolmogorov-smirnov-table/)

 

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