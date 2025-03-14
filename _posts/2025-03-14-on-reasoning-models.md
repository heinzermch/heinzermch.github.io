---
layout: post
author: Michael Heinzer
title:  "Reasoning Models"
description: On LLMs that "think"
date:   2025-02-23 18:00:00 +0530
categories: LLMs Reasoning RL
comments: yes
published: false
---

Hot topic of 2025. No papers, too proprietary. DeepSeek

## Basic Concepts and Notation

Before we start, let us quickly repeat some basic concepts and their notation. Readers familiar with the topic may skip this section.

- **Sigmoid**: $$ \sigma : (-\infty,\infty) \longrightarrow (0,1)$$, is defined as $$ \sigma(x) := \frac{1}{1+\exp(-x)} = \frac{\exp(x)}{1+\exp(x)}$$. It has the property that it maps any value to the open interval $$(0,1)$$, which is very useful if we want to extract a probability from a model.
  - The plot looks as follows:
    ![Plot of sigmoid(x)](/assets/images/loss_functions_part_1/sigmoid_plot.png)
  - Again three cases worth noting:
    - The left limit is 0: $$ \sigma(x)_{x \longrightarrow -\infty } = 0 $$
    - The right limit is 1: $$ \sigma(x)_{x \longrightarrow \infty } = 1 $$
    - At zero we are in the middle of the limits: $$ \sigma(0) = 0.5$$
  - The derivative of the sigmoid is $$\frac{\partial \sigma(x)}{\partial x} =  \sigma(x) (1-\sigma(x))$$
  
- **Dot product**: The dot product between two vectors $$ \mathbf{a}, \mathbf{b} \in \mathbb{R}^n$$ is defined as

  $$\mathbf{a} \cdot \mathbf{b} := \sum_{i=1}^n a_i b_i $$
  
The dot product for non-zero vectors will be zero if the they are in a 90 degree angle.

- **Cosine similarity**: The cosine similarity is the normalized dot product between to vectors $$ \mathbf{a}, \mathbf{b} \in \mathbb{R}^n$$:

$$ \cos(\theta) = \frac{a \cdot b}{\mid \mid a \mid \mid \cdot \mid \mid b \mid \mid}$$

It measures the angle between two vectors and is independent of their magnitude.

- **Softmax**: The basic softmax function is defined as:

$$ \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}$$

here $$z$$ is a vector of scores and $$N$$ is the number of elements in $$z$$.

- **Softmax trick**: Large values for $$z$$, $$e^{z_i}$$ can lead to numerical instability (overflow). To address this, we use the "softmax trick":

$$\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{N} e^{z_j - \max(z)}}$$

where $$\max(z)$$ is the maximum value in the vector $$z$$. This works because

Let $$c = \max(z)$$. Then:

$$\begin{align*}
\text{softmax}(z)_i &= \frac{e^{z_i - c}}{\sum_{j=1}^{N} e^{z_j - c}} \\
&= \frac{e^{z_i} e^{-c}}{\sum_{j=1}^{N} e^{z_j} e^{-c}} \\
&= \frac{e^{-c} e^{z_i}}{e^{-c} \sum_{j=1}^{N} e^{z_j}} \\
&= \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}
\end{align*}$$

- **Entropy**: The self-information of an event $$x \in X$$ in a probability distribution $$p$$ is defined as

$$I(x) = - \log(p(x))$$

For a probability $$p$$ distribution on a random variable $$X$$, the entropy $$H$$ of $$p$$ is defined as

$$H_b(p) := -\sum_{x \in X} p(x) \log_b(p(x)) = E(I(X))$$

where $$b$$ is the base of the logarithm, it is used to choose the unit of information.

- **Kullback-Leibler Divergence**: The KL-divergence of two probability distributions $$p$$ and $$q$$ is defined as

$$D_{KL}(p \mid\mid q) := \sum_{x \in X} p(x) \log\bigg(\frac{p(x)}{q(x)}\bigg) = - \sum_{x \in X} p(x) \log\bigg(\frac{q(x)}{p(x)}\bigg)$$

Often we consider $$p$$ to be the true distribution and $$q$$ the approximation or model output. Then the KL-divergence would give us a measure of how much information is lost when we approximate $$p$$ with $$q$$. If the KL divergence is zero, then the two distributions are equal.



- **Cross-Entropy**: The cross-entropy between two probability distributions $$p$$ an $$q$$ is defined as

$$H(p,q) := H(p) + D_{KL}( p \mid \mid q) $$

Cross-entropy is the sum of the entropy of the target variable and the penalty which we incur by approximating the true distribution $$p$$ with the distribution $$q$$. The terms can be simplified:

$$\begin{align*} 
 H(p,q) &=  H(p) + D_{KL}( p \mid \mid q) \\ 
  &= -\sum_{x \in X} p(x) \log(p(x)) - \sum_{x \in X} p(x) \log\bigg(\frac{q(x)}{p(x)}\bigg) \\
 &=  -\sum_{x \in X} p(x) \log(p(x))  - \sum_{x \in X} p(x) \log(q(x))  + \sum_{x \in X} p(x)\log(p(x)) \\

&= - \sum_{x \in X} p(x) \log(q(x))
\end{align*}$$

Only the KL-divergence depends on $$q$$, thus minimizing cross-entropy with respect to $$q$$ is equivalent to minimizing the KL-divergence.

- **Symmetric vs Asymmetric Cross-Entropy loss**: Since the KL-divergence is asymmetric, the CE-loss is also asymetric. This means the penalty for false positive is different from the penalty for a false negative. It is more sensitive to errors predicting the true class. This can be problematic when dealing with noisy labels, the model may overfit to incorrect information. 

The standard cross-entropy loss is given by:

$$L_{CE}(y, \hat{y}) = - \sum_{i=1}^{C} y_i \log(\hat{y}_i) $$

To create a symmetric cross-entropy loss, we can combine the standard cross-entropy with its reverse:

$$L_{RCE}(\hat{y}, y) = - \sum_{i=1}^{C} \hat{y}_i \log(y_i)$$

The symmetric cross-entropy loss, $$L_{SCE}$$, can be defined as a combination of these two losses. A simple combination is the average:

$$ L_{SCE}(y, \hat{y}) = \frac{1}{2} \left[ L_{CE}(y, \hat{y}) + L_{RCE}(\hat{y}, y) \right] $$

Substituting the definitions of $$L_{CE}$$ and $$L_{RCE}$$, we get:

$$ L_{SCE}(y, \hat{y}) = -\frac{1}{2} \sum_{i=1}^{C} \left[ y_i \log(\hat{y}_i) + \hat{y}_i \log(y_i) \right]$$

- **InfoNCE loss**: Introduced by van der Oord et al in their paper [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748). We want to maximize the mutual information between two original signals $$x$$ and $$c$$ defined as


$$ I(x, c) = \sum_{x, c} p(x, c) \log \frac{p(x \mid c)}{p(x)}$$

We want to model the density ratio of the signals as

$$f(x_t, c_t) \propto \frac{p(x_t \mid c_t)}{p(x_t)}$$

where $$f$$ is a model that is proportional to the true density, but does not have to integrate to 1. Given a set $$X = \lbrace x_1, \dotsc, x_N \rbrace$$ of $$N$$ random samples containing one positive sample from $$p(x_t \mid c_t)$$ and $$N − 1$$ negative samples from the ’proposal’ distribution $$p(x_t)$$, we optimize

$$L_{RCE}(x, c) = - E_X \bigg \lbrace \log \frac{f(x_t, c_t)}{\sum_{x_j \in X} f(x_j, c_t)} \bigg \rbrace $$

This loss will result in $$f(x_t, c_t)$$ estimating the density ratio in the previous equation. For a proof see the [paper](https://arxiv.org/abs/1807.03748). In other words, the InfoNCE loss encourages similar items to have similar embeddings and disimilar items to have different embeddings.



# What

Poker part, just scaling up, doing self play. Humans would talk long instead of act instantly for difficult problems.

Importance of search in poker. It would think for 30s towards the end of the game? Having think for 30s is same as scaling up 100000x.

2017 brains vs pokers won by 15bb/100 compared lost to 9bb/100 loss.

Why wasn't search/planning considered in poker before? This is extra compute at test/inference time. Scaling test time compute makes inference/experiments much more expensive. Painful to run this. Incentives: it was all about winning the annual competition at 2 CPU cores for 2 seconds. Didn't think about beating the best humans

People underestimated the impact of search, thought it would be 10x not 100'000x. Similar results in AlphaGo. Which used MCTS. Raw NN performance is below human performance, only becomes superhuman when you increase test time compute. How much would you need to scale up to get raw NN score to go to superhuman. Here again if you want to go from 3000 to 5200 elo you need 100'000x compute scale.

All of this applies to state of AI as well. We do pre-training for 100M+ and doing inference is costing pennies.


Is there a way to scale inference compute in LLMs?

Consensus is simplest way to scale up compute. Minerva paper. Get lift by sampling 10x or 100x but don't get much more afterwards for majority voting. Not great retunrs for scaling inference compute. But there is often at least one answer that is correct.

OpenAI o1 compute scales differently, it goes from 20% to 80% in pass@1 accuarcy at AIME.

It works with CoT RL compute at it, so there is benefits from spending more compute on training and test time. Effective way to scale inference compute.


There are gains in O1 from many different areas. No big boost for english and literature. But more than STEM improves.

See research blog post from OpenAI "learning to reason with LLMs".



Prompt with CoT, this increases quality results from prompt.

Optimize chain of thought, generate large scale CoT Wei et al NeuriPS 2022



Why does o1 work at all?
Generator-Verifier gap. Easier to verify than generate a solution.
When a generator-verification gap exists, we can spend more compute at infrence time to achieve better performance!

Models can do verification on their own? Know when they are making or not making progress??? (Seems vague).


Pre-training cost and data have become bottlenecks, not so for o1. Because we can scale up inference compute. And we have a lot of room to scale up inference compute. 

Concern: Will increase cost and waiting time of queries. What inference time cost are we willing to pay for difficult problems like life-saving drugs?

AI can be much more than chatbots.

The bitter lesson by richard sutton (its a book/blog post?? TODO read). Two things that seem to scale arbitrarily well are search and learning.

He says: Don't try to edge out current state of the art but try to think of techniques that scale with compute.

He says: Pre-training limitations are no longer a blocker, but we don't know what other blockers exist or how far we can scale this.

Can we tell the model to tell how long to think? Can we make the model judge how long it should think about a specific task?


## Using recurrence to achieve weak to strong generalization
Youtube talk

Train on 9x9 maze and solve 800x800 maze. Works with 20000 iterations instead of 30 during train tim ewith RNN

Test time computation is for weak to strong generalization.

Having skip connections helps not forget. Using RNNs

Apply this to chess, 

Sudoku is solved too.

But here we generalize in the same class of problems mostly. Easier task in same class chess/sudoku into harder task. But no generalization.

Transformers need to train with many positional embeddins. Take into account least significant at first because transformers are causal. So for 123 write 321

Training: Backward pass, do a progressive loss. Gets compute cost down for training recurrent transformer loss. Also called truncated backprop.

Testing time compute is actually the ovearching topic here?

He is not a huge fan of CoT (why?). CoT needs human generated data to see recurrence (this seems not obviously true anymore after DeepSeek).

Why doesn't it fully work? Positional encoding is not precise enough. Abacus embeddings solve this issue to see what number belongs to what number in the sequence.

[Video link for the talk](https://www.youtube.com/live/M7Kq0ooFFco).

Easy to hard vs weak to strong generalization. Easy to hard is generalization outside of current problem space.

Boosting is in a weaker category than recurrent computation, or even CoT.

Recurrent models vs diffusion models. Diffusion models are trained to solve problems in one step, not like these models that are trained to do multi step problem solving.








# Conclusion

AAA

## Final Thoughts

Some follow up questions and remarks for my future self or experts in this field:

* Similarly to LLMs, we seem to run out of training data as WebLI represents the entire (high quality) internet data, what else can be done? Generate more data with GenAI and use captioning for training smaller models?
* Are there any tasks that are not covered yet with the current training approaches?
* Could we use videos as another data source for further advances?
* A very 2025 take, can we apply RL to improve vision further? Explored to some extend already in 2023 by the [Big Vision team](https://x.com/giffmana/status/1626695378362945541).


## References

A list of resources used to write this post, also useful for further reading:


- [Learning to Reason with LLMs](https://www.youtube.com/live/Gr_eYXdHFis) talk of Noam Brown at Simons Institute
- [Title](link) for XYZ



## Comments

I would be happy to hear about any mistakes or inconsistencies in my post. Other suggestions are of course also welcome. You can either write a comment below or find my email address on the [about page](https://heinzermch.github.io/about/).



