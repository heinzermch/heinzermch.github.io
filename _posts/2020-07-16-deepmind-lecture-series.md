---
layout: post
author: Michael Heinzer
title:  "DeepMind lecture series"
description: Taking notes and summarizing
date:   2020-06-31 11:00:00 +0530
categories: None
comments: no
published: no
---
I was watching this DeepMind lecture series. Would like to see more about Deep RL, Graph

## Episode 1 - Introduction to Machine Learning and AI

Overview: Not technical at all, few formulas

Highlights: Definition of intelligence, T-SNE view of how policy functions cluster

Gives three examples, AlphaGo and AlphaZero, CTF and Folding. 

On limits of deep learning

- Lack of data efficiency
- Energy consumption of by computational systems
- Common sense, adapt quickly to new situations in environment

Autonomous driving AI complete? Need maybe physical simulations and multi-agent systems.

Follow up: T-SNE, Netflix Documentary about AlphaGo, papers

## Episode 2 - Neural Networks Foundations

Overview: Grow in Compute for parallel computation (matrix multiplication), data and scale of models with data. Deep Learning as blocks which can be arranged, not easily defined. Or rather Deep Learning foundations

One neuron is a projection on one line (see slide). Affine transformation. Optimized matrix multiplication goes down from $$n^2$$ to 2.7???. Linear means affine, parameters are weigths, neurons are units. Gradient magnitude as amount of information that flows through model. Cross-Entropy is also called negative log likelihood or logistic loss. Losses are additive with respect to samples. Softmax does not scale well with number of classes. XOR problem, can not separate with one line. Playground.tenserflow.org for examples of how to make problems linearly separable, play around (great for building intuition). Constructive vs. existential proof in mathematics (Universal Approximation Theorem). Three bumps by 6 neurons to make an approximation. ReLU needs careful initialization due to dead neurons (debug metric). Neural networks as computational graphs. Gradient and Jacobian. Linear Layer as computational graph, forward and backward pass. Backwards pass of max in computational graph. If weights are small, functions can not be too complex. Blogpost of andrey karpathy on diagnosing and debugging. Always over-fit first. Monitor norms of weights? Add shape asserts, because of broadcasting in modern framework. 

Highlights: Highly dimensional spaces are surprisingly easy to shatter with hyperplanes. XOR example with two layers, how it makes the problem linearly separable (sigmoid bends or squishes). Universal Approximation Theorem, size can grow exponentially (universal approx. theorem does not hold for ReLU???), difference between approximation and representation (we can approximate everything, but possibly at cost of huge number of parameters). Number of linear regions grows exponentially with dept, and polynomially with width (go deeper than wider!), ReLU can be seen as folding space on top of each other (read paper). Double descent, over-parameterization does not make things worse anymore (over-fitting), mapped onto Gaussian processes, model complexity is not as simple as number of parameters, holds for deep big models. MLPs can not represent multiplication, only approximate it, multiplicative units in network??

Do not look for marginally improvements (activations functions), but identify what neural networks can not do and propose a module that can do that.

Follow up: 

## Episode 3 - 

## Episode 4 - 