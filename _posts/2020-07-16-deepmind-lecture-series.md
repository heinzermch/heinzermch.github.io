---
layout: post
author: Michael Heinzer
title:  "DeepMind lecture series"
description: Taking notes and summarizing
date:   2020-07-15 11:00:00 +0530
categories: None
comments: no
published: no
---
I was watching this DeepMind lecture series. Would like to see more about Deep RL, Graph

## Episode 1 - Introduction to Machine Learning and AI

Overview: Not technical at all, few formulas

### First part, what is intelligence?

Somewhat mathematical definition

$$\Upsilon(\pi) := \sum_{\mu \in E} 2^{-K(\mu)}V^{\pi}_{\mu}$$

Why choosing games to solve intelligence? Framework of reinforcement learning, why use Deep Learning on top. 

### Second part: AlphaGo and AlphaZero. 

Having a policy and value network. Innovation of having self-play to improve, and generate training data. Interesting slide on amount of search per decision, chess engine vs. AlphaZero vs. Humans. Rediscovering human plays.

### Third part: Learning to Play with Capture the flag

Multi-Agent playing, more complicated rewards. Can procedurally generate environments. Train a population of agents?? Each agent trains individually. Sounds kind of like evolutionary algorithms. Interesting that Humans can only win against agents if playing with an agent, very collaborative.

### Fourth part: Folding Proteins with AlphaFold

Main issue is how to map the problem into a structure that allows for the application of deep learning. Levinthal's paradox, how does nature solve the problem? Deep learning applies because we have samples for 150,000 proteins from experiments, but data is difficult to get. Impressive that it can be applied at all, usually much more data is required, for example speech and vision. Can take a full PhD thesis to sequence a single protein. Problem can be mapped onto an image

### Highlights: 

Definition of intelligence, T-SNE view of how policy functions cluster

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

### Highlights

- Approximation is not the same as representation

## Episode 3 - 

### 01 - Background

Recap from last lecture, building blocks. How to feed an image, its a 2D grid, special topological structure. NN expects vectors of numbers, we can flatten the image into a vector. Problems when shifting the input, it will look very different to a network. Want the network to take into account the grid structure. Two key features:

- Locality: nearby pixels are strongly correlated
- Translation invariance: patterns can occur anywhere in the image

Images have simliar modality, sounds can occur anytime in signal is also translation invariance in time. Textual data, words can occur anytime. Graph structure, molecules can exhibits patterns anywhere

How do we take advantage of topological structure

- Weight sharing: use same parameters to detect paterns all over the image
- Hierarchy: low-level features are composed, model can be stacked to detect these

Data drives research, here ImageNet has 1000 classes, but lots of them are dog breeds, approximately 100. History of the ImageNet competition

### 02 - Building Blocks

From fully connected to locally connected using 3x3 receptive field. From locally connected to convolutional by sharing weights. Operation becomes equivariant to translation. Weights are called kernel or filter. Filter operation essentially with learned weights. Multiple feature maps are channels. Variants of the convolution operation:

- Valid convolution: no padding, slightly smaller output
- Full convolution: padding so that only one pixel overlaps in minimum, larger output than input
- Same convolution: padding so that output has same size as input, works better for odd kernel size
- Strided convolution: step size > 1, makes output resolution by at least 2x smaller
- Dilated convolution: kernel is spread out, increases receptive field, step size > 1 inside kernel. Can be implemented efficiently
- Depthwise convolution: Normally each input channel is connected to each output channel, here every input channel is connected to only one output channel
- Pooling: compute mean or max over small windows

### 03 - Convolutional neural networks

Simplified computational graph with convolutions, non-linearity, pooling and fully connected layers

### 04 - Going Deeper: case studies

Different kind of nets. Challenges of deep networks, computational complexity and optimisation difficulites. What can help: initialization, sophistiacated optimizers, normalisiation layers, network design.

- LeNet-5 (1998): 5 layer, sigmoid
- AlexNet (2012) : 8 layers, ReLU, start with 11x11 kernel, not every convolution needs to be followed by pooling
- VGGNet (2014) : up to 19 layers, use same layers to avoid reduction, only uses 3x3 kernels and stack them instead of larger kernels, this also increases receptive fields with fewer parameters and more flexible because extra non-linearity. Use data and not model parallelism, same model on 4 gpus and split data into four. error plateaus at 16 layers, 19 layers was worse
- GoogLeNet (2014): Branch out and have multiple feature arms in each block (inception module)
- Idea of batch normalization: reduce sensitivity to initialization, acts as a regularizer, more robust to different learning rates, introduces stochasticity (because mu and sigma have to be estimated for each batch, this can make model more robust. at test time this introduces dependency on other images in batch. Freeze them for test time, can be a source for a lot of bugs), speed up training
- ResNet (2015): (150 layers deep) introduce residual connections, makes training deeper networks simpler. V2 avoid nonlinearites in residual pathway. Bottleneck block, 1x1 reduces channels, 3x3 on fewer feature maps (fewer parameters), go back with 1x1 to more channels (ResNet-152 is actually cheaper than VGG)
- DenseNet (2016): connections to all previous layers, not one.
- Squeeze and excitation networks (2017): features incorporate global context
- AmoebaNet (2018): Architecture found by neural architecture search, evolutionary algorithm. DAG composed of predefined layers
- Reduce complexity: 
  - Depthwise convolutions, 
  - separable convolutions, 
  - inverted bottlenecks (MobileNetV2, MNasNet, EfficientNet)

### 05 - Advanced topics

Data augmentation, makes them robust against other transformations: rotation, scaling, shearing, warping.

Visualize what convnet learns, maximize one activation using gradient ascent. Even with respect to a specific output neuron (class). Nice article on distill.pub, Other topics: pre-training, fine-tuning. Group equivaraint convnets: invariance to e.g. rotation and scale. Recurrence and attention

### 06 - Beyond image recognition

Other tasks: Object detection, semantic segmentation, instance segmentation. Generative models: GANs, Variational autoencoder, autoregressive models (PixelCNN). Representation learning, self-supervised learning. Audio, video, text, graphs.

Prioer knowledge is not obsolete: it is merely incorporated at a higher level of abstraction

### Highlights

- BN introduces stochasistiy (see above)

## Episode 4 - Advanced models for Computer Vision

Holy grail: Human level scene understanding

### 01 - Supervised image ~~classification~~ - Tasks beyond classification

#### Task 1 - Object Detection

Output is class label, one hot encoded, and a bounding box with (x_c, y_c, h, w) for every object in the scene. Data is not ordered, do regression on coordinates. Generally minimize quadratic loss for regression task $$l(x,y) = \| x-y\|^2_2$$.

How to deal with multiple targets? Redefine problem, first classification and then regression by discretising the output values in one hot label. Case studies:

**Faster R-CNN, two-stage detector**

Discretise bbox space, anchor point for (x_c, y_c), scales and ratios for (h,w). n candidates per anchor, predict objectness score for each box, sort and keep top K.  Good accuracy at 5fps. Non-differentiable operations in there, can not backprop gradient w.r.t. bounding box because we fix them in advance (see spatial transformer Networks for differential operation)?? Train system in two parts, objectiveness labels for lower part. Afterwards train second part on pre-trained network. That's why it is called a two-stage detector

**RetinaNet - one-stage detector**

Feature pyramid network. 4 is four coordinates of bounding boxes, K is number of classes, A is the number of anchors, 4A is for coordinates of anchor boxes. Can not train straight away, because too many easy negatives. The accumulated loss of many easy examples overwhelms the loss of rare useful examples. Generally one-stage detector employ hard negative mining (in depth description), but RetinaNet uses Focal Loss. Has good accuracy at 8fps. New state of the art for object detection.

#### Task 2 - Semantic Segmentation

Assign a class label to every pixel. Problem: how to generate an output at the same resolution as input. Previous examples all have sparse output, this one needs dense output. Pooling creates larger receptive field, but we lose spatial information, hence we need a reverse operation. For example unpooling. Other upsampling methods, deconvolutions

**U-Net**

Encoder-Decoder model, skip connections to preserve high frequency details. Simliar to ResNet classifier, makes backpropagation of gradients easier. We train it with pixel-wise cross entropy loss

Connection between RetinaNet and U-Net, same U shape.

#### Task 3 - Instance Segmentation

Combine object detection and semantic segmentation. Instance segmentation allows you to separate overlapping classes

#### Metrics and benchmarks

Classification was easy: Accuracy as percentage of predictions

Object detection and segmentation: Intersection over Union (non-differentiable, so it's only for evaluation). Pixel wise accuracy would be foolish to use. Not recommended in general to use different measures between training and testing. (IoU not differentiable because of max operations)

Benchmarks, cityscapes for semantic segmentation and COCO

#### Tricks of the trade

Transfer learning. Nice technical definition, reuse knowledge learnt by $$f_S$$ in $$f_T$$. Intuition is features are shared across tasks and dataset. Don't start from scratch all the time. Reuse knowledge across tasks or across data.

- Transfer learning across different tasks. Mostly remove the last layers and add new layers to adapt to the new task. See Taskonomy paper, for an overview of different tasks and how they are related.
- Transfer learning across different domains. Train in simulation, use tricks to adapt to target domain (for example domain randomization which is data augmentation and hard negative mining to identify the most useful augmentations)

### 02 - Supervised ~~image~~ classification - Beyond single image input

Experiment with people who recovered from blindness from surgery to see which tasks are hard, recovering objects from a scene. However same type of images with moving images allows them to do much better. Motion really helps with object recognition when learning to see. Conclusion, should use videos  for training object recognition during learning using translation, scale, etc.

#### Input - Pairs of images

Optical flow estimation. Input are two images, for each pixel in image one, where did it end up in image two. Output is also dense image map

**FlowNet**

Encoder-Decoder architecture like U-Net, fully supervised with euclidean distance loss. Invented Flying chairs dataset with sim to real to learn about motion. Essence is that pixels that move together belong to the same object.

#### Input- Videos

Base case: use semantic segmentation video and apply it frame wise (but don't use temporal information), this leads to flickering in a video. We could use 3D convolutions by stacking 2D frames to get volumes, kernels are 3D objects. In 3D convolutions the kernel moves in both space and time to create spatio-temporal feature maps (we can re-use strided, dilated, padded properties). 3D convolutions are non-causal, because you take into account frames from the future (times $$t-1, t, t+1$$), which is fine for off-line processing, but what applying it in real time? Can use masked 3D convolutions.

Can do action recognition, video as input, targets are a an action label which is one-hot encoded.

**Case study: SlowFast**

Two branch model, high frame right and low frame rate processing. Inspiration from human visual system, which also has two stream visual system. Take more features from lower frame rate stream and less features from high frame rate stream. Inution is high level features, abstract information for low frame rate, the other branch takes high frequency or tracking changes.

Transfer lerning can be used by inflating 2D image filter to 3D filters by replicating along time dimension.

#### Challenges in video processing

Difficult to obtain labels, large memory requirements, high latency, high energy consumption. Basically we need too much compute for it to be broadly applicable. Ongoing area of research of how to improve that by using parallelism and exploit redundancies in the visual data. One idea is to train a model to blink, (humans do it more often than necessary to clean the eye to reduce cognitive load)

### 03 - ~~Supervised~~ image classification - Beyond strong supervision

Labeling is tedious and a research topic on itself. Humans can only label keyframes and methods will propagate labels.

Self-Supervision - metric learning

Learn to predict distances between inputs in an embedding to give similarity measure between data. Create clusters with same person, for unseen data use nearest neighbor. 

**Contrastive loss**

$$l(r_0, r_1, y)$$  label is 1 if same person or 0 otherwise

Use margin to cap the maximal distance. See the comparison with euclidean distance. But it is hard to choose $$m$$, the margin. All classes will be clustered in a ball with radius $$m$$, which can be unstable.

**triplet loss**

Better than contrastive loss, relative distance are more meaninful than a fixed margin. $$(r_a, r_p, r_n)$$, attract and reject. Need hard negative mining to select informative triplets.

#### State-of-the-art

Same data, but different augmentations. Apply many different augmentations. Achieves comparable to supervised results (with more parameters).



### 04 - Open Questions

Is vision solved? What does it mean to solve vision? Need the right benchmarks

How to scale systems up? Need a system that can do all the tasks at once, more common sense. Different kind of hardware.

What are good visiual representations for action? Keypoints could help

### Conclusion

Need to rethink vision models from the perspective of moving pictures with end goal in mind.

### Highlights

- Finally a good explanation of the difference between two-stage and one-stage detection (try googling the difference), 
- Connection between object detection and semantic segmentation via U-Net structure.
- Motion helps when learning to see
- Non-causality of 3D convolutions

## Episode 5 - Optimization for

## Episode 6 - Sequences and

## Episode 7 - Deep Learning for

## Episode 8 - Attention and

## Episode 9 - Generative

## Episode 10 - Unsupervised

## Episode 11 - Modern Latent

## Episode 12 - Responsible