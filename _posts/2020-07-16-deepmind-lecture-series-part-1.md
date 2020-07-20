---
layout: post
author: Michael Heinzer
title:  "DeepMind Lecture Series - Part I"
description: Notes from the first six lectures
date:   2020-07-15 11:00:00 +0530
categories: MachineLearning AI Convolutions Optimization Sequences NeuralNetworks ComputerVision
comments: no
published: no
---
I was watching this DeepMind lecture series. Would like to see more about Deep RL, Graph

## Episode 1 - Introduction to Machine Learning and AI



### 01 - Solving Intelligence

What is intelligence? Intelligence measures an agent’s ability to achieve goals in a wide range of environments. Or more technically we can define intelligence as

$$\Upsilon(\pi) := \sum_{\mu \in E} 2^{-K(\mu)}V^{\pi}_{\mu}$$

Where 

- $$\Upsilon(\pi)$$ is the universal intelligence of agent $$\pi$$.
- $$\sum_{\mu \in E}$$ is the sum over all environments $$E$$
- $$K(\mu)$$ the Kolmogorov complexity function, a complexity penalty. Note that $$2^{-K(\mu)}$$ becomes smaller for more complex environments, this measure gives more weight to simple tasks.
- $$V^{\pi}_{\mu}$$ is the value achieved in environment $$\mu$$ by agent $$\pi$$.

 For more details you can read the paper [here](https://arxiv.org/abs/0712.3329), the above formula can be found on page 23. This formula can be easily applied inside the reinforcement learning framework, where an agent interacts with the environment and receives rewards. The image below visualizes the concept in a very abstract way, the agent (left) interacts with the world (right) by giving it instructions (bottom) and receiving updates (reward, state).

![Abstract visualization of reinforcement learning](/assets/images/deepmind_lecture_part_1/e01_01_reinforcement_learning.png)

Why did DeepMind choose games to solve intelligence?

1. Microcosms of the real world
2. Stimulate intelligence by presenting a diverse set of challenges
3. Good to test in simulations. They are efficient and we can run thousands of them in parallel, they are faster than real time experiments
4. Progress and performance can be easily measured against humans.

There is another [talk](https://www.youtube.com/watch?v=3N9phq_yZP0&list=PLqYmG7hTraZC9yNDSlv0_1ctNaG1WKuIx) where Demis Hassabis explains in more detail the reasoning of why DeepMind chose to solve games first.

![Abstract visualization of reinforcement learning in games](/assets/images/deepmind_lecture_part_1/e01_01_reinforcement_learning_in_games.png)

 The above image visualizes the reinforcement learning feedback loop in games. As the image shows, large parts of the agent consist of neural networks, using deep learning. Why is Deep Learning used now?

- Deep Learning allows for end-to-end training
- No more explicit feature engineering for different tasks
- Weak priors can be integrated in the architecture of the network (such as in convolutions, recurrence)
- Recent advances in hardware and data
  - Bigger computational power (GPUs, TPUs)
  - New data source available (mobile devices, online services, crowdsourcing)

### 02 - AlphaGo and AlphaZero

AlphaGo bootstraps from human games in Go by learning a policy network from thousands of games. Once it has weak human level it stars learning from self-play to improve further.

![Abstract visualization of reinforcement learning in games](/assets/images/deepmind_lecture_part_1/e01_02_alpha_go.png)

The policy network makes the complexity of the game managable, you can do rollouts by selecting only promising plays. The value network allows to reduce the depth of rollouts by evaluating the state of the game at an intermediate stage.

![Exhaustive search visualization](/assets/images/deepmind_lecture_part_1/e01_02_exhaustive_search_tree.png)

AlphaGoZero: Learns from first principles, starts with random play and just improves with self-play. Zero refers to zero human domain specific knowledge (in Go)

AlphaZero: Plays any 2-player perfect information game from scratch. It was also tested on Chess and Shogi, which is Japanese chess. Achieves human or superhuman performance on all of them. The way this works is to train a policy and value network, and then play against for a hundred thousand plays. Then a copy of the two networks learns from that generated experience, once the new copy wins 55% of the games against the old version, the data generation process is restarted.

![AlphaZero visualization](/assets/images/deepmind_lecture_part_1/e01_02_alpha_zero.png)

An interesting part of the lecture is that the amount of search an agent does before making a decision in chess. The best humans look at 100s of positions, AlphaZero looks at 10'000s of positions and a state-of-the art Chess engine looks at 10'000'000 of positions. This means the network allow for better data efficiency by selecting only promising plays to explore. AlphaZero beats both, humans and state-of-the-art Chess engines. AlphaZero also rediscovers common human lays in chess, and discards some of them as inefficient.

![Amount of search per decision in Chess, Chess engine vs. AlphaZero vs. Human Grandmaster](/assets/images/deepmind_lecture_part_1/e01_02_amount_of_search_per_decision.png)

Conclusions about AlphaZero:

- Deep Learning allows to narrow down the search space of complex games
- Self-Play allows for production of large amounts of data which is necessary to train deep networks
- Self-Play provides a curriculum of steadily improving opponents
- AlphaZero is able to discover new knowledge and plays in the search space

### 03 - Learning to Play Capture the Flag

Capture the flag is a multi-agent game which is based on Quake III arena. Agents play in teams of two against each other and need to coordinate. The maps are procedurally generated, in indoor and outdoor style. A team can only score if their flag has not been captured by another team, this leads to a complex reward situation.

![Capture the flag environment, four agents per map](/assets/images/deepmind_lecture_part_1/e01_03_capture_the_flag_overview.png)

Here an entire population of agents are trained, a game is played by bringing multiple agents together, each will only have access to his individual experience. There is no global information as in the previous cases. The population of agents serve two purposes:

- Diverse teammates and opponents, naive self-play leads to a collapse
- Provides meta-optimization of agents, model selection, hyper-parameter adaption and internal reward evolution

![Capture the flag training procedure](/assets/images/deepmind_lecture_part_1/e01_03_capture_the_flag_training_procedure.png)

Achieves superhuman performance and easily beats baseline agents. The agent was evaluated in a tournament playing against and with humans. Humans can only beat them when their teammate is an agent. Surprisingly did humans rate the FTW agent as the most collaborative. 

![Capture the flag training procedure](/assets/images/deepmind_lecture_part_1/e01_03_performance.png)

Agents develop behavior also found in humans, such as teammate following and base camping. Different situations of CTF can be visualized in the neural network by applying t-SNE to the activations.

![Visualization of the state of NN and CTF situations](/assets/images/deepmind_lecture_part_1/e01_03_clustering_neural_network_activity.png)

Conclusion:

- This shows that Deep Reinforcement learning can generalize to complex multi-player video games
- Populations of agents enable optimization and generalization
- Allows for understanding of agent behavior

### 04 - Beyond Games: AlphaFold

Protains are fundamental building blocks of life. They are the target of many drugs and are like little biomechanic machines. The shape of proteins allows to make deductions about their functions. The goal is to take as input a amino acid sequence and predict a 3D structure, which is a set of atom coordinates.

![From amino acid sequence to 3D structure](/assets/images/deepmind_lecture_part_1/e01_04_sequence_to_3D.png)

The problem is parametrized as follows:

- Predict the coordinates of every atom, especially the ones of the backbone
- Torsion angles $$(\Psi, \Phi)$$ for each residue are a complete parametrization of the backbone geometry
- There are $$2N$$ degrees of freedom for chains of length $$N$$

![Capture the flag training procedure](/assets/images/deepmind_lecture_part_1/e01_04_prediction_parameters.png)

Levinthal's Paradox says that despite the astronomical number of possible configurations, proteins fold reliable and quickly to their native state in nature. Finding the structure usually takes 4 years of human work, despite this there is a database of 150'000 entries available. This is not as much as in other domains (such as speech), but allows for Deep Learning to be applied.

The model takes a protein sequence as input to a neural network consisting of 220 deep dilated convolutional residual blocks, and predicts a map of pairwise distances plus angles. Then gradient descent is run on the resulting score function to obtain a configuration estimate.

![Structure of AlphaFold](/assets/images/deepmind_lecture_part_1/e01_04_alpha_fold_structure.png)

The whole system was evaluated on the CASP13 competition against other world class researcher on unseen data and achieved first place.

Conclusion:

- Deep Learning based distance prediction gives more accurate predictions of contact between residues ...
- ... but accuracy is still limited and only the backbone can be predicted. The side chains still rely on an external tool
- Work builds on decades of work by other researchers
- Deep Learning can deliver solutions to science and biology

### 05 - Overview of Lecture Series

This part gives a quick overview of what will be discussed in the lecture series

1. Introduction to Machine Learning and AI
2. Neural Networks Foundation
3. Convolutional Neural Networks for Image Recognition
4. Vision beyond ImageNet - Advanced models for Computer Vision
5. Optimization for Machine Learning
6. Sequences and Recurrent Networks
7. Deep Learning for Natural Language Processing
8. Attention and Memory in Deep Learning
9. Generative Latent Variable Models and Variational Inference
10. Frontiers in Deep Learning: Unsupervised Representation Learning
11. Generative Adversarial Networks
12. Responsible innovation

### Question session

What are the limits of Deep Learning:

- Lack of data efficiency
- Energy consumption of by computational systems
- Common sense, adapt quickly to new situations in environment

Is autonomous driving AI complete?

- Probably needs more than RL simulations only. Need maybe physical simulations and multi-agent systems.

### My Highlights: 

- Definition of intelligence
- T-SNE view of how policy functions cluster

## Episode 2 - Neural Networks Foundations

### 01 - Overview

Deep Learning has been made possible by advances in parallel computiation (important for matrix multiplication) and larger data sets. In general Deep Learning is not easy to define, an attempt from Yann LeCun:

#### "DL is constructing networks of parameterized functional modules & training them from examples using gradient-based optimization."

We can see Deep Learning as a collection of differentiable blocks which can be arranged to transform data to a specific target. 

![Seeing Deep Learning as a collection of differentiable blocks](/assets/images/deepmind_lecture_part_1/e02_01_deep_learning_puzzle.png)

### 02 - Neural Networks

An artificial neuron is losely inspired by real neurons in human brains. However the goal of an artificial neuron is to reflect some neurophysiological  observation, not to reproduce their dynamics. One neuron is described by the equation

$$\sum_{i=1}^d w_i x_i + b \qquad d \in \mathbb{N}, w_i, x_i, b \in \mathbb{R}$$

and can be seen as a projection on a line.

![Artificial neuron visualized](/assets/images/deepmind_lecture_part_1/e02_02_artificial_neuron.png)

A linear layer is a collection of artificial neurons. In machine learning linear really means affine. We also call neurons in a layer units. Parameters are often called weights. These layers can be efficiently parallelized and are easy to compose.

![Linear layer visualized](/assets/images/deepmind_lecture_part_1/e02_02_linear_layer.png)

In order to make them non-linear, they are combined with point wise sigmoid activation functions. We call them non-linearities, these are applied point-wise and produce probability estimates.

![Sigmoid activations visualized](/assets/images/deepmind_lecture_part_1/e02_02_sigmoid_activation_function.png)

As mentioned in the puzzle above, we also need a loss function. It is often called negative log likelihood or logistic loss.

![Cross-entropy loss visualized](/assets/images/deepmind_lecture_part_1/e02_02_cross_entropy_loss.png)

These components allow us to create a simple but numerically unstable neural classifier, because the gradient can vanish through some of these layers. We can see the gradient magnitude as amount of information that flows through a model.

![Cross-entropy loss visualized](/assets/images/deepmind_lecture_part_1/e02_02_neural_classifier.png)

The above model is equivalent to a logistic regression. It is also additive over samples, which allows for efficient learning .We can generalize the sigmoid to $$k$$ classes, this is called a softmax:

$$f_{sm}(\mathbf{x}) = \frac{e^{\mathbf{x}}}{\sum^k_{j=1} e^{\mathbf{x}_j}} \quad \mathbf{x} \in \mathbb{R}^k$$

This has the additional advantage that it is numerically stable when used in conjunction with a Cross-Entropy loss. This is equivalent to a multinomial logistic regression model. But softmax does not scale well with number of classes (if you have thousands of classes).

![Softmax combined with cross-entropy](/assets/images/deepmind_lecture_part_1/e02_02_softmax_cross_entropy.png)

Surprisingly often high dimensional spaces are easy to shatter with hyperplanes. For some time this was used in natural language processing under the name of MaxEnt (Maximum Entropy Classifier). However it can not solve some seemingly simple tasks as separating these two classes (or can you draw a line to separate them?)

![XOR problem for two classes](/assets/images/deepmind_lecture_part_1/e02_02_xor_limitation.png)

To solve the above problem we need a hidden layer which projects the four group of points into a space where they are linearly separable (top left of the image)

![XOR problem for two classes](/assets/images/deepmind_lecture_part_1/e02_02_xor_hidden_layer_transform.png)

The hidden layer allows us to bend and twist the input space and finally apply a linear model to do the classification. We can achieve this with just two hidden neurons. To play around with different problems and develop some intuition go to [playground.tensorflow.org](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.30401&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false). 

One of the most important theoretical results for neural networks it the **Universal Approximation Theorem**:

#### For any continuous function from a hypercube $$[0,1]^d$$ to real numbers, and every positive epsilon, there exists a **sigmoid** based, 1-hidden layer neural  network that obtains at most epsilon error in functional space.

This means a big enough network can approximate, but not necessarily directly represent any smooth function. Neural networks are very expressive! This theorem can be slightly generalized to:

#### For any continuous function from a hypercube $$[0,1]^d$$ to real numbers, **non-constant, bounded and continuous activation function f**, and every positive epsilon, there exists a 1-hidden layer neural network using **f** that obtains at most epsilon error in functional space.

However these theorems tell us nothing about how to learn or how fast we can learn those functions. The proofs are just existential, they don't tell us how to build a network that has those properties. Another problem is that the size of the networks grows exponentially.

### 03 - Learning



### 04 - Pieces of the Puzzle

### 05 - Practical Issues

### 06 - Bonus: Multiplicative Interactions



Constructive vs. existential proof in mathematics (Universal Approximation Theorem). Three bumps by 6 neurons to make an approximation. ReLU needs careful initialization due to dead neurons (debug metric). Neural networks as computational graphs. Gradient and Jacobian. Linear Layer as computational graph, forward and backward pass. Backwards pass of max in computational graph. If weights are small, functions can not be too complex. Blogpost of andrey karpathy on diagnosing and debugging. Always over-fit first. Monitor norms of weights? Add shape asserts, because of broadcasting in modern framework. 

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

## Episode 5 - Optimization for Machine Learning

### 01 - intro and motivation

Learn from data by adapting prameters, minimization of an objective function. Works by small incremental changes to model parameters which each reduce the objective by a small amount.

Notation $$\theta \in \mathbb{R}^n$$, objective function $$h(\theta)$$, goal of otimization $$\theta^* = \argmin h(\theta)$$

Neural network training objectvie, sum over examples, loss is a disagreement between labels and predictions, $$f(x, \theta)$$ taking input x and outputting some prediction. 

### 02 - gradient descent

Basic gradient descent iteration. Definition of gradient and learning rate. Intuition about gradient descent, it is 'steepest descent'. High smoothness vs. low smoothness in the objective function. Gradient $$\nabla h(\theta)$$ gives greatest reduction in $$h(\theta)$$ per unit of change.

Intuition.Gradient descent is minimizing a local approximation. Linear approximation using Taylor series, first order. $$d$$ small enough. Gradient update computed by minimizing within a sphere of radius $$r$$.

Problems of gradient descent in multiple dimensions, the narrow valley problem, hard to get a good learning rate change. No good signal in low learing rate case, too many steps.

Convergence theory, assumptions

- Lipschitz continues derivatives: means gradient doesn't change too much as we change parameters (upper bound on curvature)
- Strong convexity: function curves as least as much as the quadratic term (lower bound on curvature) (why would you want a lower bound? signal?)
- Gradients are computed exactly, i.e. not stochastic

If conditions apply upper bounds on number of iteratiorns to achieve proximity with epsilon. (Key is kappa, prefer smaller values of kappa, its ratio between smallest curvature and highest curvature, often called condition number but globally. Similar to biggest eigenvalue in hessian divided by smallest one).

useful in practice? Often too pessimistc (cover worst case examples), too strong assumptions (convexity), or too weak (real problems have more structure), rely on crude measures such as condition numbers, most important: focused on asymptotic behavior, often in practice we stop k long before that bound and there is no information about behavior in that section.

Design/choice of an optimizer should always be informed by practice, but theory can help to guide the way by builidng intuition. Be careful about anything 'optimal'

### 03 - Momentum methods

Motivation: Graident descent has tendency to flip back and forth (narrow valley example)

Key idea: Accelerate moviement along directions with low curvature

How:Use physics guidance like a ball rolling along the surface of the objective function (funny illustration)

Classical momentum: velocity vector $$v$$, friction is $$\eta$$

Nestrov variant: more useful in theory, sometimes in practice.

2D valley example. Never oscillate out of control, veloctiy keeps acummulating in the downward direction.

Upper bounds for nestrov momentum variant, now term depends on the square root of kappa, number of iterations is roughly the square root instead of linear. Makes bigger difference when kappa is large

In technical sense this is the best we can do for first order methods. Definition of first-order method. Updates are a linear combination of previous gradients we have seen, this includes a certain class of algorithms: gradient descent, momentum emthods, conjugate gradients. Not included: preconditioned gradient descent, 2-nd order methods. Given the definition of first order method, we can get a lower bound which says we can not converge faster than square root of kappa. Overview of iteration counts.

### 04 - 2nd-order methods

Problem with first order methods: dependency on condition number kappa. Very large for certain problems (some deep architectures), although not for ResNets. 2nd Order methods can improve or eliminate the dependency on kappa.

Derivation of Newtons method, using 2nd-order Taylor series, easiest but not necessary the best way. When minimizing with respect to d, you get the inverse of the Hessian. Update iteration remains the same. Can also add momentum into this, but can't help if you have perfect second order method (won't use momentum in this discussion, in practice it is used). Valley example revisited.

Comparison to gradient descent. See gradient descent as primitive second order method, maximum allowable global learning rate for GD to avoid divergence. Gradient descent implicitly minimizes a bad approximation of 2nd order taylor series. Subsittue Hesian with L times Identity matrix, says curvature is maximal everywhere (in all directions, max curvature), too pessimistic an approximation

Issues with local quaratic approximations

- Quadratic approximation of objective is only correct in  very local region (often neglected). 
- Curvature can be approximated to optimistically in a global sense (gradient descent doesn't have the issue because it takes the max global curvature)
- Newtons method uses the hession $$H(\theta)$$ which may become an understimate in the region once we take an updating step
- Solution: restrict updates into region around point where approximation is good enough (implementation is tricky)

Trust-regions and dampening, take region $$r$$ ball. Is often equivilant to global global hessian approximation plus $$\lambda I $$ for some $$\lambda$$. Lambda depends on $$r$$ in a complicated way, not need to worry to much about that in practice.

Alternative curvature matrices, going beyond the hessian. No one uses Hessian in neural netowrks, and you don't want to use it. It is local optimal (2nd order taylor seres), but that might not be what we want. Might want a more global view, or a more conservative approximation.

Most important families of examples:

- Generalized gauss-netwon matrix
- Fisher information matrix (often equivalent to first)
- "Empirical Fisher" (cheap but mathematically dubious)

Nice properties: Always positive semi-definite (no negative curvature, would mean you can go infintiely far). Parametrization invriant of updates in small learning rate limit (not like Newton). Emperically, it works better in practice (not that clear why).

Problems when applying 2nd order method for neural networks, 

- $$\theta$$ can have 10s of miilions of dimensions
- Can not compute and invert $$n \times n$$ matrix.
- Hence we must make approximations of the curvature matrix for computation, storage, and inversion. Approximate matrix with simpler form.

#### Diagonal approximation

Zero out all the non-diagonal elements, inversion is easy and O(n). Computation depends on form of original matrix. Unlikely to be accurate, but can compensate. Used in RMS-prop and Adam methods to approximate empirical fisher matrix. Obvious problem, very primitive approximation, only works if axis aligned scaling issues appear (valley example with two coordinates), normally curvature is not axis-aligned.

#### Block-diagonal approximations

Group could correspond to  weights going into neural net block, all weights associated with one particular layer, all weights for given layer. Storage cost O(bn), inversion cost O(b^2n), only realistic for small block sizes.

Blocks for one layer can still be in the millions, which is way to big. Not a popular approach for many years now.

#### Kronecker-product approximations

Block-diagonal as start, but approximate block digagnoal by kronecker product fo two smalle matrices. Has storage cost of more than O(n), but not that much more (slides are wrong). Apply inverse of O(b^0.5 n), could still be very large for one million parameters this will be a thousand. Used in most powerful but heavyweight optimizer, (K-FAC).

### 05 - stochastic optimization

So far everything was deterministic, because inttuion is easier to build. Why use stochastic methods? Number of samples can be very big, $$m$$ too large.

Mini-Batching: Often strong overlap between $$h_i(\theta)$$, gradients will look similar. Randomly subsample a mini-batch of size $$b << m$$ and estimate with mean.

Stochastic gradient descent, same as gradient descent but with mini batch.

To ensure convergece, we need to do

- Decay learning rate
- Use Polyak averaging, taking an average of all the parameter values visited in the past. Or use exponentially decaying average (works better in practice but theory is not as nice)
- Slowly increase the mini-batch size during optimization

Convergence properties of stochastic methods: Convergence is slower than non-stochastic versions. Polyak averaging is not that bad, no log of one over epsilon, this makes the depencency much worse for small epsilon. Do not that badly in practice. Best you can do in theory, can be proved due to intrinsic uncertainty. So Polyak averaging is optimal in asymptotical sense.

#### Stochastic 2nd order and momentum methods

Can use mini-batch gradient for 2nd-order and momentums too. Estimate curvature matrices stochastically with decayed averaging over multiple steps. But no stochasic optimization method that has seen same amount of data can have better asymptotic convergence speed than SGD with Polyak. But pre-asymptotic peformance matters! 2nd order can still be useful if: loss surface curvature is bad enough or mini batch size is large enough (slides are out of sync here, he points at no log slide (convergence of stochastic methods)). Nets without skip connections, i.e. harder to optimize profit from second order methods such as K-FAC (impossible to optimize with first order methods), if you include skip connections the performance difference disappears. Hence implicit curvature number of ResNet is pretty good. Second order methods are twice as slow, but with some implementation tricks you can get to only 10% slower.

making nets easier to optimize vs. making optimization better, new classes of models might profit from better optimization models.

Questions:

- Initialization: It is very hard, deep subject new results upcoming this year. For now every initialization start with the basic Gaussian factor. Becomes important when not using batch norm or skip connections.
- Batchnorm and resnet make network look linear in beginning, so its easy to optimize in beginning and only adds complexity over time.
- Regularization: Not important these days, should not rely on optimizer to do regularization
- No one measures condition number in practice, and it would not capture the essence (some dimensions might be completely flat)
- New emergent research, if you start with good initialization, neural networks loss surface will look quadratically convex in your neighborhood and you can get to zero error (global minimum) with the above mentioned techniques.
- Reason for lowering learning rate or Polyak averaging is inversely related to batch size (larger batch size reduces gradient variance estimate, so there is less need for those techniques.). Double batch size, double learning rate (as a naive rule)

### Highlights

- Gradient descent as linear or quadratic approximation by using Taylor series in small neighborhood.
- Optimality for various methods, especially SGD with Polyak in stochastic case
- ResNet and especially skip connections make the optimization problem much easier for the moment
- This line of research might become highly important again once we use different blocks



## Episode 6 - Sequences and Recurrent Neural Networks

### 01 - Motivation

So far vectors and images as inputs, now we look at sequences. Collections of elements where

- Elements can be repeated
- Order matters
- Variable length (potentially infinite)

Sequences are everywhere: words, letters, speech, videos, images, programs, decision making.

### 02 - Fundamentals

Training machine learning models, the four basic questions to answer: data, model, loss and optimization for the supervised case. In sequence we have

- Data: $$\lbrace x \brace_i$$
- Model: probaliity of a sequence $$p(x) = f_{\thetha}(x)
- Loss: loss of log sum
- Optimization similar than supervised case,

Modeling word probabilites is really difficult. Simplest model, product of individual words. But independence assumption does not match the real distribution.

More realistic model, conditional probability on previous words $$p(x_t) = p(x_t \mid x_1, \dotsc, x_{t-1})$$. Use the chain rule to get the joint probability. Has scalability issues, matrix of of size $$n \times n$$. Scales with $$vocabulary^N$$ where $$N$$ is the length of the sentence. Early NLP, fix windows size, called N-grams. Downsides, does not take into account words that are more than $$N$$ words away, data table is still huge.

Summary: Modeling probabilities of sequences scales badly

Learning to model word probabilities

1. Vectorising the context, approximate $$p(x_t \mid x_1, \dotsc, x_{t-1}) = p(x_t \mid h)$$

   Need the following properties for $$f_{\theta}$$

   1. Order matters
   2. Variable length
   3. Lernable (differentiable)
   4. Individual changes can have large effects (long-term dependency)

2. Modeling conditional probabilities. Desirable properties

   1. Individual changes can have large effects
   2. Returns probability distribution

#### Recurrent Neural Networks (RNN)

Persistent state ariable $$h$$ stores the information from the context observed so far

Two steps

1. Calculate hidden state $$h_t$$ from $$x_t, h_{t-1}$$
2. Predict the target $$y_{t+1}$$

We can unroll RNNs to backpropage. Loss is cross-entropy at each time step $$t$$, and sum up over all the words in a sentence. Paremeters $$\theta = W_y, W_x, W_h$$. Differentiating w.r.t. to $$W_y,$$, $$W_h$$ and $$W_y$$.

- $$h_t$$
- $$p(x_{t+1})$$
- $$L_{\theta}(y, \hat{y})_t$$

Need to back propagate through time. (Equations)

Vanishing gradients. Intuition example

- $$h_t = W_h h_{t-1}, h_t = (W_h)^t h_0$$ This will either go to infinity or to 0 depending on the norm of $$\mid W_h \mid$$.

Values are bounded by thanh, but the gradients are still affected by it. No gradients if value is not close to -1 or 1. How can we capture long-term dependencies?

#### Long short-Term Memory (LSTM) networks

Keeps a cell state $$c_t$$ and a some gates

- forget gate, combines current input and previous state : formula
- input gates: formula
- Output gate: formula

Whats missing: gradients are not vanishing due to the path they can take through $$c_t$$, not going through, not talking about gate gate, LSTM are more closer to ResNet because of the skip connections. There are also GRU, similar than LSTM and more recent.

### 03 - Generation

Using a trained model to generate new sequences, so far we focused on optimizing the log probability estimates produced by model. An alternative way to use them is generation.

Input first word and get $$\hat{y}$$, autoregressively  create a sequence. Use argmax to get maximum probability from the first output and put that in again

#### Imags as sequences PixelRCNN

Softmax Sampling over the entire image, starting on top left, interesting to look on the distribution for each pixel. It produces some okay looking image, but not great to state of the art.

#### Natural languages as sequences

Sequence-to-sequence models, start with english words and use that as initial state, then start outputting japanese words. Flexible setup

- One to one
- One to many
- many to one
- many to many (RNN)
- many to many  (sequence to sequence)

Google Neural machine Translation: Encoder and Decoder structure. Almost closed the gap between human and machine translation.

Image captioning: start with features from an image passed trough a neural network.

#### Audio waves as sequences

Audio waves as sequences: convolutions. Using dilated conolutions, predict one signal at once. Taking into account various time scales.

#### Policies as sequences

Models which sequentially decide where to paint on a canvas. Using RL to draw like a human inside a drawing program. OpenAI five and Alphastar playing games using LSTM.

AlphaStar architecture. The core is an LSTM which gets fed different inputs

#### Attention for sequences: transformers

Transformers vs. convolutions. Transformers is connected to every input, but weighted by attention. Example GPT2 which adapts to style and content.

### Questions

- Do models focus only on local consistency? Could be mixed with a memory module to incorporate truly long term connections, need more hierarchy in the model.
- Deep learning is hard to reconcile with symbolic learning, reasoning level is hard to measure. Can't see if it learns concepts.
- No constraints on hidden states, hard to interpret them, how they are working. Very model specific. Hard to get an intuitive understanding.

### Highlights

- Nice gradient calculation to show vanishing gradients in RNN
- Overview of properties and different models (RNN, LSTM, Conv, Transformers)

## Conclusion of first half of the course

Love the breadth of the lectures, sad that the questions are so hard to hear, the lecturers should repeat or summarize them quickly (as in the Stanford class o)