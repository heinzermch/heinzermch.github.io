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

*<u>Remark</u>: Considering that this theorem is for bounded activation functions, how does ReLU fit into the universal approximation theorem? It does not apply to that case. Most of the modern neural network use ReLU as non-linearity.*

What is the intuition behind the Universal Approximation Theorem? We want to approximate a function in $$[0,1]$$ using sigmoid functions, then we can combine two of them, red and blue to get a bump (orange):

![XOR problem for two classes](/assets/images/deepmind_lecture_part_1/e02_02_two_sigmoid_function_approximation.png)

The more neurons we add, the more of these bumps we can get. The closer we can approximate the function. In the example below we use six neurons to approximate the function and can get already close. Intuitively it should be clear that the more neurons we use, the closer we can approximate the target function (grey) with the sum of our bumps (orange).

![XOR problem for two classes](/assets/images/deepmind_lecture_part_1/e02_02_three_times_two_sigmoid_function_approximation.png)

The theorem confirms the intuition that we can get arbitrarily close to any target function by simply using more of these bumps.

In practice we often don't use sigmoid but rectified linear units (ReLU) as activation functions. It has the advantage that the derivative does not vanish in the right side. However it can lead to 'dead' neurons which only output zero, due to this property careful initialization is necessary. Even though the function is not differentiable at zero, this is not an issue in practice.

![XOR problem for two classes](/assets/images/deepmind_lecture_part_1/e02_02_relu.png)

Our overall goal is always to make the problem linearly separable in the last layer. In general is it more beneficial to create a wider than a deeper network? The question was answered in this [paper](https://arxiv.org/abs/1402.1869), expressing symmetries and regularities is much easier with a deep than a wide network. The number of linear regions grows exponentially with depth and polynomially with width.

![XOR problem for two classes](/assets/images/deepmind_lecture_part_1/e02_02_depth_as_folding.png)

It is helpful to see neural networks as computational graphs on which we perform operations. Light blue nodes are data, input and target. Red blocks are parameters or weights which influence the operation in dark blue nodes. Note that not every operation node has parameters, this is only the case for linear layers in this example. Generally we pass the data from left to right (forward pass) and the gradients from right to left (backward pass).

![Neural network as computational graph](/assets/images/deepmind_lecture_part_1/e02_02_neural_networks_as_computational_graphs.png)

Later levels in the network perform higher level tasks, such as line/corner detection compared to shape/object detection.

### 03 - Learning

A quick recap on linear algebra (*I would rather say calculus*). The gradient of a function

$$\begin{align}  f &: \mathbb{R}^n \longrightarrow \mathbb{R}  \\ 
y &= f(\mathbf{x})  \end{align}$$

is

$$ \frac{\partial y}{ \partial \mathbf{x}} = \nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f}{ \partial \mathbf{x}_1}, \dotsc, \frac{\partial f}{ \partial \mathbf{x}_2}\bigg] $$

The Jacobian of a function 

$$\begin{align}  f &:  \mathbb{R}^d \longrightarrow \mathbb{R}^k  \\ 
\mathbf{y} &= f(\mathbf{x})  \end{align}$$

is

$$\frac{\partial \mathbf{y}}{ \partial \mathbf{x}} = J_{\mathbf{x}} f(\mathbf{x}) = \begin{pmatrix}
 \frac{\partial f_1}{ \partial \mathbf{x}_1} & \dotsc &  \frac{\partial f_1}{ \partial \mathbf{x}_d} \\
 \vdots & \ddots & \vdots \\
 \frac{\partial f_k}{ \partial \mathbf{x}_1} & \dotsc &  \frac{\partial f_k}{ \partial \mathbf{x}_d}  \\
\end{pmatrix}$$

Gradient descent is then for a sequence of parameters $$\mathbf{\theta_t}, t \in \mathbb{N}$$:

$$ \mathbf{\theta_{t+1}} := \mathbf{\theta_t} - \alpha_t \nabla_{\mathbf{\theta}} L(\mathbf{\theta_t})$$

where $$\alpha_t$$ is the learning rate at time $$t$$. Choosing a correct learning rate is crucial. In the case of non-convex "smooth enough" functions this converges to a local optimum. For each in the computational graph we pass the input in the forward pass and the gradient with respect to the loss function in the backward pass.

![Forward and backward pass for each node](/assets/images/deepmind_lecture_part_1/e02_03_forward_backward_pass.png)

The whole optimization process is built on a few basic principles. The first one is automatic differentiation, for most loss functions and layers we have programs such as TensorFlow or PyTorch which are able to automatically differentiate the common loss functions and layers. No manual gradient calculation is required. There are two fundamental tools which are applied throughout the graph. The first is the chain rule visualized in the image below and written in its basic form on the box on the lower left.

![Chain Rule](/assets/images/deepmind_lecture_part_1/e02_03_chain_rule.png)

The other tool is back-propagation visualized in the graph below. It allows us to calculate the partial derivatives efficiently for each node (linear in the number of nodes). 

![Backpropagation](/assets/images/deepmind_lecture_part_1/e02_03_backprop.png)

A basic example of how the forward and backward pass work for a linear layer is illustrated below. Note that the derivative is taken with respect to the two parameters $$W$$ and $$b$$ as well as the input $$x$$.

![Linear layer backward pass](/assets/images/deepmind_lecture_part_1/e02_03_linear_layer_backward_pass.png)

The same can be done for other parts of the network, for more details check the presentation. Note that for numerical stability the cross-entropy of logits is usually done in a single operation.

*<u>Remark</u>: For more details on the forward and backward pass of these basic building blocks see my [blog post](https://heinzermch.github.io/posts/creating-a-NN-from-scratch-part-1/) on building a neural network from scratch*

### 04 - Pieces of the Puzzle

Other frequent operations used in neural networks are the max operation, where gradients only flow through the element which is maximal. This is a part of max pooling. No parameters can be learned

![Max as computational graph](/assets/images/deepmind_lecture_part_1/e02_04_max_computational_graph.png)

Another way of choosing which elements to pass forward is to do element wise multiplication such as in an attention layer (*see lecture 8*). Here it is possible to learn the probability distribution (using softmax).

![Conditional execution as computational graph](/assets/images/deepmind_lecture_part_1/e02_04_conditional_execution_computational_graph.png)

### 05 - Practical Issues

A classical problem in ML is overfitting, our model has too many parameters and learns the training data set too precisely. Hence it will perform poorly on the test set, it does not generalize. There are a couple of techniques which can help mitigate or avoid those cases for neural networks:

![Classical ML overfitting](/assets/images/deepmind_lecture_part_1/e02_05_overfitting_regularization.png)

The above are classical results from statistics and Statistical Learning Theory, however this isn't always true for neural networks. There  are cases where the model initially becomes worse, before it becomes better again when we increase parameters (double descent). These are very [recent](https://arxiv.org/abs/1812.11118) [results](https://openai.com/blog/deep-double-descent/) from 2019. What this tells us is that model complexity is not as simple as the number of parameters they have. Even if bigger models don't necessarily over-fit anymore, they can still benefit from regularization techniques.

![Overfitting in the NN scenario](/assets/images/deepmind_lecture_part_1/e02_05_overfitting_regularization_in_nn.png)

Training, diagnosing and debugging neural networks is not always easy. Often they fail silently, unlike classical programs a bug is not obvious and won't make your program crash instantly. Here are some pointers to verify the correctness:

- Be careful with initialization
- Always over-fit on a small sample first (*can highly recommend this, the simplest test to see if your basic setup works*)
- Monitor training loss
- Monitor weight norms and NaNs (*in my experience a NaN loss or accuracy is the closest you get to an obvious bug*)
- Add shape asserts (to detect when your NN framework broadcasts a parameter into a different shape than you expect)
- Start with Adam (optimizer)
- Change one thing at a time (*Something I can recommend highly from experience, otherwise it is almost impossible to attribute changes in accuracy to changes in code/data. In papers this is called ablation studies*)

More detailed information can be found in Karpathy's [blog post](http://karpathy.github.io/2019/04/25/recipe/).

### 06 - Bonus: Multiplicative Interactions

What can Multi Layer Perceptrons (MLPs) not do?

$$f(x,y) = \langle x,y \rangle$$

They can not represent multiplication. As the graphs below show, the number of parameters required to approximate multiplicative interaction grows exponentially with the number of input dimension. Being able to approximate something is not the same as being able to represent it. Approximation can be highly inefficient.

 ![Multiplicative interactions and a parameters to approximate them](/assets/images/deepmind_lecture_part_1/e02_06_multiplicative_interactions.png)

 Closing words:

#### "If you want to do research in fundamental building blocks of Neural Networks, **do not seek** to marginally improve the way they behave by finding a **new activation function**. Ask yourself what current modules cannot represent or guarantee right now, and propose a module that can."

*<u>Remark:</u> This is something I can wholeheartedly agree with, too many of todays papers are happy to show an increase in 0.5% in accuracy in some benchmark, by doing a minor tweak. There hasn't been a relevant improvement for practitioners since BatchNorm layers and residual connections.* 

### My Highlights

- Approximation is not the same as representation
- XOR example with two layers
- Universal Approximation Theorem
- ReLu can be seen as folding space on top of each other ([paper](https://arxiv.org/abs/1402.1869))
- Deep Double Descent and over-parametrization in neural networks
- MLPs and multiplicative operations ([paper](https://openreview.net/pdf?id=rylnK6VtDH), follow up [read](https://arxiv.org/abs/2006.07360))

## Episode 3 - Convolutional Neural Networks for Image Recognition

### 01 - Background

What is an image? An image is a 2D grid of pixels, but neural network expects a vector of numbers as an input. So how can we transform it? One way is by flattening the grid to a vector, simply attaching each row at the end of the previous one. One issue with this is that if we shift the input in the original space by some pixels, it will produce a completely different input. The case is illustrated below.

 ![Feeding an image into a network](/assets/images/deepmind_lecture_part_1/e03_01_feeding_images_to_a_neural_network.png)

Moreover we would also want to make the network take into account grid structure. We can summarize this in two key features:

- Locality: nearby pixels are strongly correlated
- Translation invariance: patterns can occur anywhere in the image

 ![Locality and translation invariance](/assets/images/deepmind_lecture_part_1/e03_01_locality_and_translation_invariance.png)

The above is true for images but also for other modalities, for example sounds can occur anytime in signal. This is translation invariance in time. And in textual data, words can occur anytime. In graph structures, molecules can exhibits patterns anywhere. How can we put these properties into a network and take advantage of topological structure? By using the two following properties

- Weight sharing: use same parameters to detect same patterns all over the image
- Hierarchy: low-level features are composed, model can be stacked to detect these

 ![Locality and translation invariance](/assets/images/deepmind_lecture_part_1/e03_01_weight_sharing_and_hierarchy.png)

### 02 - Building Blocks

What we want to do is going from a fully connected operator below to a locally connected one. Note that the image does not really display a fully connected layer, there are all connections missing which do not occur at the border of the image (keeps the graphic readable).

![Fully connected layer on an image](/assets/images/deepmind_lecture_part_1/e03_02_fully_connected.png)

The example below shows how we force the network to be locally connected and share the same weights over all the image: a 3x3 convolution. The operation becomes equivariant to translation. In the context of neural networks we often call the weigths kernel or filter. This kernel slides over the whole image and produces a feature map. For each pixel in the feature map we call the input which goes into the calculation for that pixel the receptive field.

![Locally connected with weight sharing](/assets/images/deepmind_lecture_part_1/e03_02_locally_connected_weight_sharing.png)

In a real world application, we will always have more than one filter. Each one of them will go over the input and produce a feature map, these feature maps are often called channels.

![Multiple channels](/assets/images/deepmind_lecture_part_1/e03_02_multiple_channels.png)

 Filter operation essentially with learned weights. Multiple feature maps are channels. There are many variants of the convolution operation:

- Valid convolution: no padding, slightly smaller output
  ![Valid convolution](/assets/images/deepmind_lecture_part_1/e03_02_valid_convolution.png)
- Full convolution: padding so that only one pixel overlaps in minimum, larger output than input
  ![Full convolution](/assets/images/deepmind_lecture_part_1/e03_02_full_convolution.png)
- Same convolution: padding so that output has same size as input, works better for odd kernel size
  ![Same convolution](/assets/images/deepmind_lecture_part_1/e03_02_same_convolution.png)
- Strided convolution: step size > 1, makes output resolution by at least 2x smaller
  ![Strided convolution](/assets/images/deepmind_lecture_part_1/e03_02_strided_convolution.png)
- Dilated convolution: kernel is spread out, increases receptive field, step size > 1 inside kernel. Can be implemented efficiently
  ![Dilated convolution](/assets/images/deepmind_lecture_part_1/e03_02_dilated_convolution.png)
- Depthwise convolution: Normally each input channel is connected to each output channel, here every input channel is connected to only one output channel
  ![Depthwise convolution](/assets/images/deepmind_lecture_part_1/e03_02_depthwise_convolution.png)
- Pooling: compute mean or max over small windows
  ![Pooling operation](/assets/images/deepmind_lecture_part_1/e03_02_pooling.png)



### 03 - Convolutional neural networks

A basic convolutional neural network has the structure as in the image below. We stack blocks of convolutions, non-linearity and pooling together and repeat them as many times as possible. In the end there are some blocks consisting of fully-connected and non-linearity layers. This structure is typical for networks used for classification. Note that we will not use explicit blocks for weights and loss functions anymore.

![Convnet as graph](/assets/images/deepmind_lecture_part_1/e03_03_conv_net_as_graph_simplified.png)

### 04 - Going Deeper: case studies

Here we are going to see the most important evolutionary steps in neural network for images. They will adress some of the key challenges of deep networks: computational complexity and optimization. Some of the techniques that can help are: initialization, sophisticated optimizers, normalizations layers, network design. Lets start simple with the example from the previous image:

- LeNet-5 (1998): Has 5 layer, sigmoid (used for handwritten digit recognition)
  ![LeNet as computational graph](/assets/images/deepmind_lecture_part_1/e03_04_le_net.png)
- AlexNet (2012) : Has 8 layers, ReLU, start with a 11x11 kernel, not every convolution needs to be followed by pooling
  ![AlexNet](/assets/images/deepmind_lecture_part_1/e03_04_alexnet.png)

- VGGNet (2014) : up to 19 layers, use same layers to avoid reduction. It only uses 3x3 kernels and stacks them instead of larger kernels. It uses same padding to avoid resolution reduction.
  ![VGGNet](/assets/images/deepmind_lecture_part_1/e03_04_vggnet.png)

  Stacking increases the receptive fields with fewer parameters and more flexibility.
  ![AlexNet](/assets/images/deepmind_lecture_part_1/e03_04_stacking_convolutions.png)

  Interestingly the error plateaus at 16 layers, training a network with 19 layers was worse. This is because these kind of networks are hard to optimize. There are two main innovations which made deeper networks more trainable.

- Batch Normalization: Normalize each input pixel over the batch dimension by subtracting the mean $$\mu$$ and dividing by the standard deviation $$\sigma$$. Add and multiply the result with two lernable parameters $$\gamma$$ and $$\beta$$.
  ![BatchNormalization](/assets/images/deepmind_lecture_part_1/e03_04_batch_normalization.png)
  This simple idea reduces sensitivity to initialization and acts as a regularizer. It makes networks more robust to different learning rates and introduces stochasticity (because $$\mu$$ and $$\sigma$$ have to be estimated for each batch, this can also help make a model more robust). However at test time this introduces dependency on other images in batch. What we generally do is freezing those values for testing time, this can be a source for a lot of bugs.

- Residual connections, introduced in ResNet (2015): The ResNet can be up to 150 layers deep. It can do that because it introduces residual connections, which make makes training deeper networks simpler by letting the network 'choose' which layers it needs. In the simplest case the network can simply be a identity function.

  ![Residual Connections](/assets/images/deepmind_lecture_part_1/e03_04_residual_connection.png)
  Generally the first convolution acts as a bottleneck block, applying a 1x1 convolution to reduce channels. Then a 3x3 convolution can be applied on fewer channels, allowing for lower computational costs and fewer parameters. Finally a last 1x1 convolution restores the original number of channels. Interestingly this setup allows the ResNet-152 to be computationally cheaper than than VGG network. The version 2 avoids nonlinearities in residual pathways. 
  ![Residual Connections version 2](/assets/images/deepmind_lecture_part_1/e03_04_residual_connection2.png)

- DenseNet (2016): connections to all previous layers, not only one
  ![DenseNet](/assets/images/deepmind_lecture_part_1/e03_04_dense_net.png)

- AmoebaNet (2018): One of the first architectures not designed by a man but but by an algorithm. This is called neural architecture search, here an evolutionary search algorithm was used. The network is basically an acyclig graph composed of predefined layers.
  ![Amoebanet](/assets/images/deepmind_lecture_part_1/e03_04_amobaenet.png)

- Other ways to reduce complexity of networks: 
  - Depthwise convolutions, 
  - separable convolutions, 
  - inverted bottlenecks (MobileNetV2, MNasNet, EfficientNet)

### 05 - Advanced topics

So far we have seen networks being robust to tranlsation. Using data augmentation, we can make them robust against other transformations: rotation, scaling, shearing, warping and many more.

![Data Augmentation](/assets/images/deepmind_lecture_part_1/e03_05_data_augmentation.png)

We can visualize what convolutional network learns, by maximizing one activation using gradient ascent. Even with respect to a specific output neuron (class). 

![Visualizing layers](/assets/images/deepmind_lecture_part_1/e03_05_visualizing_layers.png)

For more details and explanations there is read the article on [distill.pub](https://distill.pub/2017/feature-visualization/), Other topics: 

- pre-training and fine-tuning
- Group equivaraint convnets: invariance to e.g. rotation and scale
- Recurrence and attention (*these are discussed in later lectures*)

### 06 - Beyond image recognition

Other tasks beyond classification are: Object detection (top right), semantic segmentation (bottom left), instance segmentation (bottom right). We will see more of them in future lectures.

![Beyond visual recogntion or classification](/assets/images/deepmind_lecture_part_1/e03_06_visual_tasks_beyong_recognition.png)

### My Highlights

- BatchNorm introduces stochasticity

## Episode 4 - Advanced models for Computer Vision

Holy grail: Human level scene understanding

### 01 - Supervised image ~~classification~~ - Tasks beyond classification

#### **Task 1 - Object Detection**

![Object Detection](/assets/images/deepmind_lecture_part_1/e04_01_object_detection.png)

Input are again images, the output is class label, which is one hot encoded, and a bounding box with $$(x_c, y_c, h, w)$$ for every object in the scene. Data is not ordered, do regression on coordinates. Generally minimize quadratic loss for regression task $$l(x,y) = \| x-y\|^2_2$$.

![Bounding box prediction](/assets/images/deepmind_lecture_part_1/e04_01_bounding_box_prediction.png)

How to deal with multiple targets? Redefine problem, first do classification and then regression by discretising the output values in one hot label. 

![classification then regression](/assets/images/deepmind_lecture_part_1/e04_01_classification_to_regression.png)

Case studies:

**Faster R-CNN, two-stage detector**

When we redefine the problem we need to set up anchor points over the image, these are centers of candidate bounding boxes and correspond to the $$(x_c, y_c, h, w)$$ part of the problem.

![Anchor points](/assets/images/deepmind_lecture_part_1/e04_01_anchor_points.png)

For each of these anchor points we add multiple candidate bounding boxes. They have different scales and ratios for $$(h, w)$$. For each combination of anchor point and bounding box an objectness score will be predicted. In the end they will be sorted and the top K will be kept.

![Candidate bounding boxes](/assets/images/deepmind_lecture_part_1/e04_01_candidate_boxes_per_anchor_points.png)

These top K predictions are then passed on through another small MLP which will refine them further by performing regression. This is a so called two-stage detector as a non-differentiable step is involved. This means we have to train the system in two parts, one which predicts objectness and and the second which performs the regression on the result of the first one. This system achieves good accuracy at a speed of approximately 5 frames per second. There exist versions which involve only differential operations, as for example in [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025). 

![Two stage detector](/assets/images/deepmind_lecture_part_1/e04_01_region_proposal_network.png)

**RetinaNet - one-stage detector**

One way of solving the two stage problem is to predict classes and coordinates of bounding boxes straight away. The RetinaNet architecture uses a feature pyramid network to combine semantic with spatial information for more accurate predictions. A basic ResNet architecture extracts features which are then upsampled back to a higher resolution and enriched with spatial information from earlier stages of the network using skip connections. For each level of the pyramid layer K classes and A anchor boxes are predicted. A parallel network will predict four regression coordinates for each anchor box.

![RetinaNet architecture](/assets/images/deepmind_lecture_part_1/e04_01_retina_net_architecture.png)

However this construction will not train out of the box with a standard loss such as cross-entropy. The problem is the large class imbalance, out of thousands of bounding boxes almost all of them will be background. Even if they are classified with high confidence, their sheer number will overwhelm the signal generated by the rare useful examples. Even if we correctly assign a high probability (i.e. > 0.6) to the background class, the loss value will still be significantly larger than zero.

![CE loss overwhelming simple examples](/assets/images/deepmind_lecture_part_1/e04_01_cross_entropy_easy_samples.png)

 Faster R-CNN pruned most of these easy negatives in the first stage during the sorting. One-stage detectors solved the problem by employing [hard negative mining]((https://www.semanticscholar.org/paper/Learning-and-example-selection-for-object-and-Sung/f3317b98195fe0be4acf7b450f015c1abca13ab9)) procedure:

1. Get set of positive examples
2. Get a random subset of negative examples (as the full set of negatives is too big)
3. Train detector
4. Test on unseen images
5. Identify false positive examples (hard negatives) and add them to the training set
6. Repeat from step 3

However one of the main contributions of RetinaNet was the introduction of focal-loss:

$$l_{CE}(p_t) = -(1-p_t)^{\gamma}\log(p_t)$$

Where $$\gamma$$ is a hyperparameter, $$\gamma=0$$ corresponds to a classical cross-entropy loss. The introduction of this additional factor of $$(1-p_t)^{\gamma}$$ reduces the loss of high confidence examples enough to make training stable without additional procedures such as hard negative mining.

![Focal loss for different values](/assets/images/deepmind_lecture_part_1/e04_01_focal_loss.png)

This development lead to new state of the art results at 8 frames per second.

#### **Task 2 - Semantic Segmentation**

Semantic Segmentation is the task of assigning each pixel to a class. The problem here is to generate an output at the same resolution as the input. All the examples we have seen so far have sparse output.

![Semantic segmentation](/assets/images/deepmind_lecture_part_1/e04_01_semantic_segmentation.png)

The spatial resolution loss is mostly caused by the pooling layers, which create a larger receptive field, but we lose spatial information, hence we need a reverse operation. There are multiple operations which do 'unpooling' operations: unpooling, upsampling, deconvolutions.

![Unpooling operation](/assets/images/deepmind_lecture_part_1/e04_01_unpooling.png)

**U-Net**

The U-Net architecture follows an encoder-decoder model, with skip connections to preserve high frequency details. A pixel-wise cross-entropy loss is applied to the output.

![U-Net](/assets/images/deepmind_lecture_part_1/e04_01_unet.png)

The skip connections are similar to the residual blocks in the ResNet architecture, they help back-propagating the gradients, and make the network easier to optimize. The architecture is also closely related to what we saw the previously in RetinaNet, where the architecture was described as feature pyramid architecture, they both follow an U shape.

#### **Task 3 - Instance Segmentation**

If we combine object detection with semantic segmentation, then we get instance segmentation. It allows us to separate overlapping classes, such as multiple sheep in the same image below. The general setup for instance segementation is very similar to object detection, individual objects are predicted with a bounding box, but then an additional semantic segmentation head is applied to get more precise contours. The model generally used for instance segmentation is very similar to Faster R-CNN and is called [Mask R-CNN](https://arxiv.org/abs/1703.06870v3).

![Instance segmentation](/assets/images/deepmind_lecture_part_1/e04_01_instance_segmentation.png)

#### **Metrics and benchmarks**

Classification was easy: Accuracy as percentage of predictions. Sometimes we also use metrics which count a prediction as correct when it is in the top-5.

Object detection and segmentation: Intersection over Union (non-differentiable, so it's only for evaluation). 

![Intersection over union](/assets/images/deepmind_lecture_part_1/e04_01_iou.png)

Pixel wise accuracy would be foolish to use. Not recommended in general to use different measures between training and testing. (IoU not differentiable because of max operations)

*<u>Remark</u>: Here some more details would have been nice. While for semantic segmentation, IoU or rather mIoU (mean Intersection over Union) is frequently used, in object detection mAP (mean average precision) is more common. While it is at it is also based on IoU, it adds another layer of complexity on top.*

Some common benchmark sets used to evaluate those tasks: cityscapes for semantic segmentation and COCO for instance segmentation.

#### **Tricks of the trade**

Transfer learning. Reuse knowledge learned on one task on another one. Formally: Let

 $$\mathcal{D} = \lbrace \mathcal{X}, P(X) \rbrace, X = \lbrace x_1, \dotsc, x_n \rbrace \in \mathcal{X}$$

 be a domain and 

$$\mathcal{T} = \lbrace \mathcal{Y}, f( \cdot ) \rbrace, f(x_i) = y_i, y_i \in \mathcal{Y}$$

 a task defined on this domain. Given a source domain task $$ \begin{pmatrix} \mathcal{X}_S \\  \mathcal{T}_S \end{pmatrix}$$ and a target domain task $$ \begin{pmatrix} \mathcal{X}_T \\  \mathcal{T}_T \end{pmatrix}$$, re-use what was learned by $$f_S :  \mathcal{X}_S \longrightarrow \mathcal{T}_S$$ in $$f_T : \mathcal{X}_T \longrightarrow \mathcal{T}_T $$.

*<u>Remark</u>: This definition seems to be very close to what can be found on [Wikipedia](https://en.wikipedia.org/wiki/Transfer_learning).*

Intuition is features are shared across tasks and dataset. Don't start from scratch all the time. Reuse knowledge across tasks or across data.

- Transfer learning across different tasks: Mostly remove the last layers and add new layers to adapt to the new task. See [Taskonomy](http://taskonomy.stanford.edu/) paper, for an overview of different tasks and how they are related.
  ![Transfer learning across tasks](/assets/images/deepmind_lecture_part_1/e04_01_transfer_learning_task.png)
- Transfer learning across different domains: Train in simulation, test in the real world. Use tricks to adapt to target domain (for example domain randomization and hard negative mining to identify the most useful augmentations)

### 02 - Supervised ~~image~~ classification - Beyond single image input

Experiment with people who recovered from blindness from surgery to see which tasks are hard, by making them recover objects from a scene. They have trouble identifying some scenarios, but you show them same type of images, but with moving images then they do much better. Motion really helps with object recognition when learning to see. Conclusion: should use videos for training object recognition.

#### **Input - Pairs of images**

What is optical flow estimation? Input are two images, for each pixel in image one $$I_1$$, we would like to know where it did end up in image two $$I_2$$. The output is a map of pixel wise displacement information, for horizontal and vertical displacement.

![Optical flow visualized](/assets/images/deepmind_lecture_part_1/e04_02_optical_flow.png)

**Case study: FlowNet**

Encoder-Decoder architecture like U-Net, fully supervised with euclidean distance loss. Invented Flying chairs dataset with sim to real to learn about motion. Essence is that pixels that move together belong to the same object.

#### **Input - Videos**

Base case: use semantic segmentation video and apply it frame wise (but don't use temporal information), this leads to flickering in a video. We could use 3D convolutions by stacking 2D frames to get volumes, kernels are 3D objects. In 3D convolutions the kernel moves in both space and time to create spatio-temporal feature maps (we can re-use strided, dilated, padded properties). 

![3D convolutions visualized](/assets/images/deepmind_lecture_part_1/e04_02_3D_conv.png)

3D convolutions are non-causal, because you take into account frames from the future (times $$t-1, t, t+1$$), which is fine for off-line processing, but when applying it in real time we have to use masked 3D convolutions.

Can do action recognition, use video as input, target is an action label which is one-hot encoded.

**Case study: SlowFast**

SlowFast is a two branch model which takes inspiration from the human visual system, which has also two streams:

- Hight frame rate branch: Less features, tracks high frequency changes.
- Low frame rate branch: More features, abstract information stays the same over longer time frame.

![3D convolutions visualized](/assets/images/deepmind_lecture_part_1/e04_02_slow_fast.png)

Transfer learning can be used by inflating 2D image filters to 3D filters by replicating them along the time dimension.

![3D convolutions visualized](/assets/images/deepmind_lecture_part_1/e04_02_transfer_learning_video.png)

#### **Challenges in video processing**

It is difficult to obtain labels, models have large memory requirements, high latency, and high energy consumption. Basically we need too much compute for it to be broadly applicable. Ongoing area of research of how to improve that by using parallelism and exploit redundancies in the visual data. One idea is to train a model to blink, (humans do it more often than necessary to clean the eye to reduce cognitive load).

### 03 - ~~Supervised~~ image classification - Beyond strong supervision

Labeling is tedious and a research topic on itself. Humans can only label keyframes and methods will propagate labels. Example: PolygonRNN.

#### **Self-Supervision - Metric Learning**

Standard losses such as cross-entropy and mean square error learn a mapping between input and output. Metric learning leanrs to predict distances between inputs in an embedding to give similarity measure between data. Create clusters with same person, for unseen data use nearest neighbor. 

![Metric learning, faces example](/assets/images/deepmind_lecture_part_1/e04_03_metric_learning_faces.png)

Used for learning self-supervised representations, information retrieval and low-shot face recognition.

**Contrastive loss** 

Also called margn loss, data is triplets $$l(r_0, r_1, y)$$ where $$y=1$$ if $$r_0$$ and $$r_1$$ are the same person and $$y=0$$ otherwise. The loss function is then

$$l(r_0, r_1, y) = y d(r_0, r_1)^2 + (1-y)\big(\max(0, m-d(r_0, r_1))\big)^2$$

 where $$ d(\cdot, \cdot)$$  is the Euclidean distance. We use the margin $$m$$ to cap the maximal distance, however it is hard to choose a value for $$m$$. All classes will be clustered in a ball of radius $$m$$, which can be unstable.

Use margin to cap the maximal distance. See the comparison with euclidean distance. But it is hard to choose $$m$$, the margin. All classes will be clustered in a ball with radius $$m$$, which can be unstable.

![Contrastive loss visualization](/assets/images/deepmind_lecture_part_1/e04_03_contrastive_loss.png)

**Triplet loss**

Use triplets $$(r_a, r_p, r_n)$$ where $$(r_a, r_p)$$ are similar and $$(r_a, r_n)$$ are dissimilar ($$p$$ for positive and $$n$$ for negative example). The loss function is then:

$$l(r_a, r_p, r_n) = \max(0, m+d(r_a, r_p)^2 - d(r_a, r_n)^2)$$

This works better than the contrastive loss because relative distances are more meaningful than a fixed margin. However one needs hard negative mining to select [informative triplets for training](https://arxiv.org/abs/1706.07567).

![Triplet loss visualization](/assets/images/deepmind_lecture_part_1/e04_03_triplet_loss.png)

#### **State-of-the-art**

The current [state of the art](https://arxiv.org/abs/2002.05709) uses the same data, but different augmentations. Especially compositions of data augmentations are important, results are comparable to a supervised Resnet-50 (although with more parameters).

![Augmentations for representation learning](/assets/images/deepmind_lecture_part_1/e04_03_sota_representation_learning.png)

### 04 - Open Questions

Is vision solved? What does it mean to solve vision? Need the right benchmarks

How to scale systems up? Need a system that can do all the tasks at once, more common sense. Different kind of hardware.

What are good visual representations for action? Keypoints could help.

### Conclusion

Need to rethink vision models from the perspective of moving pictures with end goal in mind.

### My Highlights

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