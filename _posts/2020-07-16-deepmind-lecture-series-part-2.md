---
layout: post
author: Michael Heinzer
title:  "DeepMind Lecture Series - Part II"
description: Taking notes and summarizing
date:   2020-12-21 11:00:00 +0530
categories: DeepLearning NLP Transformer Attention Memory
comments: no
published: no
---
I was watching this DeepMind lecture series. 

## Episode 7 - Deep Learning for Language Understanding

### 01 - Background: Deep Learning and Language

What is not covered. Speech, many NLP tasks, field is much bigger. Some tasks still use almost no neural networks, hard to do end to end with home assistants dialogue systems. Trend is moving towards neural and deep papers. Performance in GLUE benchmark, represents challenging language tasks, performance is still increasing up to this day. 

![Visualization of the language field](/assets/images/deepmind_lecture_part_2/e07_01_language_field.png)

Why is deep learning such an effective tool for language processing? Need to think about language itself, and why it fits together with deep learning. Mapping symbols to symbols, but its not actually that simple:

- Did you see the look on her **face**?
- We could see the clock **face** from below
- It could be time to **face** his demons
- There are a few new **faces** in the office today

 Delve into the meaning of face, have certain aspects in common but are not exactly the same. Pointing aspect of face. 

![Visualization of the language field](/assets/images/deepmind_lecture_part_2/e07_01_face_ambiguous.png)

Disambiguation depends on context. Went and event in handwritten sentence is easily read by humans.

![Visualization of the language field](/assets/images/deepmind_lecture_part_2/e07_01_hand_writing.png)

Important interactions can be non-local, look at wider context. Examples:

- The man who ate the pepper sneezed
- The cat who bit the dog barked

People are much slower to make sense of the second sentence, even they have the same overall structure. It is unusual that the cat does the barking, "the dog barked" captures our attention. The urge to consider this comes from our understanding of the world.

How meanings combine depends on those meanings:

- Pet: brown, white, black
- Fish: silver, grey
- Pet fish: orange, green, blue, purple, yellow

Combining pet and fish to pet fish changes the color associated with them.

![Visualization of the language field](/assets/images/deepmind_lecture_part_2/e07_01_pet_fish.png)

### 02 - The Transformer

Published just couple of years ago, large impact on the field. Any problem that requires a model to process a sentence, or multiple sentences and computes behavior or prediction on that. State of the art for these predictions. See paper attention is all you need.

First layer

distributed representation of words. First we have to define vocabulary, chop up the units the network is going to see. Could go down to pixels, can either be character level or word level. Split input according to white space in text. A model that just takes symbols might not be optimal as seen in previous section. First consider all the words which the model needs to be aware of. Parse each of the words through an input layer. Put an activation of one corresponding to one word, turns on the weights of "the"? Basically describes how you vectorize the input. Embedding dimension of $$D$$. Model can move the embedding during training. For $$V$$ words and $$D$$ dimension of embedding gives you $$V \times D$$ weights. Words with similar meaning or synthax (noun, verb, subject) cluster together in space.

Self-attention over word input embeddings

Query matrix $$W_q$$. Key matrix $$W_K$$ , value matrix $$W_V$$. Applied to each input. End up with three vectors per input vector, $$q, k, v$$. Take the dot product, how strong should the connection between the words be. Key value inner product query value, normalize with softmax layer. How query correspond to key in the input. Gives us a probability distribution, tells us to what extent is there an interaction. Beetle relates strongly to drove. Use it to know how much of the value should be propagated where through the network. Insert weighted sum.

Number of embeddings has not been changed, but the information in each embedding has been updated according on how to it interacts with other representations.

Multi head self-attention

Previous process can be parametrized by these three matrices. We have multiple of the operations applied in parallel, here four times. Four sets of three matrices. Can use a lot of memory, so reduce dimensionality, go from 100 in input to 25, we do that 4 times and those can be aggregated, in linear layer $$W_0$$. Have four independent ways of analyzing the interactions, without expanding the dimensionality.

Feedforward layer

Output of self attention run this, conceptually not that interesting

A complete transformer block

Multiple times apply multi-head self-attention. Have skip connections. Why is the skip connection important? Importance of expectation of what something is, wider understanding of the world and reconsider how the world works. Higher level of understanding has a possibility to interact with lower level of understanding. Remodulate how we understand the input

Remark: Connections to Resnet, LSTM, optimization

Position of encoding of words

So far we did not take the order into account. Did not express the closeneess in a sentence of words. Need positional encoding of words for that. Use sinosoid function, look at relationship of certain wavelength. Each unit can look at different distances in word.  LSTM have the order built in, but hard to remember things long time in the past. Transformer does not have this problem, easier to learn the word order than to be given it but have to learn to pay attention to things long time in the past. Shorter path for the gradient to pass through.

Summary

4 things we want to do an map them to transformer features

### 03 - Unsupervised and transfer learning with BERT

Arrow: we know how it flies. Similar sentecnes "times flies like". Fruit flies, background knowledge.

Understanding is balancing input with knowledge (of the world). Key motivating fact behind the approach in BERT. Application of transformer architecture, rather than simply train input. We do pre-training with a much larger amount of text. Transformer is really just a mapping from a set of distributed word representations to another set of distributed word representations (of same length). Extract knowledge from unlabled data, to give the model knoweldge or meaning in an unsupervised way.

Consider the problem of mapping a sequence of words to the exact same sequence of words. In the input, mask one word, model has to make a prediction on what the missing word was (missing word is knoweldege in image). probability of 15% that word is masked in training. But there is a risk that the model would then not behave well if you give it a complete sentence in the test set, so in 10% of the previous cases you leave the word in place.

Next sentence prediction pretraining. Make a prediction if it was two consecutive sentences, binary classificaion. Can shuffle data, give two special tokens to say where sentences start and end. Makes BERT get knowledge on how sentences and words flow. Because no labels are required, you can use any text from the internet to do training. How do you do evaluation? Use the knowledge as pre-trained weights. Then train on the standard NLP tasks to fine tune. These tasks normally have considerable less data available. Also often add a bit machinery on top of BERT to adapt to tasks. Massively improved the state of the art on glue. Basically transfer learning. 

Remark: Could we do something similar for vision? Huge number of unlabled images available.

### 04 - Ground language learning at DeepMind: towards language understanding in a situated agent

Further ways to endow models with conceptual knowledge. Extract knowledge from the surroundings. Exciting techniques in other fields, vision or actions. More robust understanding of the surroundings.

Knodwledge aggreation from predection. Old idea in psychology. Create a simulated world in unity game engine, study if an agent moving around in the world can gain understanding from his surroundings. Would that knowledge be useful for language tasks afterwards? Questions for random rooms generated. Propositional vs procedural knowledge. if something is true or false in the environment. Give the agent the policy of exploring the room. Model learns from that experience. Question is not asked at beginning, only asked at the end of the episode. Agent needs a large amount of general knowledge of how the room is structured, the QA module will have to extract the answer from that knowledge afterwards. There is no back-propagation from the answer to the question. Agent needs to have general knoweldge not specific question. Baseline is LSTM, more complicated to apply transformer. Endow agents with predictive learning, predicts the experience forward in time. Afterwards compare those, unrolls. Two algorithms, SimCore and Action-conditional CPC.

Only SimCore works well, generative model. CPC was much less predictive. Green line is if you back-propagate from the answer of the question, much more specialized.

Conclusion. Context can be non-local and non linguistic. Can get background knowledge not only from text but from other modalities. Pipeline view of language previously, more realistic view of language processing.

### My Highlights

- Intuition of skip connection in Transformers
- Aggregate knowledge from environment in single model with conceptual understanding. In a single agent.

## Episode 8 - Attention and Memory in Deep Learning

### 01 - Introduction

Memory can be thought of as attention through time. Ability to focus plays a vital role in cognition, example cocktail party problem. Attention is about ignoring things, removing some of the information. What do neural networks have to do with attention? NN learn a form of implicit attention. Can visualize that by looking at the network Jacobian.

Dueling network: Applied to Atari games. Has two headed output, one predicts value of the state the other predicts the action advantage. Video with Jacobian overlay. Network focus on the horizon, also focused on the car itself and the score at the bottom. Left value function, right Jacobian of the action advantage. Flare up just in front of the car that drives. Attention mechanism allows to focus on different parts of the image for different tasks.

RNNs sequential Jacobian, which part is remembered. Analyze how the network responds to input in the sequence it needs for the task. Network does online hand writing recognition, pen position is recorded live. Sequential Jacobian shows the magnitude of the Jacobian, the task is to transcribe the online pen position (it misses the v). Peak of sensitivity around the time of i is written. sensitivity is to the end, why? Suffix ng is very common. The dot of the i is the very last peak, when the person was writing the dot at the end.

Challenge of machine translation, order of words can change between languages. Network can do the rearrangements, that's what the heat-map shows. There is a diagonal line for one to one translation. Reach has peak sensitivity at end of German sentence.

Explicit attention mechanism still has advantages

- efficiency
- scalability (fixed size part of image, can scale to any image size)
- sequential processing (visual example: gaze moving around a static image. Get a sequence of sensory input. Can improve the robustness of the systems, recent paper, more robust to adversarial examples)
- easier to interpret, can analyze more clearly of what the network is looking at, Jacobian is only a guide but not entirely reliable.

Neural Attention models framework

Glimpse factor, non standard terminology. Network makes decision about what to attend to and that decides what to look at. System is recurrent. Glimpse Distribution, softmax to pick a tile. In general we can use RL methods for supervised tasks any time some module in the network is non-differentiable.

Complex glimpses, glimpses at multiple resolutions. Mimick the effect of the human eye. Theory is that the human eye can be alerted by movement in the periphery and then focus the high resolution part in that place. Network has to discover the MNIST data, green shows movement of the focus. Why bother doing that when you can pass the whole image? Scalability, street view can contain multiple numbers and you want to scan through all of them instead of making a simple prediction.



### 02 - Soft Attention

Last examples used hard attention. soft attention can train end to end. Samples made it non-differentiable, we can do mean field approach, take the expectation. Now can use backprop. All we need is a set of weights. don't need proper probability distribution. Its just a sigma unit, $$w$$ look more like network weights. Data dependent dynamic weights, fast weights. Ordinary weights change slowly with gradient descent. Convolution vs. weights. attention has data dependent weights. Define a network on the fly, this is what makes it so powerful.

Handwriting Synthesis, take text and transform it to handwriting. Output is a trajectory. can be seen as sequence to sequence problem. Problem, alignment is unknown. Mechanism is different from normal attention. Decides how far along the slide the gaussian window. One hot vectors are letters, where to put the gaussian.

Alginment shows the interpretability. Early example of location based attention. If you take the attention mechanism away, then you get unconditional writing. Entire sequence fed at once. It looks similar but does not make much sense. Conditioning signal doesnt reach the network, doesn't know what to write at which time.

Associate Attention. This is what has taken over nowadays. Dot product between key and data gives weights. Content based lookup, multi dimensional.

Key and values. Separation between what you use to look up and what you get back. Proper names have been replaced by entities, otherwise they are hard to deal with in NLP.

Attention can also be used to speech recogntion. Aligment discoved between spectogram and text output.

Attention is a very general framework, we are only going to see few examples An example is the draw network. Gaussian filters applied to the image, can also focus on image and is differentiable end to end. 

Video shows attention when classifying digits in MNIST. First it has a broad focus and then becomes very narrow. using attention to trace the stroke of the digits. Transforming a static task to a sequential one.

### 03 - Introspective Attention

Previously attention to external data, now to internal state or memory

Neural Turing machines, read and write to memory. Shows the link between attention to memory nicely. Even if it is feedforward newtwork, its actually recurrent. Heads turing machine vocabulary. Are soft attention mechanism. separate out computation from memory. Normally you need to make the number of parameters larger, which increases the computation as well (if you want more memory). Goal is to separate that out. Like in a computer, small cpu access to large ram.

Selective attention. Selective read is attention, now we can do writing as well

Addressing by content.

Addressing by location. Shift kernel, softmax with a number of plus minus. Data structure and accessors.

Reading and Writing. Reading is standard soft attention. Novel is write head, inspired by LSTM gates forget and input gate. $$e$$ is a set of numbers which are between 0 and 1.

Can it learn a primitive algorithm (NTM Copy). It was able to learn very simple algorithms, copy task. Its difficult for an ordinary NN to do that, they usually mostly do pattern recogntion. White is focus, black is ignored. Was learned end-to-end. Task was not built in the network. You can learn this with a NN but it will not generalize. NTM can do that, although it is not perfect.

Copy N Times

N-Gram inference, given last three inputs there is a set probabilities telling you next input should be zero or one. Baysian algorithm does it optimally. Meta learning algorithm has to be learned. Red arrows indicate errors made by NTM, but it was better than LSTM, uses its memory. It learns to count the occurences, which is what optimal bayesian algorithm does.

Viedo shows system in action. 

Differentiable Neural Computers. Rather than looking at algorithm it was about looking at graphs. Much more data is naturally expressed as graphs. Train with random graphs, test with real graphs and ask questions like shortest path. Questions about familiy tree.

### 04 - Further Topics

Transformer network. They take attention to the logical extreme. Just use attention to repeateadly transform the data. Hence the name attention is all you need. It is mathematically the same as we saw before, rather every vector in the sequence emits its own query. No central control mechanism. Similar to content based attention. Multiple attention heads. Check out the blogpost the annotated transformer for more details. Patterns that show up are very intruiging. Important to attend to elements which are spaced wide out in a sequence. Transformers provide state of the art for language, speech and images. Iteratively predicting the next word in generating text with transformers. repeat the process. It can keep the context for a relatively long time. It names the biologist and keeps his name constant. Attention allows to span very long data, before LSTM would struggle to do that.

Universal Transformers (just one of many extensions) Mixing RNN and transformers. Makes it act like a RNN in depth. It starts behaving algorithmically, good at learning functions like the NTM. Because weights are tyied you can enact it variable times, can become data dependend

Adaptaive computation time (ACT). Ordinary NN is 1 to 1 with input and output. Ties computation time to data time. Idea of ACT is network can learn how long to think how much computation time to spend. Thinks or ponders for variable number of steps. Determined by halting probability, when sum of probabilites is above one then it will stop and give an output. what is the relationship with rest of the lecture? Amount of time thinking is same as time spent giving attention. Variable amount of computation for each prediction it has to make. Time goes up when there is space between words.

ACT with transformers. Task from baby dataset series of sentences are the input or the context. Question is at the end. All the ones which mentions John are inducing longer computation time. Attention is about ignoring thins and being selective.

Summary

### My Highlights

- Memory is attention over time
- Hard vs soft attention. Learn if non-differentiable parts with RL
- Transformers reduced to the essential

## Episode 9 - Generative Adversarial Networks

### 01 - Overview

What are generative models?

We can learn an explicit ditribution from data. Ask how likely is a point or sample from it. Explicit vs. implicit distribution. Implicit means we can sample from the distribution without knowing it directly. Progress has been fast in the past couple of years since the original GAN paper.

Discriminator vs. generator. Generator has latent noise as input. Distribution as an input. Often multivariate gaussian noise, much lower dimensionality. Take sample and pass ith through deterministic NN to generate a sample. Discriminator has the task to distinguish between the two distributions, can think of the Discrimnator as a a teacher. Discriminator is a learned loss functions, it guides and improves itself. In practice we don't have the resources to update the discriminator to optimality. Update discriminator first, and then the generator afterwards. Playing a two player game.

Also think of GANs as distance minimization. Maximum likelihood maximization same as KL minimization. KL gives us nice connection to optimality. Want p(x) as high as possible, ratio as small as possible?

Effectos of the choice of divergence. Model might be misspecificed, maybe distribution such as imagenet is too complex to model. Whant to ask what kind of tradeoff do different models have (if they can't fit full distribution). want to know what happens when we train KL(pstar, p) and KL(p, pstar). Blue is model, red is true distribution

Are GANs doing divergence minimization? yes, see original paper. If D is ptimal, then generator is minimizing jenson shannon divergence. Gives us connection to optimality. So what is Jensen Shannon divergence? It does actually KL and reverse KL. Practice it depends on initialization of the model, can go both ways.

In practice. D not optimal, compute limited, not have access to true distribution.

Properties of KL and JS divergences. No learning signal from KL/JSD divergence if non-verlapping suppor between odel and data. Probability under data distribution will be infinity because p(x) (blue) is zero, KL divergence will be infinity. What happens if model goes closer, green line. But the infinity property still holds, ratio is infinity.

Can we choose another V for min max game? And what will it correspond to? Wasterstein distance! Wasserstein has no ratio, and only does a maximization over one lipschitz functions. Function has to be realtively smooth. Find function f that can sepearate these points as much as possible. Under model f will be negative. Wasserstein distance goes down even if we don't have overlapping support, have restricted amount of growth. Distance has property of we do right thing we get rewared. But how to make it in a GAN? We can turn this into Wasserstein GAN, min max game but Lipschitz norm over D. 

Other divergences and distances: MMD. Different kind of functions to optimize, reproducing kernel hilbert space. Can turn that into a MMD-GAN again. Also f-divergences and variational lower bound. But don't have access on $$p(x)$$ but we can find find variatianl lower bound, replace that in training objective. It tells us to optimize something simliar. Optimal T is density ratio, but that will cause problems. But overall similar objective.

Why train GAN instead of doing divergence minimization? Problem: KL divergence requires knowledge of $$p(x)$$ which we don't have, because we only have implicit distribution. By using GANs we have expanded the class of model we can use to train KL divergence.

Wasserstein distance and computationl intractabilit. Wasserstein is computationally intracable for complex cases. Would not be able to do that at each iteration step. But WGAN has same algorithmic implementation as original GAN. Its not exact but inspired by Wasserstein distance. Problem with smooth learning signal, is it really the case that there is no signal if there is no overlap? GANs only approximate the ratio, so it will never be infnity. So in practice model still learns. Why? Look at true ratio which is problematic, but when we train GANs we use lower bound because we don't have access. So the ratio is only estimated, and has to be in a certain class of functions. These functions are relatively smooth, and the won't be able to jump from zero to infinity such as the true ratio would. So we get a smooth learning signal in practice.

Crucial idea: D is smooth approximation to the decision boundery of the underlying divergence. In practice GANs do not do divergence, hence they do not fail that hard. Think of discriminator as learned distances. We use NN features also in the loss.

GANs vs divergence minimization. Take home message: practice not divergence minimization, learned distance.

Empirically, underlying loss matters less than neural architectures, training regime and data.

Onconditional and conditional models. Up until now only unconditional data generation.

Now want to provide additional information, often in form of one hot vector, to specify class. Conditional label also fed to the discriminator. (Use that for domain adaptation GANs?)

There can be mode collapse. G loss is not very easily interpretable, does not tell us much. Hence how can we evaluate GANs?

### 02 - Evaluating GANs

No evaluation metric is able to capture all desired properties.

- Sample quality
- Generalize
- Representation learning

have to evaluate on end goal

- semi supervised learning: classification accuracy
- reinforcement learning: agent reward
- data generation: user evaluation

But this is hard to do. Log likelihoods are not available (can't do that) because model is implicit distribution

Incepton score: Model provides ratio of classes which stays the same. Dropping classes will be penalized. Does not measure anyting beyond class labels

Frechet Inception Distance: Looks at label distribution and diversity inside the class. Looking at features in a pretrained classifier. Correlates with human evaluation. Biased for small number as samples.

Check overfitting: nearest nighbours. Want to find closest images in feature space. Use pre trained classifier. We see that exact same dog does not exist.

Take home: need multiple metrics to evaluate

### 03 - The GAN Zoo



#### **3.1 Image Synthesis with GANs: MNIST to ImageNet**

Original GAN paper: simple data, small images. Not convolutional, only MLP. Works for digits. Mostly proof of concept

Conditional GANs: generalize to the conditional setting. Category ID or image from another ID

Laplacian GANs: starts from tiny image and upsample via gaussian, generate Laplacian to fill in details and add it up. Discriminator takes two input images. Discriminator and generator are conditional. First to produce relatively large images with decent results. Was fully convolutional operator.

Deep Convolutional GANs: simple architecture, used BN. Made training much easier. Can do interpolation between to noise vectors in z space. Shows model is able to generalize, continuous distribution and not simply memorizing. Meaningful semantics in latent space, man plus glasses minus men plus women is women with glasses. Simliar to word2vec results.

Spectrally Normalised GANs: First real try to do imagenet generation. No matter what input to layer, output values are not increased due to clamping singluar values. Regularizes discriminator. Uses hinge loss basically.

Projection Discriinator. Learn a class embedding. Interesting theoretical justification for that projection

Self-Attention GAN: Do global reasoning, used in language domain normally. In image domain gives you the opportunity to learn global statistics about the image. Better global coherence in generated image. Can say where the model is looking at.

BigGAN: Main idea is to make GAN really really big, digest all the previous work and scale it up. Big Batch sizes are really important, use 2048 instead of 256. Was important because imagenet has 1000 classes, so each class should be in each batch. Also trained on JFT. A lot of tricks from previous papers, but lots of ablations studies. new tircks. Orthogonal regularisation, skip connections from noise, class elabel embedding shared accross layer fro class conditioning. Truncation trick was introduced, change standard devation of noise input. The smaller the variance, the more simliar samples become, givs you prototypical example of one class. Trick between variety and fidelity of the samples you can generate. Architecture trick which improved performance. Example of class leakage. 

LOGAN: latent optimisation. Natural gradient descent step inside 

Progressive GANs: parallel line of work compared to BigGAN. Start generating at low resolution and wait until convergence, then add an umpsampling level plus upsampling. Went up to 1024 to 1024. Great results on faces.

StyleGAN: follow up, used more challenging data set. More diverse and not so famous people. Had stracured latent inputs z, and had spatial noise input. spacial vs global latent. Two kind of layers. Can have stochastic varations at different scales.

Takeaways: still not easy to do. and methods have weaknesses

#### **3.2 GANs for Representation Learning**

Motivating example: DCGAN latent space without being explicitly told about it.

BigGAN: Associates with high level image categories. Goal: model learns to associate latent space with a label.

InfoGANs: Adds inference network, force the generator to use each latent variable meaningfully. Only works for small categories

ALI or Bidirectioanl GANs: Jointly learn to generate data and learn representations from it. Two directions, generator and predictor. Discriminator sees both places. What is the enocders job here? there is a global optimum here, if there is perfect discriminator then the generator and encoder have to invert each other. that would be the global optimum. In autoencoder you explicitly minimize this objective, but here during training time the encoder and generator never see each others results. Its all trough the discriminator. Encoder never suffers from domain shift issues, because it never sees degenerate images from generator, it only ever sees real data.

BigBiGANs: scales up the previous work: Reconstruction is pretty simliar even at high level. Lots of semantic properties are maintained. All this happens because the structure of the discriminator is shaping an implicit reconstruction error metric in semantic ways. Discriminator is convolutional network, so it is good at extracting semantic information. Enforaces semantics staying the same. Has fuzzy semantic memory. Get something similar to SOTA of self supervised images. NN tends to be very semnatically relevant.

#### **3.3 GANs for Other Modalities & Problems**

Pix2Pix: Paired images. Translate between two different domains. Have L1 reconstruction error to tell what exactly the output should be

CycleGAN: Non-paired examples. Unsupervised translate between two different domains. Enforce cycle consistency next to all the GANs. Can translate between any two domains which have somewhat similar content.

GANs for Audio Synthesis: WaveGAN, produce raw audio wave forms. Produce 1 sec clips. Text to speech for speech synthesis. 

GANs for Video syntehsis and prediction: Time makes the problem quite harder than for images. Also due to compute requirements. Decompose discriminator into two different discriminators. Spacial discriminator few individual frames, ensures that each frame rooks coherent indepentely. Temproal discriminator sees downsamples images and ensues fluidity over time.

GANs everywhere: RL, Image Editing, program synthesis. Motion transfer: everybody dance now, map dancing movements. Domain Adaptation. Art

#### **3.4 More GANs at DeepMind**

### My Highlights

- KL vs reverse KL

## Episode 10 - Unsupervised Representation Learning

[download](https://storage.googleapis.com/deepmind-media/UCLxDeepMind_2020/L10%20-%20UCLxDeepMind%20DL2020.pdf) for more

### 01 - What is unsupervised learning

Birds eye view. Supervised learning: Learn mapping from given inputs to given outputs, that hopefully generalizes. RL: which actions to take at any state to maximize expected future rewards, only gets feedback once it finishes the task, unless supervised supervised learning reward is sparse. Unsupervised learning: No teaching signal, only have input data.

Do we need unsupervised learning?

What can we do with that data? Clustering, dimensionality reduction, find dimensions which explain most of the variance (sounds like PCA)

How do we evaluate it?

No ground truth to compare it to. Clusters, how do we know which clusters are good? So many possibilities. How do we know which choices are good or bad. Simliar, pca and ICA orthogonality

### 02 - Why is it important?

How necessary is it. Why not develop better supervised algorithms instead?

History of representation learning. 1949 first mention of machine learning. Kernel methods hugely influential until 2006, hinton introduced unsupervised pre training then with deep networks. 2012 AlexNet makes pre training unnecessary. Then from 2012 more data, deeper models, better hardware made huge leaps in many fields. Is machine learning solved? Data efficieny is bad compared to human learning. Takes orders of magnitude more time than humans to play atari. Often data is not readily available. Another issue is robustness, there are adversarial attacks, pandas can be tranlsated to gibbon . Can lead to huge issues in self driving cars. Generalization, once you learn to play a game, it should not matter what the background color is. Two state of the art RL algorithms fail, inability to grasp the core idea of the game. Transfer, re use previously acquired knowledge. Current state of the art models are not able to solve that if you change atari game slightly. All of the algorithms don't know causality, intuitive physics and abstract concepts.

Solve many tasks efficiently is trending. We know how to solve single tasks with end to end deep learning given update and compute. For the next generation which is efficient, robust and generalizable for multiple tasks, we need a new paradigm. Leading minds think unsupervised learning is the solution. Represent the world before knowing a task. We won't need as much data anymore.

Add representation between task and AI, flat are hierarchical? Once or learn it again. What would it look like?

### 03 - What makes a good representation?

Ill defined and challenging question. Look to related disciplines for inspiration, for example neuro-science.

Def what is representation: Formal system for making explicit certain entities or types of information, together with a specification of how the system does this. Three ways of writing 37, representational from is orthogonal to information content. It is a useful abstraction and makes different types of computations more efficient. Think about the full form of the manifold.

What happens in the brain when information moves in it. Becomes a manifold when you rotate or change perspective. Makes it hard to untangle different objects. Ventral stream transform reformats into better form for that task. Untangle representations.

How cross a street? What other properties should representations have. Approach from RL perspective. Think of representations as states which form an MDP, can shed light on what properties representations should have. Two MDPs, should contain information about presence of cars. What information to include in representation?

Solving tasks requires. Want to exclude all irrelevant information for task. Should support attention to remove unimportant details. Allows for better clustering. Want to pull a taxi, need a different representation than going home task. Want to keep the information about the color of the car. hence latent representation should have as input the task we are trying to solve, to support the downstream tasks.

Compositinality. Ability to infer the meaning of a complex expression. Man with binoculars example, who has the binoculars. Important because it leads to open-endedness.

Evidence from neurosience: list

Another related discipline: Physics. It constrains the form of tasks take. Can we bias a  representations to reflect certain fundamental pysical propertties to make it useful. There exists fundamental idea in pysics, that is the ideas of symmetries. Pysics is the study of symmetry. Example of spring. Can go forward in tame or translate in space, commutative property of two transformations. means that one is the symmetry of the other. Unify existing theories in physics. Same idea applies in natrual tasks, 3D scenes can be changed scale of an object or its positions, and the two tranformations can be interchanged without changing the final state. Can be seen as symmetries of each other. Representation should reflect symmetries. What ideas from ML tool box can be used.

Information bottleneck. Helps analyze Deep nets. Find maximally compressed mapping that preserves infromation as much as possible. Data processing inequality by shannon. Goal of layerwise processing is to discard any infromation whcih is not necessary.

Invaraince vs. equivaraince. Invaraince key idea behind CNNs. 

Example is disentagled representation learing. No general accepted definition of this term exists, but one common example is unding the generative process and do the inference process. Closely related to untagneling in neurosciene.

Group theory, want to caputre a set of symmetry transformations, want to capture this in the representations. For example horizontal and vertical translations and changes in color in a grid world. They affect the state of the world. Assume there is generative process that maps state to map. Goal is to learn a mapping from observation o to representation z, such that f is an equivaraint map (definition of equaivariant map). If we can find such a map f, then our representation is said to be reflective of the underlying symmetry transformations. 

### 04 - Evaluating the merit of a representation

How can we verify that we are on the correct path? Evaluating representations. Representations should adress the follwing:

- Symmetry in definition
- Untangled 
- Should be compositional
- Can implement attention in for example binary mask
- Clostering: need metric, can assume it is in a vectorspace

Representation should help with shortcomings

- Data efficiency: Majority of natural task. Color transformation is easily learned by linear mapping. Research shows that incorporating symmetry does help in supervised task (see paper)
- Robustness: Mapping f needs to be equivariant. Functional form quite constrained, help it might be more robust to adversarial attacks. Example on slide, CVPR 2020 paper, includes attention
- Generalization: can be increased if the decision on which action to take can be made without those aspects of representation that are not imporant of to the task. Since our symmetry based representation of Z preserves the important information about the stable course of the world in a format that allows for fast attention attenuation, we quickly adapt the minimal set of informative subspaces available to the decision network when faced with solving diverse tasks. Thus increasing the generalization of such decision networks.
- Transformer: Mapping f connects the underlying symmetry transformation to the representation, it should not care about the nature of the intermediate observation. Was shown that the schema networks could transfer its ability to breakout much better than the unstructured deep RL baseline.
- "Common sense": Least explored area of machine learning, preliminary evidence suggests that our hypothesized representations may support compositional abstract imagination and maybe a solution for grounding many promising discreet or symbol based algorithms. They have concept induction and abstract reasoning.

Currently no algorithm exists that can learn such symmetry equivariant representations in a robust and scalable manner. Aiming for such representations may be a good research direction.

Recap:

Deep learning successes may be due to its ability to implicitly learn good representations.

### 05 - Techniques and applications

Three main pillars to achieve outlined goals.

Generative modeling: modeling underlying distribution

Contrastive losses: classification losses to learn representations that preserve temporal or spatial data

Self-supervision: Information about data modality, images audio. learn representation that preserve data.

Downstream tasks: semi-supervised learning, reinforcement learning, model analysis. First learn from unsupervised data only, then ask questions about the representations learned. Build downstream tasks to asses what kind of information is in there. How much information about the label is still present, often by building a simple classifier, linear layer. have in mind data efficiency and want to be able to generalize. Can also ask additional questions. First train on ImageNet without any label information, then use small percentage to train classifier to see how well representation are doing. Allows to compare different representations.

RL: Agents in multiple different environment, or learn from different experience. Tasks that are very hard to learn from online data. Learning disentangled representation can speed things up

Model analysis. Understand what the model is doing, do they satisfy the property. Learning interpretable models, want to see what it is learning before deploying it in production.

Keep in mind what we want, discrete and continuous representation (face has glasses is binary, haircolor is continuous). Representations adapt with experience, continual learning, RL). Consistency in data should be represented, example a scene, from different angle.

#### **Generative Modeling**

What kind of distribution could have generated our dataset? Main question. Example mixture of two Gaussians. Learning probability distributions efficiently has a lot of connections with compressing data. Want representations that are efficient and compressed.

Latent variable models: Mapping from low dimensional to high dimensional space. Being able to model the sampling process, assume generative process looks like that. Assume that they are generated by very complicated mapping (NN). Inference p(z|x). In practice learn inference and generation together. Don't have access to true distribution of z.

Variational autoencoders. Uses maximum likelihood, can't see p*(x) directly but use Monte Carlo simulation. Challenge is latent variable and train them with MLE, it is given by an integral (describes the integral, what it does). p(z) is a prior. We use a lower bound, ELBO, lower bound on MLE objective. Optimize bound, make it as close as possbile to the real objective. Difference between EM and MLE. For complex models we cannot calculate the posterior exactly, this makes the difference between EM and variational inference. In EM the inequality becomes equality. VAE as close as possible to log probability. First term is likelihood term, encode in latent variable as much information as possible, assign high probability to orogianl x we have seen, only possible if we encoded original information efficiently. Second term, want representation to be close to prior. 

VAE. prior we choose it. Can choose our prior to be disentagled, example gaussian with independent diemnsions. KL will regularize it.

VAE and NN. Both inference and generation model are deep neural networks.

KL term is important for regularizations, to force disentangeled representations. No longer exact bound, but model really wants to learn disentangeled representations. beta-VAE. This is called latent traversal. All six are fixed except for one. (6th and 7th latent variable.) See what changes in the scene if you change one variable. Entangeled is non betaVAE.

Downstream tasks, evaluate how good representations are for transfer learning and generalization. Allows RL learnings to transfer quicker form simulation to reality.

Sequential VAE - ConvDraw, car example, first draw the outline then give more details. Have a recurrent component. Move from high level outline to final detailed  image. Can have posterior distributions that are autoregressive and way more complex, can be closer to true posteriror. Can get closer to the bound.

Layered models - Monet. VAE and segmentation network. beta VAE and attention netowrk to segment objects in an unsupervised way. Masked input after first attention network is inputed to the VAE. Monet and monet in reinforcement learning.

Generative Query Networks (GQN) look at consistency property. Information about how scene would look lke from different angles. Provide data from scene from different angle. Multiple generation steps. Model has to learn to draw from different angle. 

GQN can capture uncertainty. It is able to imagine that there are multiple objects behind a wall. We want representations to encode uncertainty about the world.

GQN in RL: less varaince in learning, and even being able to learn the task and not being able to learn the task at all.

Vector quantized VAEs: learning iscrete latent variables is challenging. start with continuous vector, look into learned table of embedding and look for NN in that table, and the index will give the discrete varialbe. Now we are able to learn with discrete latent spaces. can get very good compression algorithm, but reconstruction are a bit blurry, because we are using probability model.

GAN: (basic structure): GANs can t answer the inference qustion.

BigBiGan can do that, learn to encode by changing the adversarial game. Crucial how discriminator changes. Want to go beyond that. want to match data and latent variables in prior. And invert from latent to other model. (see paper from Donahue). Marginals will be matched, latent variable distributions will be matched as in the VAE case, and we matched the relationship between xhat and x and z to zhat. No use of reconstruction loss. Learn how to reconstruct and.

No pixelbased loss, captures high level information.

GPT: useful for downstream tasks. Use very well tuned neural architecture with large amount of data.

Contrastive larning, completely unsupervsied, removes need for generative model. Use classification loss instead, built from unsupervised data. Done so that the right context is encoded.

Contrastive losses: word2vec. One hot represents no semantic information whatsoever, no relationship between words. But how can we do that? By learning a model which predicts the representation it should expect from past data. Provide positive and negative examples in training, predict next word to expect. Want to test it how? Unsupervised learning in english and spanis. Then learn simple linear mapping from few data points. Is this mapping generalizing? Can we do dictionary translation. 

Conrastive predicitve codeing. Maximize mutial information between data and learnind representations. Learns what we have seen so far. Think of the idea as temporal coherence structure. Can also be used for spatial data such as images, different patches. Do very well when we don't have many labels.

SimCLR another contrastive loss idea. If you transform the image a little bit, then you still want to have same representation after transformation f. Should contain most of the information, but it should not be fully same information. so we add a different mapping g we then obtain a mapping which is the same. Downstream task uses f results to do things. This is shown for images. Have a nice plot where you compare for number of parameters in models.

#### Self supervised learning

Colorful image colorization. Needs no label, and ask model to revert the mapping. Use this again for representation learning. We can do context prediction, learn spatial consistency. Given a patch, which one do you think the other patch is? It only has to do 8 way classification, but it has to understand and learn representations that are useful for semi supervised learning. We can also go and look at sequences, shuffle images and let the model sort the images as in the original sequence. it has not to know how the next image looks neither has to predict it. learn temporal coherence. One example is BERT, leverages tasks that learn local and global structure. Leanrs which words have been masked in a sentence. Given sentence A and sentence B which one will come first? this is long term structure not such as the other task. Also uses bidirectional models as transformer. Put in production as part of google search.

Keep in mind that:

- task desing is important
- modality
- context
- learning generative models is hard, maybe able to get awawy without it
- Benefits by incorporating changes in neural architectures.

### 06 - Future

- Generative models: powerful posteriors (autoregressive) and better priors (disentanglement)
- Contrastive learning
- Self supervised learning: more task design by exploiting information about modality
- Incorporating changes in neural representations. DL will advance and use these improvements
- Causality, causal coherence and have the representations that are able to answer these tasks.

### My Highlights

- EM and VAE comparison

## Episode 11 - Modern Latent Variable Models and Variational Inference

### 01 - Generative Modeling

What are they? think of them as mechanisms of generating more data points. Key distinction, modeling distributions are really high dimensional, not like classification. Often there is no input. Has been seen as part of unsupervised learning. Many type of models, can handle any kind of data

Uses of genrative models

- From statistis, density estimatiion and outlier detection, given data point estimate its likelihood, fraud detection is possbile
- Data compression, duality between these two areas, can use arithmetic coding to do data compressor
- Mappring rom one domain to another. Sentences in different languages, multiple translations possible
- Model based reinfrocement learning, model acts as probabilitc siulato of environment, algorithms can use simulator to plan optimal seuence of acttions
- Representation learning, condense obeservations into low dimesnional irepresnetation which caputre the essence, might be more useful for downstream taks.
- Understanding the data (also from statistics), laten variabels will be interpretable, or will have real world significance.

Progress in generative models, samples from particular years. 2014 mnist, 2015

2019 model much higher dimensional images

Types of genrative models

- Autoregressive models: language modeling, RNN or transformers
- Latent variable models
  - Tractable
  - Intractable
- Implici models: GANs and their variants

Autoregressive models. Solve the problem. Can use off teh shelf classifier technology. No random sampling at training time. Sampling from such models is sequential and slow, can not parallelize, better on local sructure than global structure.

Latent variable models: also likelihood based, different approach to joint distribution. Trained useing ML or some approximation to it. Powerful and well understood, easy to incorporate prior knowledge and structure, fast generation. Conceptually complex, require understanding inference, opposing of generation. Inference is intractable, either restrict the models we can use or introduce additional complexity to use approximation

GANs: Not likelihood based, implicit models, just give you samples. Trained using adversarial training rather than ML, train a classifier, by far the best ones to create images. Provide fast generation, simple a forward pass in NN. Don't give us ability to assign probabilites to obesrvation, no outlier or lossless compression, suffer from mode collapse, models only subset of data sometimes, training can be unstable.

### 02 - Latent Variable Models & Inference

Latent variable models, defines. prior and likelihood. z can be vector or tensor, makes no difference. model is completely specified by joint likelihood. Two observations are of interest to us, marginal liqkelihood and posterior. plausible values which could have generated x. Generate by sampling from z, then sample x from p(x|z). Much of the lecture is about inference.

Inference

compute posterior given observation. need to calculate p(x) by doing the integral, integrate over z. 

Inference is the inverse of generation, in a specific formal sense. two ways to generate observations. generate paris in two ways. Disribution of these cases is exactly the same as

$$p(x|z)p(z) = p(x,z) = p(z|x)p(x)$$

Inference probabilitc inverse of generation.

Why is inference important? Explaining obsercations. Learning, comes up naturally in training, subproblem which needs to be solved.

Inference for a mixture of gaussians (simplest model). compute posterior ditsribution in linear time number of latent variable values.

ML learning: dominant estimation principle for probabilitc models. choose those parameters that make training data most probable. For latent variable models we can't solve this in closed form, either use gradient descent or expecation maximization

Look at gradient of the marginal log likelihood. Look at transition more closely(not that clear). Need to compute the posterior distribution to compute the graident. this means inference performs credit assignments among latent configurations for given observations.

Exact inference is hard in general. Integrating over high dimensional space of a non-linear function, analytical and numerical integration is not an option. For discrete cases summing over finaite number of values, but exponentially many latent configurations, curse of dimensionality, never be able to compute sum exactly. Exceptions: mixture models, linear-gaussian modeles, all induced distributions are gaussian. Inverteble models are special because they are powerful and allow for exact inference through constraints on their structure.

How to avoid intractable inference

1. Designing models so that they are tractable, gets us less expressive models
2. Using approximate inference, more flexible, more expressive, will end up with intractable model but the aprox inference allows for optimization



### 03 - Invertible Models & Exact Inference

Invertible models / normalizing flows. High expressive powers combined with tractablility. All the parameters are in the invertible function.

Invertible model, need prior and then use invertible transformable . All parameters are in f, gives us one to one correspondence between latent and observations. Inference is fully deterministic

How do to compute the marginal likelihood? Need to relate prioer and, can apply change of varaibles. Would like to get rid of z by replacing it with inverse of f(x)

Indepenen component Analysis, simplest and oldest inverible model. Solve cocktail party problem, n microphones in room and n people, recover sources z, prior can not be gaussian because they are rotaitnoally symmetric in high dimensions, need heavy tailed distributions.

Constructing general invertible models Chain a bunch of invertible transformations together, don't need f to be analytically invertiable, as long as the algorithm is efficinet. List of invertible builidng blocks

Limiations of invertible models. Limitations

- Dimensionality of latent space and data space has to be the same
- Latent space has to be continuous, although they are discrete flows research
- Hard to model discrete data
- Expressiveness by number of parameters is not great, needs lot of memory
- Hard to incorporate structure because we need invertibility

They do make good building blocks for larger latent models.

### 04 - Variational Inference

The appeal of intractable models. Quote from David Blei, often we go with the wrong anwer to the right question, approximate inference.

Example: ICA variations, how quick we can go intractable. Supoose we add some noise, makes model intractable. or change the number of dimensions.

Approximate inference: Markov chain monte carlo, generate samples from exact posterior using markov chain. Very general method, exact in limit if we have infinite computation, convergence is hard to diagnose.

Apprioximate inference: variational inference. Approximate the posrior wit a tractable distribution, fully factorized or autoregressive. Much more efficient than Monte Carlo, cannot trade computation for greater accuracy. Can theoretically quantify approximation error.

Variational Inference. Why variation? Optimizing over a space of distributions, we are approximating unknown posterior distribution, we will approximate the varational posterior. Often use fully facorized distribution.

Training with variational inference. Allows us to train models by approximating marginal log likelihood which is intractable, but by introuducing alternative objectibe which is lower bound we can optimize wr.t. lower bound.

How to obbtain the marginal log-likelihood. Implement formula here

Will focus on ELBO. IWAE allows for trade off between computation and apporximation.

Review KL Divergence.

Fitting the variational posterior. 

Training the model. What happens whe we update the parameters?

Two ways of decereasing the variational gap. Can update the variational paramers, can update the model parameters. Can spend some of model capcaity to approximate posterior instead of approximate data. Should use most expressive varitional posterior we can to diminish that effect.

Variational pruning. Model refuses to use some hidden variables. When you prune out variables it becomes easier to perform variational inference, extra pressure on model to be simpler. Also known as posterior collapse, good or bad thing? Can be good choosing dimensionality of latent space based on data. Or bad if takesaway some of the freedom to overfit to data. Model can refuse to use extra varaibles if we make latent space really big, can lead to suboptimal results even though we give enough options.

Choosing the form of the variational posterior. Default option is fully factorized distribution, called mean filed apprximation from physiscs.

More expressive posteriors:

- Mixture distrbution
- Gaussian with non-diagonal covatiance
- autoregressive
- flow-based, use an invertible model which is tractable

Ultimately a trade-off between quality and computational cost.

Amortized variational inference. Posteriro distrubiton is different for each observation x. Do inference network, functional approximation. Phi is the network parameters, populariased by variational autoencoders. Allows us to easily scale up variational inference. Variational parameters are jointly trained.

Variational vs. exact inference. Can train intractable models in efficient way, inference is fast compared to MCMC. But we can give up some model capacity and lead to suboptimal performance.

### 05 - Gradient Estimation in VI

Maximizing the ELBO. ELBO is an expectation. Classical was expectation with closed form, objective function was analytically tractable. Models had to be simple, and variational posteriors need to be fully factorized. Applicable only to a small set of models. Here we use monte carlo sampling which allows us to handle any kind of latent variable model.

Graident w.r.t. the model parameters. Easy case, move gradient inside expectation. In practice only one sample can be enough to train model. Noise in gradiens can be bad thing, we need to use low learning rate, can use more samples for hihger learniner rate.

Graient wr.r.t. the vriational parameters. Cant take gradient inside the expectation. Is reasearch problem

Gradients of expctations.

- REINFORCE / likelihood-ratio estimator. Very geeral but high variance
- Reparametricazaton, pathwise estimator. Less general, need continuous latent variables, needs f(z) to be differentiable. Gives you fairly low variance

Reparametrization trick. Transformation of samples from fixed distribution epslion, to get z. Example with reparametrizing gaussian variable.

Reparametrizing distributions. Not every distribution can be reparametrized in a differntiable way. Can not do that with discrete distribution, can do implicit reparametrization. Modern frameworks such as TensorFlow and Pytorch support reparametrization for many continous distribution

### 06 - Variational Autoencoders

Most successfull application of variational inference in the last years. Generative model swith gontiuous latent variables. Was a breaktrhough in 2014, made them very popular. Highly scalable and very expressive.

Variational autoencoders. Decoder / Likelihood. Variational posterior / encoder

ELBO decomposition for VAEs, slightly different write up as before. second term often computed in closed form. First term measures how well to construct. Second term ascts as regularizes, pushing variational posterior towards prior. Computed analytically. 

The VAE framework, model with continous latent variables and reparametrizaton trick and amortized variational inference.

VAE have been improved in many ways.

Conclusion

- Two modern approaches, based on likelihood
- Different tradeows between flexibility/power and ease of infernce
- Models can be combined.
- Many substential contributions to be made

## Episode 12 - Responsible Innovation & Artificial Intelligence

### 01 - Motivation

The power and potential of artifiical intelligence

Risks

Failure modes of machine learning. Intriguing properties of neural networks paper. Adversarial attacks basically

Studies on GPT-2 has gender bias

What are our responsibilities as machine learning practicioners? Open questions We are responsible for enuring our neural networks satsify. Have quality control on deployment

How can we make sure algorithms are safe for deployment? Satify law of physics. Robust to feature changes that are irrelevant. Uncertain if it has images out of distribution.

### 02 - Specification Driven ML

What is specification driven macine learning? Limited data model can learn spurious prediction. How can we enforce these specifications

### 03 - Adversarial and Verification Algorithms

Robustness to adversarial perturbations. Important for applications which have real adversaries in the mix. Unchanged under an additive perturbations

Adversarial robustness specification. Function with one hot encoded vector. delta denotes perturbation. We want our nn output to be correct under perturbations. Second line want this to be true under all perturbations under a specific norm.

Adversarial Training. Similar to standard image classification training. Adversarial training is similar but with data augmentation step, want this to be labelled correctly as wel. But we can not iterate over all these perturpbations. So we want to find the worst case.

Want to maximize the difference between the prediction and the class. New objective, now it is a min max problem. Want to find the max for the perturbations, now we want to minimize in the outer loop. Adversarial training is significantly more expensive than standard classification training.

#### 3.1 Advesarial Evaluation: Finding the worst case

Adversarial Evaluation / Attacks, worst case accuracy is known as adversarial accuracy. Complications, find maximum exactly is NP hard. Constrained optimization problem because delta is constrained in set B(epsilon). Cant find maximum exactly but can do gradient ascent to approximate it. Hence to projected gradient ascent, yellow box, project back onto limits of it.

Update step: delta is projection of gradient ascent step, definition of proj(delta)

Fast Gradient Sign method (iterated). Can replace the gradient with any alterations. Explore parameters to have strongest evaluation possible.

Strenghts of Advesarial evaluation

- Adversrial acuracy is dependen on your choice of evaluation. The strong the evaluation is the lower the accuracy will be
- Should always aim for lowest adversarial accuracy, closest to the true adversarial 
- Heuristics
  - Steps: The more steps the closer you are on maximizing objective
  - Number of random initialistions of perturbations, start from a number of different initializations. Graident obfuscation??
  - The optimizer is important, try a few different ones to have lowest
  - Use black box adveraarial evaluation. Assume you are not given the weights of the network, the opposite is the whitebox

Danges of weak adversarial evauation. Two papers in 2018, weak evalation can give you false sense of security, need adversarial training to be robust in first paper.

Stronger adversarial evaluation gives better evaluation progress. Another paper. Reported numbers and the updated evaluation

#### 3.2 Gradient Obfuscation

What is it? Look at objective has outer minimization and inner maximization step, focus on inner maximization. Conundrum, more steps we take, better accuracy but also more expensive. How to make it cheaper? Fewer steps of gradient ascent, if you take to few steps network learns to cheat by making high nonlinear loss surface. Example for gradient obfuscated surface If you do adversarial training correctly, you expected a much smoother loss surface, on the right, compared to what you get on the left.

#### 3.3 Verification Algorithms

Profable guarantee that no algorithm will ever change your specification.

Verification algorithms: Complete, exhaustive proof or counter example, which are very difficult to scale. Incomplete verification algorithms, a proof can not always be found, even if NN satisfied the specification, they give you a lower bound on specification satisifaction.

Verification fo specifications, y = f(x). Make two assumptions, input comes from bounded set denoted by X, and NN consists of linear and activation layers. Get output set captial Y, does it lie on one side of decision boundary? NP hard problem to find propagation the boundaries.

Incomplete findes boundaries which are easier to propagte, but makes the result less precise. If approximation is too lose, then the incomplete verification can not mean very much sometimes. Example of Y satisifes the specification, and if overapproximated set spans both sides of decision boundary we can not say much.

Graph: Difference between empirical and incomplete verification. X size of input set, y amount of specification viaolation. Solid line is upper bound on  violation, dotted line is lower bound. True value is in between, gap needs to be small to make a conclusion.

Other specifications. 

### 04 - Ethics and Technology

What is ethics and why does it matter? How does it connect with machine learning

Ethics is a field of inquery which is concerned with identifiying the right course of action with what we ought to do. Converned with equal value and imporance of human life and with understandig what it means to live well in a way that does not harm other human beings, sentient life or the natural world. Challenges can arrise in everyday life but also in machine learning research. Technologiests and reasearchers are making ethical decisions all the time.

Start with training data, data is not only a resource but has ethical properties. For example data been collected with consent of those in the data? Celecbrities or images taken from image. Representation, is it diverse or overfocued on certain groups of people? How is the label collected and curated, may contain predjuices.

Algorithmic bias. Software used to make decision has often the bias of its creators. Example criminal justice where a system discriminated against black people. Another issue with job search tools, preferring men over women by significant ratio. Image recontion works less well for minorities. Medical Diagnosis works less well for non-binary people?

Power and responsibility. Have significant impact on the lives of others. Responsibility to what? What can we do when building ML systems?

Science and Value. What does it mean to do machine learning well? Research is not value neutral, social practice shared norms change over time. 

Responsible innovation. Appropriate standards and norms for research. Unique challenge. Good science is algined with democratic processes and based upon the alginment with social good. What precisely are researchers responsible for?

### 05 - Principles and Processes

The responsibility of technologists. Intrinsic features. Extrinsic factors. Both elements are necessary

The AI Ethics Landscape.

Key Values,

- Fairness privacy and transparency non malfesence
- Individual rights, informed consetn and equal recongiton before the law
- Everyone should benefit from science. All of humanity should share the progress

How do we move to clear processes from these abstract ideas? How do you balance different ethical principles? Lot of research is highly theoretical.

#### A five step process

1. Does the technology have socially beneficial uses? Is there are a reason to develop it? If it is unclear it is a red flag. 
   1. Well-being
   2. Autonomy - empower people, useful information
   3. Justice - produce fairer outcomes
   4. Public institutions - health care or education, global challenges
   5. Global Challenges - Climate warming
2. Risk of direct or indirect harm? Most technologies have some risk, map them out
   1. Undermine health or well-being or human dignity - mental health
   2. Restrict freedom or autonomy - addictive content
   3. Lead to unfair treatment or outcomes - algorithmic bias
   4. Harm public institutions or civic culture
   5. Infringe on human rights
3. Is it possible to mitigate these risks? Are there steps in place to do that
   1. Control the release of technologie or the flow of information - fall in the hands of wrong people
   2. Adopt technical solutions and countermeasures
   3. Help the public understand new technologies
   4. Seek out policy solutions and legal frameworks to contain the risk
4. Evaluate stage, violate a red line? Moral constraint
   1. Consent - infringe on peoples personal space without consent
   2. Weapons - lethal autonomous weapns, delegate respnsiblity to machines
   3. Surveillance - corroding effect on public trust
   4. International law and human rights 
5. If we haven't hit any constraint, does the benefit outweigh the risk? Also consider other options to us

Two final tests

1. Have you thought about all the people affected by your decisions. Have you sought out their input? They have a right to be included
2. Might you have reson to regret it later? Someone in the future, even our children might ask us why we acted the way we did? Technology used to violate human rights. Act in ways to minimize the future regret.



### 05 - The path ahead

Key ideas. Those who design and evelop technologies ahve a responsiblity to think about how they will be used.

There are concrete steps and process taht we can put in place to make sure this responsiblity is succesfully discharged

We are responsbile for what we can reasonably forsee and shoudl take the steps to bring about psotive outcomes. Even if it means incurring certain costs.

New directions

- What is the path ahead? Focus on AI saftey fairness and accountability. 
- New norms and standards what it means to do research well, for the right reasons and in the right way.
- Practice, emergence of new practices . Model cards of intended usage of models. Bias bounties. 