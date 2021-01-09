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

Published just couple of years ago, large impact on the field. Any problem that requires a model to process a sentence, or multiple sentences and computes behavior or prediction on that. State of the art for these predictions. See the paper [attention is all you need](https://arxiv.org/abs/1706.03762).

#### **First layer**

distributed representation of words. First we have to define vocabulary, chop up the units the network is going to see. Could go down to pixels, can either be character level or word level. Split input according to white space in text. A model that just takes symbols might not be optimal as seen in previous section. First consider all the words which the model needs to be aware of. Parse each of the words through an input layer. Put an activation of one corresponding to one word, turns on the weights of "the"? Basically describes how you vectorize the input. Embedding dimension of $$D$$. Model can move the embedding during training. For $$V$$ words and $$D$$ dimension of embedding gives you $$V \times D$$ weights. Words with similar meaning or synthax (noun, verb, subject) cluster together in space.

#### **Self-attention over word input embeddings**

Query matrix $$W_q$$. Key matrix $$W_K$$ , value matrix $$W_V$$. Applied to each input. End up with three vectors per input vector, $$q, k, v$$. Take the dot product, how strong should the connection between the words be. Key value inner product query value, normalize with softmax layer. How query correspond to key in the input. Gives us a probability distribution, tells us to what extent is there an interaction. Beetle relates strongly to drove. Use it to know how much of the value should be propagated where through the network. Insert weighted sum.

Number of embeddings has not been changed, but the information in each embedding has been updated according on how to it interacts with other representations.

#### **Multi head self-attention**

Previous process can be parametrized by these three matrices. We have multiple of the operations applied in parallel, here four times. Four sets of three matrices. Can use a lot of memory, so reduce dimensionality, go from 100 in input to 25, we do that 4 times and those can be aggregated, in linear layer $$W_0$$. Have four independent ways of analyzing the interactions, without expanding the dimensionality.

#### **Feedforward layer**

Output of self attention run this, conceptually not that interesting

A complete transformer block

Multiple times apply multi-head self-attention. Have skip connections. Why is the skip connection important? Importance of expectation of what something is, wider understanding of the world and reconsider how the world works. Higher level of understanding has a possibility to interact with lower level of understanding. Remodulate how we understand the input

Remark: Connections to Resnet, LSTM, optimization

#### **Position of encoding of words**

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

### 01 - What is unsupervised learning

Birds eye view. 

- Supervised learning: Learn mapping from given inputs to given outputs, that hopefully generalizes. 
- RL: which actions to take at any state to maximize expected future rewards, only gets feedback once it finishes the task, unless supervised learning, reward is sparse. 
- Unsupervised learning: No teaching signal, only have input data. Find structure in data

Do we need unsupervised learning?

What can we do with that data? 

- Clustering, groups similar data
- Dimensionality reduction, find dimensions which explain most of the variance (sounds like PCA)

How do we evaluate it?

No ground truth to compare it to. Clusters, how do we know which clusters are good? So many possibilities. How do we know which choices are good or bad? We can reduce dimensionality by

-  orthogonality (PCA)
- independence (ICA)
- something else

![TODO](/assets/images/deepmind_lecture_part_2/e10_01_dimensionality_reduction.png)

### 02 - Why is it important?

How necessary is it. Why not develop better supervised algorithms instead?

History of representation learning. 1949 first mention of machine learning. Kernel methods hugely influential until 2006, hinton introduced unsupervised pre training then with deep networks. 2012 AlexNet makes pre training unnecessary. 

![TODO](/assets/images/deepmind_lecture_part_2/e10_02_history_of_machine_learning.png)

Then from 2012 more data, deeper models, better hardware made huge leaps in many fields. Is machine learning solved? Data efficiency is bad compared to human learning. Takes orders of magnitude more time than humans to play Atari. Often data is not readily available. 

![TODO](/assets/images/deepmind_lecture_part_2/e10_02_data_efficiency.png)

Another issue is robustness, there are adversarial attacks, pandas can be translated to gibbon . Can lead to huge issues in self driving cars. 

![TODO](/assets/images/deepmind_lecture_part_2/e10_02_robustness.png)

Generalization, once you learn to play a game, it should not matter what the background color is. Two state of the art RL algorithms fail, inability to grasp the core idea of the game. 

![TODO](/assets/images/deepmind_lecture_part_2/e10_02_generalization.png)

Transfer, re use previously acquired knowledge. Current state of the art models are not able to solve that if you change atari game slightly. 

![TODO](/assets/images/deepmind_lecture_part_2/e10_02_transfer.png)

All of the algorithms don't know causality, intuitive physics and abstract concepts. The lack "common sense"

![TODO](/assets/images/deepmind_lecture_part_2/e10_02_common_sense.png)

Solve many tasks efficiently is trending. We know how to solve single tasks with end to end deep learning given update and compute. For the next generation which is efficient, robust and generalizable for multiple tasks, we need a new paradigm. Leading minds think unsupervised learning is the solution. Represent the world before knowing a task. We won't need as much data anymore.

![TODO](/assets/images/deepmind_lecture_part_2/e10_02_representation.png)

Add representation between task and AI, flat are hierarchical? Once or learn it again. What would it look like?

### 03 - What makes a good representation?

This is a ill defined and challenging question. Look to related disciplines for inspiration, for example neuro-science. What is the definition of a representation?

#### "Formal system for making explicit certain entities or types of information, together with a specification of how the system does this"

Three ways of writing 37, representational from is orthogonal to information content. It is a useful abstraction and makes different types of computations more efficient.

![TODO](/assets/images/deepmind_lecture_part_2/e10_03_representation_numbers.png)

What happens in the brain when information moves in it. Becomes a manifold when you rotate or change perspective. Makes it hard to untangle different objects. Ventral stream transform reformats into better form for that task. Untangle representations.

![TODO](/assets/images/deepmind_lecture_part_2/e10_03_untangling_representations.png)

How cross a street? What other properties should representations have. Approach from RL perspective. Think of representations as states which form an MDP, can shed light on what properties representations should have. Two MDPs, should contain information about presence of cars. What information to include in representation?

![TODO](/assets/images/deepmind_lecture_part_2/e10_03_crossing_a_street_mdp.png)

Solving tasks requires focus, we would want to exclude all irrelevant information for task. Should support attention to remove unimportant details. This allows for better clustering, when we try to group similar experiences. Want to pull a taxi, need a different representation than for a "going home task". Want to keep the information about the color of the car, hence latent representation should have as input the task we are trying to solve.

Compositionality, the ability to infer the meaning of a complex expression is also important. The example can be read in multiple ways (who has the binoculars?). Compositionality introduces open-endedness.

![Compositionality, man with binoculars](/assets/images/deepmind_lecture_part_2/e10_03_compositionality.png)

So we would want to have the following properties from neuro-science:

- Untangeled
- Attention
- Clustering
- Latent information
- Compositionality

Another related discipline is physics. It constrains the form of tasks take. Can we bias a  representations to reflect certain fundamental physical properties to make it useful. There exists fundamental idea in physics, that is the ideas of symmetries. Physics is the study of symmetry. For example of spring, it can go forward in time or translate in space. This is the commutative property of two transformations. means that one is the symmetry of the other.

![Spring and time space translations](/assets/images/deepmind_lecture_part_2/e10_03_symmetry_spring.png)

Studying symmetries can help to unify existing theories in physics. Same idea applies in natural tasks, 3D scenes can be changed scale of an object or its positions, and the two transformations can be interchanged without changing the final state. This can also be seen as symmetry. Representations should reflect these symmetries.

![Symmetry in 3D scenes](/assets/images/deepmind_lecture_part_2/e10_03_symmetry_3d.png)

Another idea is Information bottleneck, which can help us analyze deep nets. One way of seeing supervised learning is: "find maximally compressed mapping of the input variable that preserves information as much as possible on the output variable". This leads us to the data processing inequality by Shannon, telling us that post-processing can not increase information.

![Network for Shannon inequality](/assets/images/deepmind_lecture_part_2/e10_03_shannon_network.png)

$$ I(Y;X) \geq I(Y; h_j) \geq I(Y; h_i) \geq I(Y; \hat{Y}) \qquad j < i$$

The goal of layer-wise processing is to discard any information which is not necessary.

Two other core ideas are invariance and equivariance:

- Invariance: representation remains unchanged when a certain type of transformation is applied to the input.

  $$ f(g \cdot x) = f(x)$$

  ![Invariance example](/assets/images/deepmind_lecture_part_2/e10_03_invariance_example.png)

- Equivariance: representation reflects the transformation applied to the input.

  $$f(g \cdot x) = g \cdot f(x)$$

  ![Equivariance example](/assets/images/deepmind_lecture_part_2/e10_03_equivariance_example.png)

Example is disentangled representation learning. No general accepted definition of this term exists, but one common example is undoing the generative process and do the inference process. The idea of disentangling in machine learning is closely related to untangling in neuroscience. The two terms can be understood as follows:

- Generative process: Generating $$x$$ from attributes $$z_i$$:

  $$p(z) = \prod_i p(z_i)$$

  ![Generative process example](/assets/images/deepmind_lecture_part_2/e10_03_generative_process_example.png)

- Inference process: deducing attributes $$z_i$$ from observing input $$x$$

  $$p(x,z) = p(x, \hat{z})$$

  ![Inference process example](/assets/images/deepmind_lecture_part_2/e10_03_inference_process_example.png)

Group theory, we want to capture a set of symmetry transformations in the representations. For example horizontal and vertical translations and changes in color in a grid world which affect the state of a world.

![Position and color changes](/assets/images/deepmind_lecture_part_2/e10_03_position_color_groups.png)

*<u>Remark</u>: A group is defined as follows: A group is a set $$G$$ with binary binary operator, often denoted as $$\cdot$$, that combines two elements with the following three requirements:*

1. *Associativity: For all $$a,b, c \in G$$ we have: $$(a \cdot b) \cdot c = a \cdot (b \cdot c)$$*
2. *Identity element: There exists an element $$e \in G$$ such that for every $$a \in G$$ we have: $$e \cdot a = a$$ and $$a \cdot e = a$$.*
3. *Inverse element: For each $$a \in G$$ there exists an element $$b \in G$$ such that $$a \cdot b = e$$ and $$b \cdot a = e$$. For each $$a$$, the element $$b$$ is unique. It is called the inverse and denoted by $$a^{-1}$$.*

Our symmetry group example then consists of the three operations mentioned in the image above, vertical and horizontal translations plus changes in color and is denoted by:

$$G = G_x \times G_y \times G_c, \qquad \cdot : G \times W \longrightarrow W$$

where $$W$$ is an object with attributes $$(x, y, c)$$, coordinates and color. They affect the abstract representation of our grid world. 

*<u>Remark</u>: In group theory, the symmetry group of a geometric object is the group of all transformations under which the object is invariant.*

Assume there is generative process that maps state to map. The goal is to learn a mapping from observation $$o$$ to representation $$z$$, such that $$f$$ is an equivariant map. So we can either apply the transformation in the abstract space or in the representation space and still get the same result. If we can find such a map $$f$$, then our representation is said to be reflective of the underlying symmetry transformations. We want to find an equivariant map $$f$$ s.th.

$$g \cdot f(w) = f(g \cdot w) \qquad \forall g \in G, w \in W$$

![Equivariant map example](/assets/images/deepmind_lecture_part_2/e10_03_equivariant_map_example.png)

### 04 - Evaluating the merit of a representation

How can we verify that we are on the correct path when evaluating representations? Lets use example and go through the potential verification steps​, and check how it fulfills the list scavenged from physics and neuro science

- Symmetry: is included by definition
- Untangled: is also included by definition
- Compositionality: the underlying symmetry group is meant to be composition of subgroups (changes in position in color) the resulting representation should also be compositional
- Attention: Since representation is split into independent sub spaces, it should easily support attention. For example implemented as a binary mask over the subspaces.
- Clustering: Need a metric to compare representations, we can assume that it lives in vector space and pick a metric that is defined on a vector space.

Hence the representation does tick all the boxes. But does it address some of the shortcomings?

- Data efficiency: It is likely that the majority of natural tasks would be specified in terms of some of the entities conserved by symmetry transformations, which in turn would correspond to the independent subspaces recovered in our representation. For example: naming the color of an object would only require a linear transformation from a single subspace. Research shows that incorporating symmetry does help improve the data efficiency in supervised tasks (see the paper [Deep Symmetry Networks](https://papers.nips.cc/paper/2014/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html))
- Robustness: Mapping f needs to be equivariant, its functional form will have to be quite constrained. This might help make it more robust to adversarial noisy perturbations. Evidence suggests that may be the case, which has been provided by recent work (see the paper [Achieving Robustness in the Wild via Adversarial Mixing with Disentangled Representations](https://arxiv.org/abs/1912.03192)). Furthermore augmenting the representation with attention is likely to help with robustness too (see the paper [Towards Robust Image Classification Using Sequential Attention Models](https://arxiv.org/abs/1912.02184)). The attention based model was much harder to fool than the baseline, you can see the beaver in the image which had to be drawn.
- Generalization: It can be increased if the decision on which action to take can be made without those aspects of representation that are not important of to the task. Since our symmetry based representation of Z preserves the important information about the stable course of the world in a format that allows for fast attention attenuation, we quickly adapt the minimal set of informative subspaces available to the decision network when faced with solving diverse tasks. Thus increasing the generalization of such decision networks.
- Transfer: Mapping f connects the underlying symmetry transformation to the representation, it should not care about the nature of the intermediate observation. Was shown that the schema networks, models that use hand engineered features, could transfer its ability to breakout much better than the unstructured deep RL baseline (see the paper [Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics](https://arxiv.org/abs/1706.04317)).
- "Common sense": This is the least explored area of machine learning, preliminary evidence suggests that our hypothesized representations may support compositional abstract imagination and maybe a solution for grounding many promising discreet or symbol based algorithms. Which have been shown to have some of the desirable properties missing in deep neural networks, like induction or abstract reasoning.

Currently no algorithm exists that can learn such symmetry equivariant representations in a robust and scalable manner. Aiming for such representations may be a good research direction in unsupervised representation learning.

Recap of this part of the talk:

We have seen that designing good representations played a crucial historical role in machine learning. this early role was made obsolete by the successes of end to end deep learning, which seemed to be perfectly capable of finding good representations implicitly. These algorithms however still have many shortcomings that are becoming more and more prominent as we start reaching a plateau in exploiting the strengths of the current systems. Many of the current advances addressing the current shortcomings have already been attributed to learning better representations. For example by adding auxiliary losses or inductive biases to the models. This suggests that further advances may be accelerated if we start thinking about what makes good representation explicitly and try and bias models to learn such representations intentionally. One way to gain insights of what makes good representations is by looking into  related disciplines such as neuro science, cognitive science, physics or mathematics. One can wonder, is all machine learning ultimately about representation learning?

### 05 - Techniques and applications

Three main pillars to achieve outlined goals.

![Three pillars of representation learning](/assets/images/deepmind_lecture_part_2/e10_05_three_pillars_of_representation_learning.png)

- Generative modeling: learn the data distribution using generative modeling, often through reconstruction

- Contrastive losses: use classification losses to learn representations that preserve temporal or spatial data consistency

- Self-supervision: exploit knowledge of data to design learning tasks which lead to useful representations.

There are several downstream tasks for representation learning. 

- Semi-supervised learning, use the learned representations for classification. We aim for data efficiency and generalization.
- Reinforcement learning, use the learned representations for model based RL or model free RL. We aim for data efficiency and transfer learning.
- Model analysis, use the learned representations to analyze what the model learns. We aim to have interpretable models.

The general approach is to first learn from unsupervised data only, then ask questions about the representations learned by building downstream tasks to asses what kind of information is in there. In classification, we want to see wow much information about the label is still present, often by building a simple linear classifier on top of the representations. One way to do this is by first training on ImageNet without any label information, then use a small percentage to train classifier to see how well the learned representations are doing. This can allow us to compare different representations.

In reinforcement learning tasks are often very hard to learn from online data. Learning disentangled representation can speed things up. In model analysis we would like to under stand what the model is doing, do they satisfy a certain property. By learning interpretable models, want to see what it is learning before deploying it in production.

We want to do the above for discrete (example: face has glasses) and continuous (example: color of the hair) representations. Be able to do online learning, where the representations adapt with experience. Have consistency and temporal abstractions. An example of consistency in data would be, no matter from which angle a scene is represented, we want the same representation.

#### **Generative Modeling**

The main question here is: what kind of distribution could have generated our dataset? An example could be a mixture of two Gaussians.

![Two Gaussians and their data](/assets/images/deepmind_lecture_part_2/e10_05_two_gaussians.png)

Learning probability distributions efficiently has a lot of connections with compressing data. Want representations that are efficient and compressed.

Assume that the generative process looks like in the image below. Where $$z$$ is a low and $$x$$ high dimensional space. Assume that $$x$$ are generated by very complicated mapping from $$z$$, for example a neural network. One example of models that can do that are latent variable models. They map from low dimensional to high dimensional space. Here we want to be able to model the sampling process which generates $$x$$ from $$z$$

![Generative and inference process in a latent variable model](/assets/images/deepmind_lecture_part_2/e10_05_latent_variable_model.png)

Inference is then going the other way around, getting $$z$$ from the high dimensional space $$x$$, i.e. modeling $$p(z \mid x)$$. In practice, we learn inference and generation together. Because we do not have access to true distribution of $$z$$. An example of the two spaces $$x$$ and $$z$$ are shown below:

![An example of a generative and inference process](/assets/images/deepmind_lecture_part_2/e10_05_generation_inference_example.png)

A class of models which is explicitly made for this setup are variational autoencoders. Where we try to to maximize the likelihood of the data $$p^*$$ under our model $$p_{\theta}$$ with parameters $$\theta$$:

$$\mathbb{E}_{p^*(x)}[\log p_{\theta}(x)]$$

Where we represent the model as

$$\log p_{\theta}(x) = \log \int p_{\theta}(x \mid z) p(z) dz $$

for prior $$p(z)$$. However we do not have direct access to $$p_{\theta}(x)$$ and would need the above integral with Monte Carlo, which is intractable.

Variational autoencoders. Uses maximum likelihood, can't see p*(x) directly but use Monte Carlo simulation. Challenge is latent variable and train them with MLE, it is given by an integral (describes the integral, what it does). p(z) is a prior. A way around this is instead optimizing a lower bound on the likelihood:

$$p_{\theta} (x) \geq \mathbb{E}_{q_{\eta}(z \mid x)} p_{\theta}(x \mid z) - \text{KL}(q_{\eta}(z \mid x) \| p(z))$$

The term $$ q_{\eta}(z \mid x) $$ is called approximate posterior and the entire inequality evidence lower bound (ELBO). The first term

$$\mathbb{E}_{q_{\eta}(z \mid x)} p_{\theta}(x \mid z)$$

is a likelihood term, reconstructs the high dimensional object $$x$$ from the low dimensional representation $$z$$, this is the generative process from the image above. It tries to assign a high likelihood to the original $$x$$ we have seen. This is only possible if the original information was encoded efficiently in $$z$$. The second term, 

$$\text{KL}(q_{\eta}(z \mid x) \| p(z))$$

is the KL-divergence between the true and the approximate prior. It regularizes the approximate posterior to the prior. The prior allows us the specify properties we would like the posterior to have, such as disentanglement. An example is a Gaussian with independent dimensions, if $$z$$ has $$k$$ dimensions, we can force $$z$$ to be $$z \sim N(0, I_k)$$. In practice both the inference $$q_{\eta}(z \mid x)$$  and generation $$p_{\theta}(x \mid z)$$ model are deep neural networks.

![Example image of a variational autoencoder](/assets/images/deepmind_lecture_part_2/e10_05_vae_nn.png)

*<u>Remark</u>: This is only a very brief introduction to VAE, for more details see lecture 11 or the paper [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)*

A variation of the classical VAE are $$\beta$$-VAE. The additional weight on the KL term encourages disentangled representations:

$$p_{\theta} (x) \geq \mathbb{E}_{q_{\eta}(z \mid x)} p_{\theta}(x \mid z) - \beta \text{KL}(q_{\eta}(z \mid x) \| p(z))$$

By learning disentangled continuous representations we can traverse the latent space, this is called latent traversal. Example of disentangled latent spaces are: location, object, distance to object and rotations in the image above.

![Example of disentangled latent representations](/assets/images/deepmind_lecture_part_2/e10_05_disentangled_representations.png)

 All six are dimensions are fixed except for one. See what changes in the scene if you change one variable, the entangled is a non-betaVAE. Beta-VAE can also be integrated in reinforcement learning agents, they improve generalization and transfer. This allows to transfer quicker from simulation to reality.

Downstream tasks, evaluate how good representations are for transfer learning and generalization. Allows RL learnings to transfer quicker form simulation to reality. See the [paper](https://arxiv.org/abs/1707.08475).

![Sim to reality in RL](/assets/images/deepmind_lecture_part_2/e10_05_rl_sim_to_reality.png)

Another idea are sequential VAE - ConvDraw, where a recurrent component is introduced. The recurrence helps iteratively refine the image and add additional details.

![Sequential VAE, architecture and example](/assets/images/deepmind_lecture_part_2/e10_05_sequential_vae.png)

In the car example, we first draw the outline then give more details. The model moves from high level outline to final detailed  image. Such a model can have posterior distributions that are autoregressive and way more complex, can be closer to true posterior. This can get closer to the theoretical lower bound we saw before. Read the [paper](https://arxiv.org/abs/1604.08772).

Another approach are layered models, called MONet. It combines VAEs, segmentation networks and attention. It allows the network to iteratively pay attention to different parts of the image since we feed in an attention mask as well. This allows the network to segment objects in an unsupervised way.

![MONet VAE architecture](/assets/images/deepmind_lecture_part_2/e10_05_monet_vae.png)

Using attention in multi-level process leads to generative models which learn concepts unsupervised. Experiments show that latent variables learn to encode the position of an object into a single latent variable. The resulting representations help improve tasks in reinforcement learning. For more details see the [paper](https://arxiv.org/abs/1901.11390).

![MONet example](/assets/images/deepmind_lecture_part_2/e10_05_monet_example.png)

Generative Query Networks (GQN) look at consistency property by learning a representation by looking at a scene from two different angles. This information is then encoded in a latent space and the model has to learn to draw the same scene from a different angle.

![generative query networks architecture](/assets/images/deepmind_lecture_part_2/e10_05_gqn_architecture.png)

An interesting fact is that GQN can capture uncertainty about the state of the world. See the example below where we give it an image where it looks at a wall. It is able to imagine that there are multiple objects behind a wall. This is an important property, we want representations to encode uncertainty about the state of the world. For more details, see the [paper](https://science.sciencemag.org/content/360/6394/1204).

![uncertainty in generative query networks example](/assets/images/deepmind_lecture_part_2/e10_05_gqn_uncertainty.png)

Vector quantized VAEs are solving a different problem, here we want the latent space to be discrete. Learning discrete latent variables is challenging since you can't propagate a gradient through them and have to estimate it with high variance. How it works here is to start with continuous vector, look into learned table of embedding and look for a nearest neighbor in that table. The index will then give you the discrete variable. 

![vector quantized VAE architecture](/assets/images/deepmind_lecture_part_2/e10_05_vq_vae_architecture.png)

Having a discrete latent space is giving us a very good compression algorithm, since the variables can be used to capture high and low level information from the data. But reconstruction can be a bit blurry, because we are using probability model to approximate the original content. For more details see the [paper](https://papers.nips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html).

![vector quantized VAE example](/assets/images/deepmind_lecture_part_2/e10_05_vq_vae_example.png)

What about GAN? The image below gives a quick recap of how they work, the main two components being the generator and the discriminator. Where we only use the discriminator during the training. When generating samples with GANs we have no reconstruction loss and the learned model is implicit.

![GANs architecture](/assets/images/deepmind_lecture_part_2/e10_05_gan_architecture.png)

 The problems with GANs are that we can't answer the inference question (what is the latent representation for a given sample) and there is no uncertainty around the learned representations. Hence we need a new model.

![Example GAN can't do inference](/assets/images/deepmind_lecture_part_2/e10_05_gan_inference.png)

BigBiGan can do that, it learns to encode data by changing the adversarial game. In the left part, the encoder is very similar to VAE case. The crucial change is how the discriminator changes, so far it was only evaluated between samples from data. This is the $$x$$ and $$\hat{x}$$ distribution. Want to go beyond that, we also want to match the latent variables $$\hat{z}$$ and $$z$$ using the prior and invert the generation process from latent samples to model samples. We can do that by having a discriminator that uses pairs, the model has to distinguish between the two samples. The joint distributions have to be matched. Marginal distribution of the encoder is going to be equal to that of the prior in the VAE case. But we also matched the relationship between x and z. All this is done without having any reconstruction loss.

![BigBiGAN architecture](/assets/images/deepmind_lecture_part_2/e10_05_bigbigan_architecture.png)

There is no pixel level loss for reconstruction, hence the reconstructions capture high level informations as do the latent spaces. This means the representations can be meaningful for semi-supervised learning. The [paper](https://papers.nips.cc/paper/2019/hash/18cdf49ea54eec029238fcc95f76ce41-Abstract.html) was ImageNet SOTA at the time of publishing.

![BigBiGAN example](/assets/images/deepmind_lecture_part_2/e10_05_bigbigan_example.png)

#### **Contrastive learning**

Contrastive learning is completely unsupervised and removes the need for a generative model. It uses classification loss instead, but built from unsupervised data. If that is done in the right way, the context is encoded into our representation. 

An example for a contrastive loss is word2vec, it learns representations for text. This is specifically important for text, because the simplest approach would be one hot encodings. This encodes no semantic information, there are no relationship between words. For example we have no way of representing that Beijing is to China what Moscow is to Russia. So we want to learn representations of text that encode semantic information. We can do that by learning a model which predicts the kind of representation it should expect, given the past context. Where the context is a few words. The crucial bit of that kind of loss is providing positive and negative examples. So we train it saying this is what you should expect in the next point but this are words which you should not expect.

$$\log \sigma(v_{w_O}^T v_{w_I}) + \sum^k_{i=1} \mathbb{E}_{w_i \sim P_n(w)} \bigg[\log \sigma(-v_{w_i}^T v_{w_I})\bigg]$$

<u>Remark:</u> *This formula is not clear at all without some explanation. It's better to have a look at the [paper](https://papers.nips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) instead*

In translation, the relationships in for the same words should be the same in different languages, because they represent the same concept. How do we do that? Train word2vec unsupervised in English, train word2vec model unsupervised in Spanish and then use few example to learn a simple linear mapping between the two languages. Does this map generalize? Can we do dictionary translation if we use a smart mapping for these words? The answer is: yes. With word2vec we can then translate using a very similar mapping. This is not perfect but most of the words have a very similar semantic meaning.

![Word2Vec examples for translation](/assets/images/deepmind_lecture_part_2/e10_05_word2vec_examples.png)

In Contrastive predictive coding we try to maximize mutual information between data and learned representations. Think of the idea as temporal coherence structure. It uses supervised learning to model density ratios:

$$ L_N = - \mathbb{E}_X \bigg[ \log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)}\bigg] $$

<u>Remark:</u> *Again the formula on its own is not very useful without having a closer look on the [paper](https://arxiv.org/abs/1807.03748).*

 Can also be used for spatial data such as images, different patches. Representations learned in this way are more useful for downstream tasks in the low data regime. Below is an example for audio:

![Contrastive predictive coding for audio](/assets/images/deepmind_lecture_part_2/e10_05_contrastive_predictive_coding.png)

SimCLR is another contrastive loss idea. If you transform the image a little bit, then you still want to have same representation after applying a transformation $$f$$. While it should contain most of the information,it should not be the exact same information. Hence we add a different mapping $$g$$ to then obtain a representation which is the same.

![SimCLR architecture](/assets/images/deepmind_lecture_part_2/e10_05_simclr.png)

The downstream task uses the representations produced by $$f$$. The [paper](https://arxiv.org/abs/2002.05709) which introduced this loss achieved SOTA on ImageNet benchmarks, on a linear classifier trained on input representations and on semi-supervised learning where you can train on 10% of the images.

![SimCLR results](/assets/images/deepmind_lecture_part_2/e10_05_simclr_results.png)

Take home message: contrastive losses use classifiers to learn representations which are temporally or spatially consistent.

#### **Self supervised learning**

Want to encode the kind of information which is useful for downstream tasks in our representations. This is for tasks where it is easy to obtain data and we can encode prior knowledge into the data modality. For example in image colorization, start with an image of colored data and don't need any labels. Use a tool to turn the images into black and white. Ask the model to revert that mapping, i.e. how to colorize. If we do this right we can use the representations for semi-supervised learning tasks, such as in ImageNet.

![Image colorization example](/assets/images/deepmind_lecture_part_2/e10_05_image_colorization.png)

Important here, we started with an unsupervised set and created supervised data by thinking of the kind of properties we want our data to encode. We can go beyond colors, can take an image and ask the model to learn spacial consistency by looking at different patches of the image.

![Context prediction in images example](/assets/images/deepmind_lecture_part_2/e10_05_context_prediction.png)

Given a particular patch, which one do you think the other patch is. What the model has to learn is that if we have a cat, we need to know where the ear is, in the example above it would be on the upper right, hence the label $$Y=3$$. The model only does 8-way classification, but in order to answer that kind of question, it has to understand and learn the kind of representations that are useful for object discovery.

Can go beyond single images with video data. Take a couple of frames, shuffle them and ask the model to sort the sequence. To learn temporal coherence. We can do that without predicting particular frames, only needs to learn the order. Downstream tasks include object and action recognition.

![Shuffled sequences example](/assets/images/deepmind_lecture_part_2/e10_05_shuffled_sequences.png)

One example that combines a few of the methods we have seen before is BERT. It learns representations of text by leveraging both tasks that allow the model to learn local and global structure. For example, one of the tasks is to learn which particular words have been masked in an input sentence. Given a sentence we mask a few words and ask the model what is the right word to fill in the blanks. Crucially, the model is also asked given two sentences A and B, which sentence does come first? This is long term temporal coherence and is very different than answering questions about local structure by filling in single words. BERT has sparked a revolution in NLP, and used for multiple downstream tasks. It has also been deployed into production in Google search.

![BERT deployed in google search example](/assets/images/deepmind_lecture_part_2/e10_05_bert_deployed_google_search.png)

Take home message: Self supervised learning exploits domain knowledge to build tasks useful for representation learning.

Keep in mind that:

- Task design for learning representations is important

- Modality is important

- Context is important

- Learning generative models is hard, might be able to get away without it (contrastive losses, self supervision)

- Crucial benefits by incorporating changes in neural architectures

  

### 06 - Future

- Generative models: powerful posteriors (autoregressive) and better priors (disentanglement)
- Contrastive learning: going beyond temporal and spatial coherence
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



# References

- Definition of a group: https://en.wikipedia.org/wiki/Group_(mathematics)
- Symmetry Group: https://en.wikipedia.org/wiki/Symmetry_group

# Comments