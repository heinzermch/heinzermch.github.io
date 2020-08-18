---
layout: post
author: Michael Heinzer
title:  "DeepMind lecture series"
description: Taking notes and summarizing
date:   2020-08-15 11:00:00 +0530
categories: None
comments: no
published: no
---
I was watching this DeepMind lecture series. Would like to see more about Deep RL, Graph

## Episode 7 - Deep Learning for Language Understanding

### 01 - Background: Deep Learning and Language

What is not covered. Speech, many NLP tasks, field is much bigger. Some tasks still use almost no neural networks, hard to do end to end with home assistants dalogue systems. Trend is moving towards neural and deep papers. Performance in GLUE benchmark, represents challengening language tasks, performance is still increasing up to this day. 

Why is deep learning such an effective tool for language processing? Need to think about language itself, and why it fits together with deep learning. Mapping symbols to symbols, but its not actually that simple (see face slide). Delve into the meaning of face, have certain aspects in common but are not exactly the same. Pointing aspect of face. Disambiguation depends on context

Went and event in handwritten sentence is easily read by humans. Reading symbols up and down or left to right

Important interactions can be non-local, look at wider context

Combining pet and fish to pet fish changes the color associated with them.

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

## Episode 10 - Unsupervised

## Episode 11 - Modern Latent

## Episode 12 - Responsible