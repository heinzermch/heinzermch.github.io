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

- Hard vs soft attention
- Transformers reduced to the essential

## Episode 9 - Generative

## Episode 10 - Unsupervised

## Episode 11 - Modern Latent

## Episode 12 - Responsible