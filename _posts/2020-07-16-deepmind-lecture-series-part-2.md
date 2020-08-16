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

Only SimCore works well, generative model. CPC was much less predictive. Green line is if you backpropagte from the answer of the question, much more specialized.

Conclusion. Context can be non-local and non lingusitc. Can get background knowledge not only from text but from other modalities. Pipeline view of language previously, more realistic view of language processing.

### My Highlights

- Intuition of skip connection in Transformers
- Aggregate knowledge from environment in single model with conceptual understanding. In a single agent.

## Episode 8 - Attention and

## Episode 9 - Generative

## Episode 10 - Unsupervised

## Episode 11 - Modern Latent

## Episode 12 - Responsible