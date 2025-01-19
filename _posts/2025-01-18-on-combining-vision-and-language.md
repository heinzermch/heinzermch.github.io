---
layout: post
author: Michael Heinzer
title:  "Combining Vision and Language"
description: On combining vision and language with LLMs
date:   2025-01-19 18:00:00 +0530
categories: LLMs Vision SigLip ContrastiveLoss
comments: yes
published: false
---

Hopefully a quick overview of fundamental concepts in LLM vision.

## Basic Concepts and Notation

Before we start, let us quickly repeat some basic concepts and their notation. Readers familiar with the topic may skip this section.

- **Softmax trick**

- **Sigmoid**

- **Cosine similarity**

- **Kullback-Leibler Divergence**: The KL-divergence of two probability distributions $$p$$ and $$q$$ is defined as

$$D_{KL}(p \mid\mid q) := \sum_{x \in X} p(x) \log\bigg(\frac{p(x)}{q(x)}\bigg) = - \sum_{x \in X} p(x) \log\bigg(\frac{q(x)}{p(x)}\bigg)$$

Often we consider $$p$$ to be the true distribution and $$q$$ the approximation or model output. Then the KL-divergence would give us a measure of how much information is lost when we approximate $$p$$ with $$q$$.



# Contrastive Language Image Pretraining

This paper from OpenAI in 2021 introduces a new way of combining vision and language that scales with data. It enabled zero-shot transfer to new taks that was competitive with a fully supervised baseline.

In that that time GPT-3 showed what was possible in language, pre-train with self-supervised objective on enough data with enough compute and it the models became competitive on a many tasks with little to no task-specific data. This showed that web-scale data could surpass high-quality crowd-labled NLP datasets. In computer vision this was not yet standard.

In computer vision one option to take advantage of web-scale data was to pre-train models on noisly labled datasets with classification tasks. Typical datasets would be ImageNet or JFT-300M (Google internal). These pre-trained backbones where then finetuned on task specific training sets. This is also the approach we used at places where I worked at time (Panasonic, Scandit). However these pre-training approaches all used softmax losses to predict various categories. This limits the concepts that they learn to the spefici categories in the pre-training sets and makes it harder to generalize in zero-shot settings.

CLIP closes that gap by pre-training image and language jointly at scale. Resulting in models that outperform the best publicly available ImageNet models while being more computationally efficient and robust. The core of this approach is learning perception from supervision contained in language.

The advantages learning from natural language (as opposed to classification) are:
 - Much easier to scale natural language supervision as opposed to crowd-sourced labeling for classification
 - It leanrs a representation that connects language to vision
 - Better zero-shot transfer performance due to language interface
 
The dataset for pre-training is 400 million image-text pairs obtained from the internet from 500k queries. The set is class balanced to contain up to 20k pairs per query. They call this dataset WIT for WebImageText.

## Training Process
Predicting the exact words in an image-text pair was found to be wasteful in compute. Instead they adopt an approach from contrastive representation learning. It predicts only if a text as a whole paired with an image makes sense:

Given a batch of $$N$$ (image, text) pairs, CLIP is trianed to predict which of the $$N \times N$$ possible (image, text) pairs accross a batch occured. To do this the model needs an image and text encoder that embedd the inputs in the same space. The goal is then to maximize the cosine similarity between the image and text embeddings in the $$N$$ pairs, while minimizing the cosine similarity of the $$N^2-N$$ incorrect pairings. This is done with a symmetirc cross entropy loss over all the similarity scores.

TODO: Formula

First popularized by Oord et al (2018) as the InfoNCE loss.

Clip is trained from scratch without initializeing the iamge encoder or the text encoder. The embeddings are then created with a linear projection to the multimodal embedding space.


Pseudo code for CLIP:


```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```








Pre-train text and vision encoder on query-image pairs with contrastive loss

Do some prompt engineering to get better performance


# Sigmoid Language Image Pretraining


SigLIT vs SigLIP


# Conclusion


## Final Thoughts

Some follow up questions and remarks for my future self or experts in this field:

* Question 1
* Question 2


## References

A list of resources used to write this post, also useful for further reading:

- [Proximal Policy Optimization Algorithms - John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov](https://arxiv.org/abs/1707.06347) for the original DPO paper by OpenAI.
 
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model - Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn](https://arxiv.org/abs/2305.18290) for the DPO paper by Stanford.
- [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback - Hamish Ivison, Yizhong Wang, Jiacheng Liu, Zeqiu Wu, Valentina Pyatkin, Nathan Lambert, Noah A. Smith, Yejin Choi, Hannaneh Hajishirzi](https://arxiv.org/abs/2406.09279) for a paper that compares DPO and PPO on different tasks.
- [Preference fine-tuning API by OpenAI](https://platform.openai.com/docs/guides/fine-tuning#preference) for an example of an FT API.
- [Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) by OpenAI Spinning UP.


## Comments

I would be happy to hear about any mistakes or inconsistencies in my post. Other suggestions are of course also welcome. You can either write a comment below or find my email address on the [about page](https://heinzermch.github.io/about/).


