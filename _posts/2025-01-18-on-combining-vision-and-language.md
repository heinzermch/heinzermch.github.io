---
layout: post
author: Michael Heinzer
title:  "Combining Vision and Language"
description: On combining vision and language with LLMs
date:   2025-02-23 18:00:00 +0530
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

- **symmetric vs asymmetrics cross-entropy loss**

- **InfoNCE loss** van der Oord.



# Contrastive Language Image Pretraining (CLIP) - Feb 2021

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





## Prompt Engineering
The model is embedding entire sentences and not just words, this allows for using prompt engineering to improve results. For example a common issue with only using single words is that they can have multiple meanings without context (maybe even with context). One example from ImageNet is construction cranes and the animal cranes that fly.

Another issue is that in pre-training the images will often be paired with entire sentences (alt-text) and not single words, this means at inference time using single words will be out of distribution as a task.

The paper uses a simple prompt template "A photo of a {label}" as a default. This default is then often adapted to each task.

Another option to increase performance is to ensemble over many different context prompts, which lead to an improvement of 3.5% over the default prompt.

The paper presents many results on different tasks, that we will not go into here. One of the main take-aways is that CLIP features outperform the features of an ImageNet pre-trained model on a wide variety of datasets. They tested on 27 tasks. In general CLIP features are more robust to task shift compared to ImageNet models.

One interesting result was comparing CLIP to zero, one and two shot humans.

TODO: Image

interesting point noted: humans know what they don't know, and can generalize much better from one example, large jump in accuracy. This point is still valid for LLMs today in 2025. They can't tell us what they don't know and tend to hallucinate instead. Once could argue that they are better these days with generalizing from few examples in few-shot prompts, but that has it's limits as well as any practitioner can tell.

# Sigmoid Language Image Pretraining - 2022 to 2025

These are a series of papers by (now former) Google researchers in Zurich. They build upon the ideas from CLIP, among many others, to create one of the best open source image backbones.

## LiT: Zero-Shot Transfer with locked Image-Text Tuning - 2022

The paper presents a contrastive tuning method to align image and text models, while taking advantage of pre-trained models. Here this is called "Locked-image Tuning" (LiT). It teaches a text model to read representations from a pre-trained image models for new tasks. It works for multiple vision architectures.



The main contribution from paper is a strategy they call contrastive tuning. It consists of only tuning the text tower while using a pre-trained image backbone. This is called locked-image tuning because the image tower is frozen during training (locked).

This achieves better results than from scratch training like CLIP.

TODO: Figure 1 from paper comparing performances

The paper also tests multiple variants:

 * Initialize from pre-trained models, train text only.
 * Initialize from pre-trained models, train text and image.
 * Initialize randomly, train text and image.
 
The image denotes this in L (locked and pre-trained), U (unlocked and pre-trained) and u (unlocked and randomly initialized).


TODO: Figure 2 from the paper.


The two models are trained with a contrastive loss as in CLIP. They ablate if it should be calculated on a per accelerator basis or computed jointly accross devices. The second version consistelty gives better results. This intuitively makes sense and we will see in later papers that a certain batch size is necessary for the contrastive loss to work well.

Contrastive pre-training can be viewed as learning two tasks at the same time:

1) Learning an image embedding
2) Learning a text embedding to align with image embedding space.

Contrastive pre-training works well for both of these tasks but might not be the optimal approach, hence these experiments.

Again the two options at the time were pre-training image models on a fixed set of categories like ImageNet-21k or JFT-300M, or do free-from pretraining on image-text pairs from the internet.

The first appraoch has cleaner data but the categories the image model learns are restricted to those in the training set. There is no guarantee that the embeddings work well outside of categories. And experience tells me that it probably won't.

The second approach has orders of magnitude more data available (as we will see in the next papers) and tends to generalize much better, even though the labels are of weaker quality. This might not be a suprise after the LLM revolution in the past years.



## Sigmoid Loss for Language Image Pre-Training - 2023

The main contribution of this paper was a new loss for contrastive pre-training. Unlike the previously seen loss which uses softmax normalization, the sigmoid loss only operates on image-text pairs and we do not need a global view of the pairwise similarities for normalization.

This allows increased batch size while also improving the performance at smaller batch sizes.

The disentanglement from batch size and loss allows for further studying of imapct of the ratio of positive to negative examples in a batch. Before this was fixed at N^2 pairs of which N were positive and N^2-N were negative.

### Theoretical Method

For: 

 * $$f(\cdot)$$ an image model.
 * $$g(\cdot)$$ a text model.
 * $$B = \lbrace (I_1, T_1), (I_2, T_2), \cdots \rbrace$$ a mini-batch of image-text pairs.
 * $$ \mid B \mid $$ the mini-batch size.
 * $$x_i = \frac{f(I_i)}{\mid \mid f(I_i) \mid \mid_2}$$ a normalized image embeddings.
 * $$y_i = \frac{g(T_i)}{\mid \mid g(T_i) \mid \mid_2}$$ a normalized text embeddings.
 * $$t$$ a scalar parametrized as $$\exp(t')$$, where $$t'$$ is a freely learnable parameter.


The softmax loss for contrastive learning is: 

$$ L_{softmax} = \frac{1}{2 \mid B \mid} \sum_{i=1}^{\mid B \mid} \Bigg( \log \frac{e^{t x_i y_i}{\sum^{\mid B \mid}_{j=1} e^{t x_i y_j}} +   \log \frac{e^{t x_i y_i}{\sum^{\mid B \mid}_{j=1} e^{t x_j y_i}} \Bigg)$$


The first part in the sum is image to text softmax, the second is text to image softmax. Note, due to the asymmetry of the softmax loss, the normalization is independently performed two times: across images and across texts.

In case of the sigmoid loss, we do not require computing global normalization factors. Every image-text pair is procesed independently. Turning the problem into a standard binary classification proglem of all pair combinations. So for image-text pairs $$(I_i, T_j)$$ we have $$z_{ij} = 1$$ for $$i=j$$ and $$z_{ij} = -1$$ if $$i \neq j$$. The loss is defined as:

$$L_{sig} = - \frac{1}{\mid B \mid} \sum^{\mid B \mid}_{i=1} \sum^{\mid B \mid}_{j=1} \log \frac{1}{1+e^{z_{ij}(-tx_i y_j + b)}}$$.

There is a new bias term $$b$$ in this loss to correct for the heavy inbalance of many negatives that dominate the loss initially. The two terms are initialized as:

* $$t'= \log(10)$$ for the temperature $$t$$
* $$b = -10$$ for the bias $$b$$.

This assures that the training starts close to the prior and does not require over-correction. Note: what is the prior here?

The pseudocode implementation of this loss is:

```python
# img_emb : image model embedding [n, dim] 
# txt_emb : text model embedding [n, dim] 
# t_prime, b : learnable temperature and bias 
# n : mini-batch size 
 
t = exp(t_prime) 
zimg = l2_normalize(img_emb) 
ztxt = l2_normalize(txt_emb) 
logits = dot(zimg, ztxt.T) * t + b 
labels = 2 * eye(n) - ones(n) # -1 with diagonal 1 
l = -sum(log_sigmoid(labels * logits)) / n
```

### Practical Considerations

Contrastive learning in the form of the softmax loss typically uses data prallelism, computing the loss when data is split across $$D$$ devices is expensive because we need to gather all embeddings with expensive all-gather operations. And the materialization of a $$\mid B \mid \times \mid B \mid$$ matrix of pairwise similarities.

For the sigmoid loss a more efficient implementation exists that avoids these issues. If we assume that each device holds $$b = \frac{\mid B \mid}{D}$$ examples, then for each example we have one positive and $$b-1$$ negatives on the device. Then the text representations are permtuted among the devices to calculate $$b$$ negative examples with each step. The loss is calculated on a per device basis for each local batch $$b$$ and summed up across devices. The process if visualized in the following image for three devices and four examples per device:

TODO: Image

The peak memory cost is reduced from $$ \mid B \mid^2$$ to $$ b^2$$, and $$b$$ can be kept constant while scaling up accelerators. This allows much larger batch sizes than the original cross-entropy loss. 

There is a lot more engineering details and considerations in the paper that I don't want to go into too many details here:

* SigLIP works better for smaller batch sizes, but at larger batch sizes like 32k softmax loss catches up. Larger batch sizes make more sense if training for longer, on ImageNet 0-shot transfer.
* Increased batch size leads to increased training instability due to spikes in gradient norm, decreasing $$\beta_2$$ momentum helps.
* Losses paradigm: softmax "pick the right class" vs. sigmoid "rate this pair".
* Sigmoid loss allows to remove negatives from the training process, the only way to not decrease performance significantly is to keep the "hard examples". Which intuitively makes the most sense, since learning is done on these.



## SigLIP 2: Multilingual Vision-Language Encoders - 2025






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


