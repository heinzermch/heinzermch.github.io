---
layout: post
author: michael
title:  "On Bayes And Birthdays"
date:   2020-11-25 21:03:36 +0530
categories: Probability Bayes Brainteasers Python MontyHall
---
# Introduction
One of my most favorite ways of reasoning is probabilistically. Often people tend to think, if I do A, then B must happen. For example, if I practice enough, then I will pass the exam or interview. More often than not it would be more appropriate to think if I practice a lot, I increase my chances of passing an exam or interview. Thinking probabilistically does not come naturally, at least to me. Often it is highly unintutive as this post will illustrate.

# Probabilistical vocabulary
Let us first have a look at the basics of probabilistic language. Readers familiar with basic probability may want to skip this section. Let $$A$$ be an **event** in the **sample space** $$\Omega$$, which we write as $$A \in \Omega$$. A sample space is the space of all possible outcomes. For example for a single coin flip, we have $$\Omega = \lbrace H,T \rbrace $$. $$\Omega$$ consists of the event $$H$$ and $$T$$, seeing a head and seeing a tail. We call $$P(A)$$ the probability of the event $$A$$, where $$P$$ is a **probability measure**. In laymans terms, $$P$$ measures how likely event $$A$$ is. It has the property that it is always between one and zero, which we write as

$$ 0 \leq P(A) \leq 1 \quad \forall A \in \Omega$$

Where $$\forall$$ can be understood as "for all". We can also see $$A$$ as counting the number of possible outcomes

$$P(A) = \frac{\text{number of outcomes in } A}{\text{number of outcomes in }\Omega} = \frac{\mid A \mid}{\mid \Omega \mid}$$

where $$  \mid \lbrace A \rbrace \mid $$ is the counting operator, meaning we count the number of elements in $$A$$. Now, to make statements as in the introduction, we need to relate two events to each other. One way to do this is with the **intersection** $$\cap$$ operator. If we have two events, $$A$$ and $$B$$, the event $$C = A \cap B$$ is called $$A$$ and $$B$$. If we look at the events as booleans, then it is the logical AND statement, $$A$$ and $$B$$ need both to be true for $$C$$ to be true. Two events $$A$$ and $$B$$ are said to be **independent** if

$$ P(A \cap B) = P(A)P(B)$$

As an example we can take two consecutive coin flips. What is the probability that the first and the second throw yield a head each? We assume the coin is fair with $$P(H) = P(T) = \frac{1}{2}$$. Our sample space would be $$\Omega = \lbrace (H_1, H_2), (H_1, T_2), (T_1, H_2), (T_1, T_2) \rbrace$$, four equally likely outcomes. Here $$H_1$$ denotes the event of throwing a head on the first try. Now we can formulate our question as follows

$$ P(H_1 \cap H_2) = P(H_1) P(H_2) = \frac{1}{2} \frac{1}{2} = \frac{1}{4}$$

This looks fairly clean, but what is actually hiding behind these expressions is the following

$$ P(H_1 \cap H_2) = P( \lbrace (H_1, H_2), (H_1, T_2) \rbrace \cap \lbrace (H_1, H_2), (T_1, H_2) \rbrace) = \frac{\mid \lbrace (H_1, H_2), (H_1, T_2) \rbrace \mid}{\mid \Omega \mid} \frac{\mid \lbrace (H_1, H_2), (T_1, H_2) \rbrace \mid}{\mid \Omega \mid} = \frac{2}{4} \frac{2}{4} =  \frac{1}{4} $$

What if we would like to know the probability of an event $$A$$ conditioned on another event $$B$$ happening? This is what we call a **conditional probability**, and is defined as

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} \qquad P(B) > 0.$$

Notice we can write this differently

$$ P(A \mid B)P(B) = P(A \cap B) = P( B \mid A) P(A)$$

and if we only take the right and left hand side, multiply both by $$\frac{1}{P(B)}$$ we get **Bayes formula**

$$ P(A \mid B) = \frac{P( B \mid A) P(A)}{P(B)}$$

Even though Bayes formula is simple to derive, it is one of the most frequently used results in probability and beyond.

# Children and Birthdays

# Monty Hall

```python
def calculate_probability():
   return 1.0
```
