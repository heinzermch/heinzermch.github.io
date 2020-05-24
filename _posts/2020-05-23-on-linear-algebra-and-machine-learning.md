---
layout: post
author: Michael Heinzer
title:  "On Linear Algebra and Machine Learning"
description: Bottom up introduction to linear algebra for machine learning, revising important concepts
date:   2020-05-22 11:00:00 +0530
categories: LinearAlgebra Algebra Matrix Group Field Determinant
comments: no
published: False
---
A look on linear algebra, from scratch. In a very mathematical way.

## Basic concepts and notation

Before we start, let us quickly repeat some basic concepts and their notation. Readers familiar with the topic may skip this section. This is not meant to be an introduction to probability theory or other mathematical concepts, only a quick refresh of what we will need later on.

- **Definition**: If a term on the left hand side of the equation is defined as as the right hand side term, then $$:=$$ will be used. This is similar to setting a variable in programming. As an example we can set $$g(x)$$ to be $$x^2$$ by writing $$g(x) := x^2$$. In mathematics, when writing simply $$=$$ means that the left side implies the right (denoted by $$\Rightarrow$$) and right side the left (denoted by $$\Leftarrow$$), at the same time.

- 

- 

# Theory

## 1. Introductory Algebraic Concepts

Let us start the tour with some fundamental mathematical concepts and some of their properties.

#### **Definition 1.1**: Group

A set $$G$$ together with an operation $$\cdot$$

$$ \cdot : G \times G \longrightarrow G, \qquad (a,b) \longmapsto a \cdot b$$

is called a group if

- For all $$a,b \in G$$, $$a \cdot b \in G$$ (closure)
- For all $$a,b,c \in G$$, we have that $$a \cdot (b \cdot c) = (a \cdot b ) \cdot c$$ (Associativity)
- There exists an element $$e \in G$$ such that, for every element $$a \in G$$ we have that $$a \cdot e = a$$ (identity element)
- For all $$a \in G$$ there exists an element $$b \in G$$ such that $$a \cdot b = e$$ (Inverse element)

and denoted by $$(G, \cdot)$$.

We usually denote the inverse element by $$a^{-1}$$ if the operation $$\cdot$$ is multiplication or $$-a$$ if the operation $$+$$. Some simple (non-)examples are:

- $$(\mathbb{Z},+)$$ the integers together with the plus operation.
- $$(\mathbb{N},+)$$ is not a group, there are no inverse elements in the natural numbers. $$e=0, +1-1=0$$
- $$(\mathbb{R},\cdot)$$ the real numbers together with the multiplication operation.
- $$(\mathbb{Z},\cdot)$$ however is not a group, there are no inverse elements for multiplication. $$e=1, 2\frac{1}{2} = 1$$

All of those examples have a special property which is not true in general, in their case we have that if $$a,b \in G$$ then $$a \cdot b = b \cdot a$$. This property has a name:

#### **Definition 1.2**: Abelian

A group $$(G,\cdot)$$ is called abelian, if for all $$a,b \in G$$, we have that $$a \cdot b = b \cdot a$$.

A more complex, but also very natural concept is when we connect a set with two operations:

#### **Defintion 1.3**: Field

A set $$F$$ together with two operations:

$$\begin{align} + : F \times F &\longrightarrow F  \\ 
+(a,b) &\longmapsto a + b \end{align}$$

$$\begin{align} \cdot : F \times F & \longrightarrow F  \\ 
\cdot(a,b) &\longmapsto a \cdot b  \end{align}$$

is called a Field if:

1.  $$(F,+)$$ is an abelian group. We denote the inverse of $$a \in F$$ by $$-a$$ and the neutral element by $$0$$.
2. Let $$F^* := F \setminus \lbrace 0 \mathbb{R}brace$$. Then $$(F^*,\cdot)$$ is an abelian group. We denote the inverse of $$a \in F$$ by $$a^{-1}$$ and the neutral element by $$1$$.
3. The distributive laws have to hold, i.e. for $$a,b,c \in F$$ we have:
   $$a \cdot (b+c) = a \cdot b + a \cdot c \text{ and } (a+b)\cdot c = a \cdot c + b \cdot c$$

and denoted by $$(F,+,\cdot)$$ or more shortly by $$\mathbb{F}$$.

A simple example is again the real numbers $$\mathbb{R}$$ together with addition and multiplication. Our goal is however the next definition:

#### **Definition 1.4**: Vector Space

Let $$F$$ be a field. A set $$V$$ together with an inner operation 

$$\begin{align} + : V \times V &\longrightarrow V  \\ 
+(v,w) &\longmapsto v + w \end{align}$$

called addition and an outer operation 

$$\begin{align} \cdot : F \times V &\longrightarrow V  \\ 
\cdot(\lambda,v) &\longmapsto \lambda \cdot v \end{align}$$

called scalar multiplication is called $$F$$-Vector space if the following holds:

1. $$(V,+)$$ is an abelian group, the neural element is called zero-vector and denoted by $$0$$. The negative is denoted by $$-v$$.
2. For all $$\lambda, \mu \in F$$ and $$v,w \in V$$, multiplication by scalars has the following properties:
   1. $$(\lambda + \mu)\cdot v = \lambda\cdot v + \mu\cdot v$$
   2. $$\lambda\cdot (v+w) = \lambda\cdot v + \lambda\cdot w$$
   3. $$\lambda \cdot( \mu\cdot v) = (\lambda\cdot \mu )\cdot v$$
   4. $$1\cdot v = v$$

The most common example of a vector space is probably $$V=\mathbb{R}^n$$ and $$F=\mathbb{R}$$ where $$R^n = \lbrace x = (x_1, \dotsc, x_n) : x_i \in \mathbb{R} \rbrace$$. With the following operations:

$$\begin{align}
    (x_1, \dotsc, x_n) + (y_1, \dotsc, y_n) := & (x_1+y_1, \dotsc, x_n+y_n)\\
       \lambda \cdot (x_1, \dotsc, x_n)  := & (\lambda x_1, \dotsc, \lambda x_n)
\end{align}$$

We can also see the space of matrices $$M(n \times m, \mathbb{F})$$ as a vector space. Let $$n, m \in \mathbb{N}, A,B \in M(n \times m, \mathbb{F})$$ and $$\lambda \in \mathbb{F}$$

$$\begin{align}
    A+B := & (a_{ij}+b_{ij})  \\
       \lambda A  := & (\lambda a_{ij}) 
\end{align}$$

Sometimes we will only have subsets of a vector space, then the question will be which structure it has. An import one is the 

#### **Definition 1.5**: Subvector Space

Let $$V$$ be a $$F$$-vector space and $$W \subset V$$ a subset. $$W$$ is called a subvector space if

1. $$W \neq \emptyset$$ (non-empty)
2. If $$v,w \in W$$ then $$ v+w \in W$$ (closed under addition)
3. If $$v \in W, \lambda \in \mathbb{F}$$ then $$\lambda v \in W$$ (closed under scalar multiplication)

One can see that if $$A \in M(n \times m, \mathbb{R})$$ then the set of solutions

$$W := \lbrace x \in \mathbb{R}^n : Ax = 0 \rbrace$$

is a sub vector space of $$\mathbb{R}^n$$. Let us now have a closer look at elements in a vector space. Suppose we have elements $$v_1, \dotsc, v_n \in V$$ with field $$\mathbb{F}$$. We could create a new element with them:

#### **Definition 1.6**: Linear Combination

Let $$v \in V$$, $$v$$ is called a linear combination of $$v_1, \dotsc, v_n \in V$$ if there exist $$\lambda_1, \dotsc, \lambda_n \in \mathbb{F}$$ such that:

$$v = \lambda_1 v_1 + \dotsc + \lambda_n v_n$$

Or could define the space which is generated by all the linear combinations of $$v_1, \dotsc, v_n$$. This space has a name:

#### **Definition 1.7**: Span

Let $$v_1, \dotsc, v_n \in V$$, the span of $$(v_1, \dotsc, v_n)$$  is defined as:

$$\text{span}_{\mathbb{F}}(v_1, \dotsc, v_n) := \lbrace v \in V : \exists \lambda_1, \dotsc, \lambda_n \in \mathbb{F} \text{ s.th. } v = \lambda_1 v_1 + \dotsc + \lambda_n v_n \rbrace$$

Sometimes we drop the subscript $$\mathbb{F}$$ if it is clear which field we are using. Some consequences of those definition are:

#### **Remark 1.8**: Span and Subvector space

Let $$V$$ be a $$\mathbb{F}$$ vector space and $$v_1,\dotsc,v_n \in V$$. Then

- $$\text{span}(v_1,\dotsc,v_n) \subset V$$ is a sub vector space.
- If $$W \subset V$$ is a sub vector space and $$v_i \in W$$ for all $$i \in \lbrace 1, \dotsc, n \rbrace$$ then $$\text{span}(v_1,\dotsc,v_n) \subset W$$.

Or told differently, $$\text{span}(v_1,\dotsc,v_n)$$ is the smallest sub vector space which contains all $$v_1,\dotsc,v_n$$.

#### **Definition 1.9**: Linear independence

Let $$V$$ be a $$\mathbb{F}$$-Vector space. A family of vectors $$(v_1,\dotsc,v_n), v_i \in V $$ is called linearly independent if:

$$\lambda_1,\dotsc,\lambda_n \in \mathbb{F} \text{ and } \lambda_1 v_1 + \dotsc + \lambda_n v_n = 0$$

then it follows that

$$ \lambda_1 = \dotsc = \lambda_n = 0.$$

The opposite exists of course as well: We call a family of vectors $$(v_1,\dotsc,v_n), v_i \in V $$ linearly dependent, if there exist $$\lambda_1,\dotsc,\lambda_n \in \mathbb{F}$$ with $$\lambda_i \mathbb{N}eq 0$$ for some $$i$$ such that $$\lambda_1 v_1 + \dotsc + \lambda_n v_n = 0$$. For $$n\geq 2$$ we can characterize linear independence more comfortably:

#### **Remark 1.10**: Linear Independence and Linear Combination

For $$n \geq 2$$ a family of vectors $$(v_1,\dotsc,v_n)$$ is linearly dependent if and only if one is a linear combination of the others.

We now have almost all the tools which are necessary for the definition of a basis and the related concept of a dimension of a vector space.

#### **Definition 1.11**: Generating Set

A family of vectors $$B = (v_1,\dotsc,v_n)$$ in a vector space $$V$$ is called generating set of $$V$$ if 

$$V = \text{span} (v_1,\dotsc,v_n)$$

The set of $$n$$ vectors need not be linearly independent to be a generating system, we just need to be able to describe every element inside $$V$$ with elements of the generating set. If we add linear independence however we have a new concept:

#### **Definition 1.12**: Basis

A family of vectors $$B = (v_1,\dotsc,v_n)$$ in a vector space $$V$$ is called a basis of $$V$$ if it is a generating system and linearly independent. We call $$n$$ the length of the basis.

There are theorems which proof that we can find a basis in any vector space, but this requires some more elaborate mathematics and is out of the scope of this short introduction. One of the consequences is that any two basis of a $$\mathbb{F}$$-Vector space have the same length. Which makes the following definition useful:

#### **Definition 1.13**: Dimension

Let $$V$$ be a $$\mathbb{F}$$ vector space, then we define:

$$\dim_{\mathbb{F}}(V) := n, \quad \text{ if } V \text{ has a basis of length }n$$

#### **Definition 1.14**: Norm on Vector Space

The norm $$ \| \cdot \| $$ on a vector space $$V$$ over a field $$F$$ of the real $$\mathbb{R}$$ or complex $$\mathbb{C}$$ numbers is a map from the vector space to the field

$$ \| \cdot \| : V \longrightarrow F $$

such that for all $$a \in F$$ and $$u, v \in V$$ we have

1. $$\| u+v \| \leq \| u \| + \| v \|$$ (triangle inequality)
2. $$\| a \cdot v \| = \mid a \mid \cdot \| v \|$$ (absolutely homogeneous)
3. If $$\| v \| = 0$$, then $$v = 0 \in V$$  is the zero vector

#### **Example**: Euclidean, Manhattan and Max Norm

For real valued vector spaces $$\mathbb{R}^n$$ some common norms are:

- The Euclidean Norm: $$\| x \|_2 := \sqrt{\sum^n_{i=1} x_i^2}$$
- The Manhattan Norm: $$\|x\|_1 := \sum^n_{i=1} \mid x_i \mid$$
- The Max Norm: $$\| x\|_{\infty} := \max( \mid x_1 \mid , \dotsc, \mid x_n \mid)$$

#### **Definition 1.15**: Dot or Inner Product

For two real values vectors $$x,y \in \mathbb{R}^n$$ the dot or inner product is defined as

$$\sum^n_{i=1} x_i y_i = x^Ty$$

#### **Definition 1.16**: Orthogonal Vectors

Two vectors $$x,y \in \mathbb{R}^n$$ are orthogonal to each other if 

$$x^Ty=\sum^n_{i=1} x_i y_i =0$$

#### **Example**: Cosine Similarity

The angle $$\theta$$ between two vectors $$x, y \in \mathbb{R}^n$$ has the property

$$ \cos(\theta) = \frac{x^Ty}{\|x\| \|y\|} =  \frac{ \sum^n_{i=1} x_i y_i}{ \sqrt{\sum^n_{i=1} x_i^2}  \sqrt{\sum^n_{i=1} y_i^2}}$$

this is called the cosine similarity. The correlation coefficients of two random variables can be seen as the cosine of the angle between them in Euclidean space.

## 2. Linear Maps

Now that we have a clearer image of what a vector space, its basis and dimension are, we can explore the concept of a linear map between vector spaces and how they relate to matrices. A map relates one or more elements for a set $$X$$ to one or more elements $$Y$$. Let us start with some basic concept about maps:

#### **Definition 2.1**: Injective, Surjective, Bijective

Let $$f : X \longrightarrow Y$$ be a map. It is called

- injective if for all $$ x,y \in X, x \neq y$$ it follows that $$f(x) \neq f(y)$$
- surjective if for all $$ y \in Y$$ there exists $$x \in X$$ such that $$f(x) = y$$
- bijective if it is surjective and injective.

If a map is bijective, it is a one to one correspondence between all elements in $$X$$ and $$Y$$. A simple example of a bijective map is $$ f : \mathbb{R}\longrightarrow \mathbb{R}$$, $$f(x)=x$$.

#### **Definition 2.2**: Linear Map / Homomorphism

Let V and W be two $$\mathbb{F}$$-vector spaces. A map $$ F: V \longrightarrow W$$ is called linear/homomorphism if

1. $$F(v+w) = F(v)+F(w)$$ for all $$v,w \in V$$
2. $$F(\lambda v) = \lambda F(v)$$ for all $$v \in V, \lambda \in \mathbb{F}$$

Linear maps have some simple properties:

#### **Remark 2.3**: Properties of Linear Maps

Let $$F: V \longrightarrow W$$ be a linear map between vector spaces $$V,W$$. Then the following holds:

- $$F(0) = 0$$ and $$F(v-w) = F(v)-F(w)$$.
- If $$v_1,\dotsc,v_n$$ is linearly dependent in $$V$$, then the family of $$(F(v_1),\dotsc,F(v_n))$$ is linearly dependent in $$W$$.
- $$\dim F(V) \leq \dim V$$.

Another important concept of maps is their image and kernel, we are going to link them back to the dimension of a vectorspace:

#### **Definition 2.4**: Image and Kernel

Let $$F : V  \longrightarrow W $$ be a linear map, then we call

- $$Im(F) := F(V) = \lbrace w \in W | \exists v \in V \text{ s.th. } F(v) = w\rbrace$$
- $$Ker(F) := F^{-1}(0) = \lbrace v \in V | F(v) = 0 \in W \rbrace$$

We not that image and kernel are not part of the same vector space: $$Im(F) \subset W$$ and $$Ker(F) \subset V$$. One can show that they are subvectorspaces of W and V respectively. We can also relate them to the concepts of injectivity and surjectivity if the application is linear:

#### **Remark 2.5**: Relationship between surjective/injective and Image/Kernel

$$F : V  \longrightarrow W $$ is linear, then 

- $$F$$ is surjective if and only if $$\text{Im}(F)=W$$
- $$F$$ is injective if and only if $$\text{Ker}(F) = \lbrace 0 \rbrace$$

An important number for a linear map is the dimension of its image, we call that the rank:

#### **Definition 2.6**: Rank

Let $$F : V  \longrightarrow W $$ be a linear application, we define the rank of $$F$$ as follows:

$$\text{Rank}(F) := \dim \text{Im}(F)$$

#### **Definition 2.7**: Matrix

A matrix $$A \in M(m \times n, \mathbb{F})$$ describes a linear map with

$$A : \mathbb{F}^n \longrightarrow \mathbb{F}^m, \: x \longmapsto y = Ax$$

where $$x \in \mathbb{F}^n$$ and $$y \in F^m$$ are vectors. 

The rank of $$A$$ is the rank of the above mentioned linear map. If $$(e_1,\dotsc,e_n)$$ are the a basis of $$\mathbb{F}^n$$, then the columns of $$A$$ are $$Ae_1, \dotsc, Ae_n$$ and hence 

$$\text{Im}(A) = A(\mathbb{F}^n) = \text{span}(Ae_1,\dotsc,Ae_n).$$

The space spanned by the columns. The rank of $$A$$ can be calculated simply by bringing $$A$$ into the row echelon form.

#### **Example**: Matrix in $$\mathbb{R}$$

The most commonly encountered matrix is of the form $$A \in M(m \times n, \mathbb{R})$$, where $$n$$ are the number of columns, $$m$$ the number of rows and $$a_{ij} \in \mathbb{R}$$:

$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1n} \\
 a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{pmatrix} $$

#### **Example**: Standard basis in $$\mathbb{R}^n$$

If we talk of the standard basis in $$\mathbb{R}^n$$, then this is a set of vectors in the following form:

$$e_1 = \begin{pmatrix}
 1  \\
 0  \\
\vdots   \\
 0  \\
\end{pmatrix}, e_2 = \begin{pmatrix}
 0  \\
 1  \\
\vdots   \\
 0  \\
\end{pmatrix}, \cdots, e_n = \begin{pmatrix}
 0  \\
0  \\
\vdots   \\
 1  \\
\end{pmatrix}$$

It is simple to see that  $$\text{span}(e_1,\dotsc,e_n) = \mathbb{R}^n$$.

#### **Theorem 2.8**: Rank-Nullity

Let $$V,W$$ be vector spaces, $$\dim(V)<\infty$$ and $$F : V  \longrightarrow W $$ linear. Then we have that

$$ \dim(V) = \dim (\text{Im}(F)) + \dim (\text{Ker}(F))$$

#### **Corollary 2.9**:

Let $$V,W$$ be two vector spaces with $$\dim(V) = \dim(W)$$ and $$F : V  \longrightarrow W $$ a linear map. Then the following are equivalent:

- $$F$$ is injective
- $$F$$ is surjective
- $$F$$ is bijective

This is of course also the case if $$V=W$$, this is a special, and later on important, case of homomorphism. It has its own definition:

#### **Definition 2.10**: Endomorphism

If we have the case that $$V=W$$, then we call a linear map $$F : V  \longrightarrow V $$ an endomorphism. Furthermore we define the space of linear maps from $$V$$ to $$V$$ as:

$$\text{End}(V) = \lbrace F \:| F : V  \longrightarrow V  \text{ linear} \rbrace$$

## 3. Matrices and Systems of Linear Equations

A related topic which be important later are matrices and systems of linear equations. Generally, a matrix $$A \in M(m \times n, \mathbb{F})$$ and a vector $$b \in M(m \times 1, \mathbb{F})$$ form the system

$$Ax = b \text{ or } \sum^n_{j=1} a_{ij}x_j = b_i \text{ for } i=1,\dotsc,m$$

The associated system $$Ax=0$$ is called homogeneous system. The set

$$\text{Sol}(A,b) := \lbrace x \in \mathbb{F}^n | Ax=b \rbrace \subset \mathbb{F}^n$$

is called set of solutions. We are interested in the relationship between the implied linear map by $$A$$:

$$F : \mathbb{F}^n  \longrightarrow \mathbb{F}^m, \: F(x) \longmapsto Ax$$

as we have $$\text{Sol}(A,b) = F^{-1}(b)$$, and especially due to $$\text{Sol}(A,0) = \text{Ker}(F)$$. The size of the solution space is then given by $$r:= \text{Rank}(F) = \text{Rank}(A)$$. We compare the matrix $$A$$ of rank $$r$$ and the augmented matrix $$(A,b)$$, $$A$$ concatenated with $$b$$. As the augmented matrix has a column more, we have that

$$r \leq \text{Rank} (A,b) \leq r +1$$

The following theorem tells us a condition for a non-zero solution:

#### **Theorem 3.1**: Non-empty solution

The solution space of a linear system of equations $$Ax=b$$ is non empty only if

$$\text{Rank}(A) = \text{Rank} ((A,b))$$

The theorem gives us a condition for which one or many solutions exist. We can have $$n < m$$. The following remark tells tells us when a solution is unique.

#### **Remark 3.2**: 

For $$A \in M(m \times n, \mathbb{F})$$ and $$b \in \mathbb{F}^m$$ the following conditions are equivalent:

- The linear system $$Ax=b$$ has a unique solution.
- $$\text{Rank}(A) = \text{Rank}((A,b))= n$$

In the case of $$m=n$$ the condition $$\text{Rank}(A)=n$$ is sufficient. This means that the linear map $$A: \mathbb{F}^n \longrightarrow \mathbb{F}^n$$ is surjective, and according to the above corollary even bijective. If $$A^{-1}$$ is the inverse map, the unique solution is given by $$x = A^{-1}b$$ where $$A^{-1}$$ is the inverse matrix.

## 4. Linear Maps and Matrices

In general a map $$F: X \longrightarrow Y$$ is a rule which assigns to each $$x \in X$$ a value $$F(x) = y \in Y$$. If we know the rule for some elements, we can in general not make inferences about others. The case is different for linear maps between vector spaces $$F : V  \longrightarrow W $$. Once we know the value for some vector $$F(v) \in W$$, the whole line  $$\mathbb{F} v$$ is fixed. If we want to fix the value for another $$v' \in V$$, it can not lie on the line $$\mathbb{F} v$$. The question is now how many of those rules define a linear map?

#### **Theorem 4.1**: Linear Maps and Linear Independence

Let $$V,W$$ be vector spaces with $$\dim(V)< \infty, \dim(W) < \infty$$ and let $$v_1,\dotsc, v_n \in V$$, $$w_1,\dotsc,w_n \in W$$. Then

- If $$v_1,\dotsc,v_r$$ are linearly independent, then there exists a linear map

  $$F: V \longrightarrow W \text{ such that } F(v_i)=w_i \text{ for } i=1,\dotsc,r$$

- If $$(v_1,\dotsc,v_r)$$ is a basis, then there exist exactly one linear map 
  $$F: V \longrightarrow W \text{ such that } F(v_i)=w_i \text{ for } i=1,\dotsc,r$$
  which has the following properties:

  - $$\text{Im}(F) = \text{span}(w_1,\dotsc,w_n)$$.

  - $$F$$ injective if and only if $$w_1,\dotsc,w_n$$ are linearly independent.

Now we can create a one to one correspondence between linear maps and matrices with the following corollary.

#### **Corollary 4.2**: One to One Correspondence

For each linear map $$F :\mathbb{F}^n \longrightarrow \mathbb{F}^m$$ there exists exactly one matrix $$A \in M(m \times n, \mathbb{F})$$ such that

$$F(x) = Ax$$

for all $$x \in \mathbb{F}^n$$.

This connection not only exists in $$\mathbb{F}^n$$ spaces but also in more generally in $$\mathbb{F}$$-vector spaces. Once we fix a basis $$\B = (v_1,\dotsc,v_n)$$ the linear combinations of

$$F(v_j) = \sum^m_{i=1} a_{ij} w_i \qquad \text{for } j = 1,\dotsc,n$$

are fixed, and hence also the columns of $$A$$. As matrices are so tremendously important, let us have a quick look at some matrix concepts. One of the most important ones is the transposition.

#### **Definition 4.3**: Transposed of a Matrix

Let $$A = (a_{ij}) \in M(m \times n, \mathbb{F})$$ then the transposed is

$$A^T = (a^T_{ij}) \in M(n \times m, \mathbb{F}) \text{ with } a^T_{ij} = a_{ji}$$

There are some rules which apply to matrix transpositions: 

#### **Remark 4.5**: Rules for calculations with transposed matrices

- $$(A+B)^T = A^T + B^T$$
- $$(\lambda A)^T = \lambda A^T$$
- $$ (A^T)^T = A$$
- $$(AB)^T = B^TA^T$$

We can use that definition to introduce the concept of symmetric matrices, a case in which the intuitive meaning and the mathematical definition are quite similar

#### **Definition 4.6**: Symmetric Matrix

A matrix $$A \in M(n \times n, \mathbb{F})$$ is called symmetric, if 

$$A^T = A$$

In other words a matrix is symmetric, if we can mirror it around the diagonal and it does not change, for the case of $$M( n \times n, \mathbb{F}) $$ matrices. 

**Definition 4.7**: Diagonal Matrix

A matrix $$A \in M(n \times n, \mathbb{F})$$ is called diagonal if $$a_{ij} = 0$$ for $$\forall  i \neq j$$ where $$i,j \in \lbrace 1, \dotsc, n \rbrace$$. It is of the form

$$A=\begin{pmatrix}
 a_{11} & 0 & \cdots & 0 \\
 0 & a_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
 0 & 0 & \cdots & a_{nn} \\
\end{pmatrix} $$

An important example of a symmetric and diagonal matrix is the unit matrix. It has ones in the diagonal and zeros everywhere else:

$$\begin{pmatrix}
1 & & 0\\ 
& \ddots & \\
0 & & 1
\end{pmatrix} $$

#### **Definition 4.8**: Triangular Matrix

A matrix $$A \in M(n \times n, \mathbb{F})$$ is called

- upper triangular matrix if $$\forall i > j$$ we have that $$ a_{ij} = 0$$: 

  $$\begin{pmatrix}
   a_{11} & a_{12} & \cdots & a_{1n} \\
   0 & a_{22} & \cdots & a_{2n} \\
  \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & a_{nn} \\
  \end{pmatrix} $$

- lower triangular matrix if $$\forall i < j$$ we have that $$ a_{ij} = 0 $$:

   $$\begin{pmatrix}
   a_{11} & 0 & \cdots & 0 \\
   a_{21} & a_{22} & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
   a_{m1} & a_{m2} & \cdots & a_{nn} \\
  \end{pmatrix} $$



This brings us to the concept of an invertible matrix, a matrix for which an inverse exists.

#### **Definition 4.9**: Invertible

A matrix $$A \in M(n \times n, \mathbb{F})$$ is invertible if there exists a matrix $$B \in M(n \times n, \mathbb{F})$$ such that

$$A \cdot B= I_n = B \cdot A .$$

We usually denote $$B$$ by $$A^{-1}$$.

In most cases it is not obvious that such a matrix even exists, or how it looks like. However there are certain tools which allow us to to make a statement. One of them is the rank:

#### **Remark 4.8**: Invertible and Rank

A matrix $$A \in M(n \times n, \mathbb{F})$$ is invertible if and only if $$\text{Rank}(A) = n$$

The same applies to the transposed of the matrix, $$A$$ is invertible if and only if $$A^T$$ is invertible as one can see by the following reasoning:

$$(A^{-1})^TA^T=(AA^{-1})^T=(I_n)^T=I_n$$

#### **Definition 4.9**: Orthogonal

A matrix $$A \in M(n \times n, \mathbb{F})$$ is orthogonal if

$$A^TA=AA^T=I$$

or equivalently if its transposed is equal to the inverse: $$A^T = A^{-1}$$.

Orthogonal matrices are an important concept for many matrix decompositions we will see later on.

#### **Definition 4.9**: General Linear Group

The set

$$Gl_n(\mathbb{F}) := \lbrace A \in M(n \times n, \mathbb{F}) : A \text{ invertible} \rbrace$$

is called the general linear group.

It is a group in the sense of definition $$1.1$$ with the matrix multiplication. The neutral element is the matrix $$I_n$$.


In general matrices behave almost like numbers, but without the abelian property, but we have to be careful that the dimensions match.

#### **Remark 4.10**: Rules for Matrix Multiplication

Let $$A,A' \in M(m \times n, \mathbb{F})$$ and $$B,B' \in M( n \times r, \mathbb{F})$$, $$C \in M(r \times s, \mathbb{F})$$ and $$\lambda \in \mathbb{F}$$. Then the following applies:

- $$A(B+B') = AB + AB'$$ and $$(A+A')B = AB +A'B$$ (distributive law)
- $$A(\lambda B) = (\lambda A)B = \lambda (AB)$$ and $$(AB)C=A(BC)$$ (assoctiativity)
- $$I_m A = A I_n = A$$

There are some important tools to determine certain properties for matrices, we will look at two of them in more detail: trace and determinant. The first one is defined as follows:

#### **Definition 4.11**: Trace of a Matrix

The trace of a matrix $$A \in M(n \times n, \mathbb{F})$$ is defined as

$$\text{tr}(A) = \sum^n_{i=1} a_{ii}$$

In other words the trace is the sum of the diagonal elements of a matrix. There are some properties which will be useful later:

#### **Remark 4.12**: Properties of the Trace

For $$A,B,C \in M(n \times n, \mathbb{F})$$, $$\lambda \in \mathbb{F}$$ we have that:

- $$\text{tr}(A) = \text{tr}(A^T)$$
- $$\text{tr}(A+B) = \text{tr}(A) + \text{tr}(B)$$
- $$\text{tr}(\lambda A ) = \lambda \text{tr}(A)$$
- $$\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$$ (invariant under cyclic permutation)

Those definitions and properties also hold for matrices which are not square. The determinant is a bit harder to define and calculate: 

#### **Definition 4.13**: Determinant of a Matrix

Let $$ \det : M(n \times n, \mathbb{F}) \longrightarrow \mathbb{F}$$ be a map with the following properties:

- Alternate
- Normed
- Linear in the rows

The existence and uniqueness are far from trivial to prove, we will skip this here and instead focus on the properties of the determinant

#### **Example**: Determinant calculations

In the case of a two by two matrix $$A \in M(2 \times 2, \mathbb{R})$$ the determinant is

$$\det(A) = \det \begin{pmatrix}
 a_{11} & a_{12}  \\
 a_{21} & a_{22} 
\end{pmatrix} = a_{11}a_{22} - a_{12}a_{21}$$

In the case of a upper triangular matrix $$U \in M(n \times n, \mathbb{R})$$ the determinant is

$$\det(U) = \det \begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1n} \\
 0 & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
 0 & 0 & \cdots & a_{nn} \\
\end{pmatrix} = \prod^n_{i=1} a_{ii}$$

The same applies to a lower triangular matrix.

#### **Remark 4.xx**: Determinant of an Orthogonal Matrix

If a matrix $$A \in M(n \times n, \mathbb{F})$$ is orthogonal, then its determinant is either $$+1$$ or $$-1$$ by the following reasoning

$$1 = \det(I) = \det(AA^{-1})=\det(A)\det(A^{-1})=\det(A)\det(A^T)=\det(A)\det(A)=\det(A)^2$$

The only number $$r \in \mathbb{R}, r \neq 0$$ for which the property $$r^2=1$$ holds is one.

#### **Definition 4.xx**: Positive (Semi) Definite

A real matrix $$A \in M(n \times n, \mathbb{R})$$ is positive definite if 

$$ x^TAx > 0 \quad \forall x \in \mathbb{R}^n$$

It is positive semi-definite if the inequality is not strict, i.e. if 

$$ x^TAx \geq 0 \quad \forall x \in \mathbb{R}^n$$



## 5. Eigenvectors and Eigenvalues

Suppose we have an endomorphism $$F$$ and we would like to find a basis $$\mathcal{A}$$ such that the corresponding matrix $$A$$ for $$F$$ is as simple as possible. In matrix terms this means we want $$A$$ to be diagonal. The goal of this chapter is now to introduce concepts to be able to check when and how we can diagonalize an endomorphism or a matrix.

#### **Definition 5.1**: Eigenvalues and Eigenvectors

Let $$F : V \longrightarrow V$$  be an endomorphism. A scalar $$\lambda \in K$$ is called eigenvalue of $$F$$ if $$\exists v \in V, v \neq 0$$ s.th. $$F(v) = \lambda v$$. The vector $$v$$ is called eigenvector of $$F$$ corresponding to $$\lambda$$.

It is of course possible that $$0 \in K$$ is an Eigenvalue, but $$0 \in V$$ can never be an eigenvector according to the definition. Also not every non-zero vector is automatically an eigenvector. We can already say something how eigenvectors corresponding to different eigenvectors relate to each other:

#### **Lemma 5.2**: Eigenvalues and Linear Independence

If $$\lambda_1, \dotsc, \lambda_n$$ are mutually distinct eigenvalues and $$v_i$$ is a corresponding eigenvector to $$\lambda_i$$. Then $$\lbrace v_1, \dotsc, v_n \rbrace$$ are linearly independent.

Hence if we could find $$n$$ distinct eigenvalues for an endomorphism $$F \in End(V)$$ with $$\dim(V)=n$$, then the corresponding eigenvectors would form a basis of $$V$$. This is exactly what motivates the following definition:

#### **Definition 5.3**: Diagonalizable Endomorphism

An endomorphism $$F \in End(V)$$ is diagonalizable if there exists a basis $$\mathcal{B}$$ of $$V$$ consisting of eigenvectors of $$F$$. A matrix $$A$$ is diagonalizable if the associated endomorphism

#### **Remark 5.4**:

If $$\dim(V)=n < \infty$$, then $$F \in End(V)$$ is diagonalisable if and only if there exists a basis $$\mathcal{B} = \lbrace v_1,\dotsc,v_n \rbrace$$ such that $$\mathcal{M}_{\mathcal{B}}(F)$$ is a diagonal matrix, i.e.

$$ \mathcal{M}_{\mathcal{B}}(F)
=\begin{pmatrix}
\lambda_1 & & 0\\ 
& \ddots & \\
0 & & \lambda_n
\end{pmatrix} $$

For a matrix, we have that the condition is equivalent to another concept.

#### **Remark 5.5**:

We have that for a matrix $$A \in M(n \times n, K)$$ is diagonalizable if and only if there exists $$S \in GL_n(n, K)$$ such that: 

$$ SAS^{-1}
=\begin{pmatrix}
\lambda_1 & & 0\\ 
& \ddots & \\
0 & & \lambda_n
\end{pmatrix} $$

Due to the lemma above, we can immediately formulate a sufficient, but not necessary condition for an endomorphism to be diagonalizable:

#### **Theorem 5.6**:

Let $$F \in End(V)$$ and $$\dim(V)=n$$. If $$F$$ has $$n$$ pairwise distinct eigenvalues $$\lambda_1, \dotsc, \lambda_n$$, then $$F$$ is diagonalizable.

This follows immediately from the lemma $$2.2$$. As we have seen above, there exist maximally $$n = \dim(V)$$ eigenvalues, but possibly many more eigenvectors. Hence it is useful to summarize them in one object:

#### **Definition 5.7**: Eigenspace

Let $$F \in End(V)$$ and $$\lambda\in K$$ an eigenvalue of $$F$$. Then we call 

$$ Eig(F,\lambda) := \lbrace v \in V | F(v) = \lambda v \rbrace$$ 

the eigenspace of $$F$$ corresponding to $$\lambda$$.

#### **Remark 5.8**: Properties of the Eigenspace

Some properties of eigenspaces can be shown quickly:

1. $$Eig(F, \lambda) \subset V$$ is a subvector space.
2. If $$\lambda$$ is an eigenvalue of $$F$$ if and only if $$Eig(F, \lambda) \neq \lbrace 0 \rbrace$$
3. $$Eig(F, \lambda) = Ker(F - \lambda id_V) = \lbrace v \in V | (F - \lambda id_V)(v) = 0 \rbrace$$
4. For $$\lambda_1, \lambda_2 \in K, \lambda_1 \neq \lambda_2$$ we have that $$Eig(F, \lambda_1) \cap Eig(F, \lambda_2) = \lbrace 0 \rbrace$$

Especially (3 will be useful, as we can immediately see that for $$\lambda \in K$$ which are eigenvalues of $$F$$, the kernel will not be empty. Hence the endomorphism $$(F-\lambda id_V) : V \longrightarrow V$$ is not injective. Which leads us to the following lemma:

#### **Lemma 5.9**: 

Let $$F \in End(V)$$. Then the following are equivalent:

- $$\lambda \in K$$ is an eigenvalue of $$F$$
- $$\det(F-\lambda id_V) = 0$$

The last term can be seen as a polynomial in $$\lambda$$ where the roots are eigenvalues of $$F$$. 

### 5.1 Characteristic Polynomial

With the last lemma, we came to see the search for eigenvalues as the same as the search for roots in the polynomial $$\det(F - \lambda id_V)$$. This concept is known as follows:

#### **Definition 5.10**: Characteristic Polynomial

Let $$V$$ be a vector space with $$\dim(V)=n$$, $$F \in End(V)$$ and $$A \in M(n \times n, \mathbb{K})$$: 

- $$p_F(t) = \det(F- t \cdot id_V)$$ is called characteristic polynomial of $$F$$.
- $$p_A(t) = \det(A -t \cdot I_n)$$ is called characteristic polynomial of $$A$$.

Suppose $$\mathcal{A}$$ is a basis of a vector space $$V$$ and $$A = M_{\mathcal{A}}(F)$$ the matrix which describes $$F$$ in basis $$\mathcal{A}$$. One could then show that $$p_A(t)$$ is an element of $$\mathbb{K}[t]$$, the polynomials with coefficients in $$\mathbb{K}$$. For $$F$$ we have that for all $$\lambda \in \mathbb{K}$$:

$$M_{\mathcal{A}}(F- \lambda \cdot id_V) = A - \lambda \cdot I_n$$

And therefore: 

$$\det(F - \lambda \cdot id_V) = \det(A - \lambda \cdot I_n) = p_A(\lambda).$$ 

This means the characteristic polynomial of $$F$$ is described by the characteristic polynomial of $$A$$.\\
If $$\mathcal{B}$$ is another basis of $$V$$, then $$B:= M_{\mathcal{B}}(F)$$ is similar to $$A$$. This is useful because:

#### **Lemma 5.11**:

Let $$A,B \in M(n \times n, \mathbb{K})$$, if $$A$$ and $$B$$ are similar, i.e. $$\exists S \in Gl_n(\mathbb{K})$$ s.th. $$B = SAS^{-1}$$, then $$p_B(t) = p_A(t)$$.

**Proof**: 

$$\begin{align}
    p_B(t) &=  \det(B-\lambda \cdot I_n)\\
       & =  \det(SAS^{-1}- \lambda \cdot I_n)\\
        &=   \det(S(A-\lambda \cdot I_n)S^{-1}) \\
       & =  \det(S)\det(A-\lambda \cdot I_n)\det(S^{-1}) \\
        &=  \det(A-\lambda\cdot I_n) \\
       & =  p_A(t)
\end{align}$$

Hence the characteristic polynomial of the descriptive matrix is independent of the basis. The above insights are summarized in the next proposition:

**Property 5.12**: 

Let $$V$$ be a $$\mathbb{K}$$ vector space of dimension $$n < \infty$$ and $$F \in End(V)$$. Then the characteristic polynomial $$p_F \in \mathbb{K}[t]$$ has the following properties:

1. $$\deg(p_F) = n$$
2. $$p_F : \mathbb{K} \longrightarrow \mathbb{K}$$ is $$\lambda \longmapsto \det(F - \lambda \cdot id_V)$$
3. The roots of $$p_F$$ are the eigenvalues of $$F$$
4. If $$A$$ is a matrix which describes $$F$$, then $$p_F(t) = \det(A -t \cdot I_n)$$. 

Once we have an eigenvalue $$\lambda \in \mathbb{K}$$, the corresponding eigenspace can be found by solving a system of equations:

#### **Remark 5.13**:

Let $$V \in End(V)$$ and $$A:=M_{\mathcal{B}}(F)$$ the associated matrix corresponding to the basis $$\mathcal{B}$$ of $$V$$. Then the eigenspace of $$A$$ corresponding to the eigenvalue $$\lambda \in \mathbb{K}$$ is the solution to the system of linear equations $$(A-\lambda I_n)x = 0$$, i.e

$$ Eig(A, \lambda) = \lbrace x \in \mathbb{K}^n | (A- \lambda I_n)x = 0 \rbrace$$

### 5.2 Diagonalization

The result of the previous subsections can be summarized as follows:

#### **Theorem 5.14**:  

Let $$F \in End(V)$$ with $$\dim(V)=n$$. Then we have the following:

- $$F$$ diagonalizable $$\Rightarrow p_F(t) = \pm (t-\lambda_1) \cdot \dotsc \cdot (t-\lambda_n)$$, i.e. the characteristic polynomial can be written as the product of linear factors.
- $$p_F(t) = \pm (t-\lambda_1) \cdot \dotsc \cdot (t-\lambda_n)$$ with pairwise distinct $$\lambda_1, \dotsc, \lambda_n \Rightarrow F$$ is diagonalizable.

How about the case when $$\lambda_i = \lambda_j$$ for some $$i,j \in \lbrace 1,\dotsc,n \rbrace$$, i.e. they are not pairwise distinct? Let us rewrite $$p_F$$ for that case as follows:

$$p_F(t) = \pm (t - \lambda_1)^{r_1} \cdot \dotsc \cdot (t - \lambda_k)^{r_k}$$

Where $$\lambda_1, \dotsc, \lambda_k$$ are pairwise distinct, $$1 \leq r_i \leq n$$ for all $$i \in \lbrace 1,\dotsc,k \rbrace$$ and $$r_1 + \dotsc + r_k = n$$. We call $$r_i$$ the multiplicity of the root $$\lambda_i$$ in $$p_F$$ and write $$r_i = \mu(p_F, \lambda_i)$$. We can relate the multiplicity to the dimension of the eigenspace:

#### **Lemma 5.15**: 

For $$\lambda \in \mathbb{K}$$ an eigenvalue of $$F \in End(V)$$, we have:

$$1 \leq \dim(Eig(F,\lambda)) \leq \mu(p_F, \lambda)$$

Due to those inequalities we can create a criteria which allows us to fully characterize if an endomorphism is diagonalizable:

#### **Theorem 5.16**:

Let $$V$$ be a vector space with $$\dim(V)=n<\infty$$ and $$F \in End(V)$$. Then the following three statements are equivalent:

1. $$F$$ is diagonalizable
2. The characteristic polynomial $$p_F$$ decays into linear factors and $$\dim(Eig(F,\lambda)) = \mu(p_F,\lambda)$$ for all eigenalues $$\lambda$$ of $$F$$.
3. If $$\lambda_i, \dotsc, \lambda_k$$ are pairwise distinct eigenvalues of $$F$$, then $$V = Eig(F, \lambda_1) \oplus \dotsc \oplus Eig(F,\lambda_k)$$

Hence we can find a simple process for the diagonalization of an endomorphism $$F$$ on a finite dimensional $$\mathbb{K}$$ vector space $$V$$:

1. With the help of a basis $$\mathcal{A}$$ of $$V$$ and the matrix $$A = M_{\mathcal{A}}(F)$$ we can calculate the characteristic polynomial. If the polynomial decays into linear factors, we can continue.

2. For every eigenvalue $$\lambda$$ of $$F$$ find a basis of $$Eig(F,\lambda)$$ by solving the system of linear equations. If

   $$\dim(Eig(F,\lambda)) = \mu(p_F, \lambda)$$

   for all $$\lambda$$, then $$F$$ is diagonalizable and the bases of the eigenspaces form a basis of $$V$$.



## 6. Matrix Decompositions

Often matrix decompositions allow us to simplify calculations and look at problems in a different way.

#### **Theorem 6.1**: QR Decomposition

Every non-singular matrix $$A \in M(n \times n, \mathbb{R})$$ can be decomposed into $$A = QR$$ where:

- $$Q \in M(n \times n, \mathbb{R})$$, $$Q$$ orthogonal
- $$R \in M(n \times n, \mathbb{R})$$ upper triangular with positive diagonal elements, i.e. $$r_{ii} > 0$$

QR decomposition is often used for solving linear systems of the form $$Ax=b$$ for non-singular $$A$$.

#### **Example**: Applying QR to Least Squares Regression

In linear least squares regression we often have an observation matrix $$X \in M(n \times m, \mathbb{R})$$ with $$n$$ observations for $$p$$ predictors or variables. Then we have a target vector $$y \in M(n \times 1, \mathbb{R})$$ which contains the values which we are trying to predict. The goal is now to find a weight vector $$\beta^* \in M(m \times 1, \mathbb{R})$$ which minimizes the least squares goal

$$\beta^* = \text{argmin}_{\beta} \| X\beta - y\|^2$$

This vector can be found by taking the derivative with respect to $$\beta$$ in the target function and setting it to zero:

$$ \frac{\partial}{\partial \beta} (X\beta - y)^2 = 0$$

Which gives us the following equation

$$(X^TX)\beta = X^Ty$$

If we now set $$A=X^TX$$ and $$b=X^Ty$$ then we get $$A\beta = b$$, a linear system of equations! Now we apply the QR decomposition to $$A$$ to get

$$QR\beta=b \Leftrightarrow R\beta = Q^Tb \qquad \text{ using orthogonality } Q^T=Q^{-1}$$

Because $$R$$ is upper triangular, we can solve the system of equations recursively starting with $$r_{mm}\beta^*_m = (Q^T)_m$$.

#### **Theorem 6.2**: LU Decomposition

Let $$A$$ be a non-singular matrix $$A \in M(n \times n, \mathbb{R})$$, then it can be decomposed into $$A=LU$$ where

- $$L \in M(n \times n, \mathbb{R})$$ a lower triangular matrix
- $$U \in M(n \times n, \mathbb{R})$$ a upper triangular matrix

#### **Remark 6.3**: Finding the LU Decomposition of a Matrix

We can find the LU decompostition of a non-singular matrix $$A \in M(n \times n, \mathbb{R})$$ by setting up the following system of equations:

$$\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1n} \\
 a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nn} \\
\end{pmatrix} = \begin{pmatrix}
 1 & 0 & \cdots & 0 \\
 l_{21} & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
 l_{n1} & l_{n2} & \cdots & 1 \\
\end{pmatrix}  \begin{pmatrix}
 u_{11} & u_{12} & \cdots & u_{1n} \\
 0 & u_{22} & \cdots & u_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
 0 & 0 & \cdots & u_{nn} \\
\end{pmatrix}$$

Note that the first row of $$U$$ we get for free, then we can proceed row by row.

#### **Example**: Calculating the Determinant with LU Decomposition

The calculation of the determinant of a matrix $$A$$ can be vastly simplified if the LU decomposition is known:

$$\det(A) = \det(LU) = \det(L)\det(U) = \prod^n_{i=1} l_{ii} \prod^n_{i=1} u_{ii}$$

#### **Example**: Solving a System of Linear Equations

If we have a linear system of equations $$Ax=b$$ and know the LU decomposition we can solve the system $$LUx=b$$ recursively by solving the two systems instead: $$Ux=y$$ and $$Ly=b$$.

#### **Theorem 6.4**: Cholesky Decomposition

Let $$A \in M(n \times n, \mathbb{R})$$ be a symmetric positive definite matrix, then it can be decomposed as $$A=R^TR$$ where $$R \in M(n \times n, \mathbb{R})$$ is an upper triangular matrix with $$r_{ii} > 0 \: \forall i$$.

#### **Remark 6.5**: Cholesky and LU Decomposition

The Cholesky can be seen as a LU decomposition with the additional requirement that $$L=U^T$$

#### **Example**: Cholesky Decomposition for Monte Carlo Simulations

Suppose we want to generate numbers $$x \in \mathbb{R}^n$$ coming from an $$n$$ dimensional multivariate normal distribution $$N(\mu, \Sigma)$$ with mean $$\mu \in \mathbb{R}^n$$ and covariance matrix $$\Sigma \in M(n \times n, \mathbb{R})$$ but have only access to simulated $$N(0,1)$$ numbers $$z \in \mathbb{R}$$.

Covariance matrices have the property that they are positive definite, so we can use the Cholesky decomposition to get $$\Sigma = R^TR$$ and simulate $$n$$ standard normal variables to get a vector $$z \in \mathbb{R}^n, z_i \sim N(0,1)$$. Putting this together gives us the ability to simulate from the multivariate normal by

$$x = \mu + R^Tz$$

#### **Example**: Cholesky Decomposition for Two Dimensional Normal Distribution

If we want to simulate two random variables $$x_1$$ and $$x_2$$ with correlation $$\phi$$ but only have a normal generator for $$z_1, z_2 \sim N(0,1)$$. We begin by setting up the covariance matrix and decomposing it into the $$R$$ matrix:

$$\Sigma = \begin{pmatrix}
 1 & \rho \\
 \rho & 1 
\end{pmatrix} = \begin{pmatrix}
 r_{11}  & 0 \\
 r_{12} & r_{22} 
\end{pmatrix} \begin{pmatrix}
 r_{11}  & r_{12} \\
 0 & r_{22} 
\end{pmatrix} = R^TR$$

Solving this yields:

$$\begin{align}
    1 &=  r_{11}^2 & \Leftrightarrow \qquad &r_{11}  =1\\
     \rho  & =  r_{11} r_{12} = 1r_{12}  & \Leftrightarrow\qquad &r_{12}  =\rho \\
       1 &=  r_{12}^2 + r_{22}^2 = \rho^2 + r_{22}^2  & \Leftrightarrow\qquad & r_{22}  = \sqrt{1-\rho^2} \\
       
\end{align}$$

and hence

$$\begin{pmatrix}
 x_1 \\
 x_2 
\end{pmatrix} = \mu + R^Tz = \begin{pmatrix}
 1 & 0 \\
 \rho & \sqrt{1-\rho^2} 
\end{pmatrix} \begin{pmatrix}
 z_1 \\
 z_2 
\end{pmatrix} = \begin{pmatrix}
 z_1 \\
 \rho z_1 + \sqrt{1-\rho^2}z_2 
\end{pmatrix}$$ 







# TODO

- Change of basis
- Examples positive definite
- Equivalence positive definite
- Examples Eigenvalue calculation, multiple approaches
- Add interview questions

# Conclusion

What we saw.

## References

A list of resources used to write this post, also useful for further reading:

- [Deep Learning](https://www.deeplearningbook.org/) Book by Goodfellow, Bengio and Courville
  - [Chapter 3](https://www.deeplearningbook.org/contents/prob.html) for Information Theory, softmax and softplus properties
  - [Chapter 5](https://www.deeplearningbook.org/contents/ml.html) for KL-Divergence, Maximum Likelihood Estimation
  - [Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) for Cross-Entropy and sigmoid/softmax discussion
- [Autoencoders](https://en.wikipedia.org/wiki/Autoencoder) Wikipedia

## Comments

I would be happy to hear about any mistakes or inconsistencies in my post. Other suggestions are of course also welcome. You can either write a comment below or find my email address on the [about page](https://heinzermch.github.io/about/).