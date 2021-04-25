---
title:  "머신러닝을 위한 수학 정리: Matrix Decomposition"
toc: true
toc_sticky: true
categories:
  - Machine Learning
tags:
  - linear algebra
  - vector
  - matrix
use_math: true
last_modified_at: 2021-04-24
---

본 포스트는 머신러닝에 필요한 선형대수 및 확률과 같은 수학적 개념을 정리한 포스트이다. 본 문서는 [mml](https://mml-book.github.io/book/mml-book.pdf)을 참고하여 정리하였다. 누군가에게 본 책이나 개념을 설명하는 것이 아닌, 내가 모르는 것을 정리하고 참고하기 위함이므로 반드시 원문을 보며 참고하길 추천한다.
{. :notice--info}


## Determinant and Trace

Determinant (행렬식)은 수학적 객체로, 해석과 선형 시스템의 해에서 사용된다. 이는 square matrix에서만 정의된다. 본 책에서는 $\text{det}(\boldsymbol A)$ 혹은 $\rvert \boldsymbol A \lvert$로 쓴다. $\boldsymbol A$의 **determinant**는 어떤 함수로,  $\boldsymbol A$를 어떤 실수로 mapping한다.

**Theorem 4.1.**  
어떠한 square matrix $\boldsymbol A \in \mathbb R^{n \times n}$든지 $\boldsymbol A$가 invertible하다는 뜻은 $\text{det}(\boldsymbol A) \neq 0$임과 동치이다.
{: .notice--info}

만일 어떤 square matrix $\boldsymbol T$가 $T _{i, j} = 0$ for $ i > j $이면 **upper-triangular matrix**라 부른다 (즉 대각행렬 아래로는 모두 0). 이 반대는 **lower-triangular matrix**라 부른다. triangular matrix $\boldsymbol T \in \mathbb R^{n \times n}$에 대해 행렬식은 대각성분의 곱과 같다.

$$
\text{det}(\boldsymbol T) = \prod^n _{i=1} T _{ii} \tag{4.8}
$$

또한 행렬식의 개념은 이를 $\mathbb R^n$의 어떤 객체를 spanning하는 n개의 벡터 집합으로부터의 mapping으로 생각하는게 자연스럽다. 곧 밝혀지겠지만 $\text{det}(\boldsymbol A)$는 행렬 $\boldsymbol A$의 columns가 형성하는 n차원의 평행육면체(parallelepiped)의 부호가 있는 부피 (signed volumn)이다.

![image](https://user-images.githubusercontent.com/47516855/115955691-14741880-a533-11eb-8933-2a2da5607850.png){: .align-center}{:width="300"}

만일 두 벡터가 이루는 각도가 작아진다면, 이에 따라 평행육면체 (이 경우 평행사변형)의 넓이는 줄어든다. 

$\boldsymbol A \in \mathbb R^{n \times n}$에 대해 행렬식은 다음의 성질을 만족한다.
- $\text{det}(\boldsymbol A \boldsymbol B) = \text{det}(\boldsymbol A) \text{det}(\boldsymbol B)$
- $\text{det}(\boldsymbol A) = \text{det}(\boldsymbol A ^\intercal)$
- \boldsymbol A가 regular(invertible)하면, $\text{det}(\boldsymbol A^{-1}) = \frac{1}{\text{det}(\boldsymbol A)}$
- 두 행렬이 닮음(similarity)이라면, 행렬식도 같다. 따라서, linear mapping
$\Phi: V \to V $에 대해 모든 transformation matrix $\boldsymbol A _{\Phi}$는 같은 행렬식을 갖는다. 그러므로 행렬식은 linear mapping의 basis에 invariant하다.
  - Recall: $\boldsymbol{\tilde A} = S^{-1}\boldsymbol{A}S$인 regular matrix $S \in \mathbb R^{n \times n}$가 존재하면, 두 matrix $\boldsymbol{A}, \boldsymbol{\tilde A} \in \mathbb R^{n \times n}$은 서로 **similar**하다고 한다.
- 행/열을 여러개 추가하여도 행렬식은 변하지 않는다.
- $\text{det}(\lambda \boldsymbol A) = \lambda^n \text{det}(\boldsymbol A)$
- 두 개의 행/열을 바꾸면 행렬식의 부호가 바뀐다.

마지막 3개의 성질로 인해, 가우스 소거법을 사용하여 행렬식을 구할 수 있다.

**Theorem 4.3.**  
정방행렬 $\boldsymbol A \in \mathbb R^{n \times n}$의 행렬식이 0인 것은 $\text{rk}(\boldsymbol A)=n$임과 동치이다. 즉, $\boldsymbol A$의 역행렬이 존재하는 것과 full rank임은 동치이다 (iff).
{: .notice--info}

$\text{det}(\boldsymbol A)$
\boldsymbol A
\mathbb R^{n \times n}
$\Phi: V \to V $
