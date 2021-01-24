---
title:  "머신러닝에서 사용하는 information theory 정리"
excerpt: "information theory의 기본에 대해 알아보자"
toc: true
toc_sticky: true
categories:
  - Machine Learning
tags:
  - Information Theory
use_math: true
last_modified_at: 2021-01-23
---

## Introduction

머신 러닝을 공부하다보면 정보이론에서 배우는 다양한 개념들이 등장한다. 이번 포스트에서는 머신 러닝에서 사용하는 정보이론의 개념을 살펴보도록 하자.

## Information

정보이론에서 어떤 event $X=x$의 정보(information)는 다음과 같이 정의된다.

$$
I(x) = -\log p(x)
$$

이러한 수식에는 어떠한 의미가 담겨있을까?

### Bit (binary digit)

동전던지기와 같이 확률이 $\frac{1}{2}$인 어떤 이벤트를 생각해보자. 앞선 수식에 따르면 이의 정보는 다음과 같다.

$$
I(x) = -\log _2 p(x) = -\log _2 \frac{1}{2}
$$

이처럼 확률이 $\frac{1}{2}$인 어떤 사건을 관측함으로서 얻는 정보의 양은 1 bit로 표현할 수 있다. 이는 즉, 두 개의 상태 (state)가 있을 때, 이를 구분하는 하나의 단위로 볼 수 있다.

### Nat (natural unit of information)

nat은 bit와 같이 획득하는 정보의 양인데, 다만 확률이 $\frac{1}{e}$인 경우이다.
$$
I(x) = -\log _e \frac{1}{e}
$$



그렇다면 왜 $\log \frac{1}{p(x)}$가 정보를 나타내기 위해 쓰였을까?

어떤 이산확률 변수 $x$에 대해, $x$이 정보 양은 **놀라움의 정도**로 표현이 가능하다. 매우 일어날 가능성이 높은 사건이 일어났을 때보다, 일어나기 힘든, 즉, **확률이 낮은 사건이 발생**했다는 접했을 때 **더 많은 정보**를 전달받게 되기 때문이다. 따라서, 정보량의 측정단위 $I(x)$는 확률분포 $p(x)$에 **종속**한다. 또한, 연관되지 않은 두 사건(event) $x$와 $y$가 함께 일어났을 때 얻는 정보량은 각자의 사건이 따로 일어났을 때 얻는 정보량의 합으로 생각할 수 있다. 따라서 $I(x, y) = I(x) + I(y)$가 되고, $x, y$는 서로 독립적이므로, 이의 joint probability $p(x, y)$는 $p(x)p(y)$가 된다. 이로부터 정보량 $I$와 확률 $p$의 관계는 **log로 표현할 수 있음**을 알게된다. 마이너스 기호는 정보량이 항상 양수만 갖게 하기 위해 붙여졌다.

![image](https://user-images.githubusercontent.com/47516855/105610744-83d35280-5df4-11eb-9c7e-7615fc4bbf46.png){: .align-center}{: width='500'}

위 그림은 이러한 성질을 잘 보여주고 있다.


## (Shannon) Entropy

Decision tree라던가, KL Divergence같은데서 흔히 사용하는 엔트로피는 무엇일까? 엔트로피는 앞서 살펴본 **정보의 평균**으로 나타낼 수 있다. 따라서 엔트로피 $H(x)$는 다음과 같이 표현할 수 있다.

$$
\begin{align}
H(x) & = \mathbb E _p[I(x)] \\
& = \mathbb E _p[-\log p(x)] \\
& = -\sum _x p(x) \log p(x) 
\end{align}
$$

본래의 정보이론에서 엔트로피는 $p(x)$에 대한 가능한 최적의 coding scheme을 사용한다고 가정했을 때, $X$로 부터 취한 값을 통신하기 위해 필요한 **bit의 기댓값**으로 생각할 수 있다.

엔트로피의 성질은 다음과 같다.
- 엔트로피를 이루는 성분은 모두 양수 (확률, log)이므로, $H(p) >= 0$
- 만일 $p(x)=1$라면, $H(p) = 0$
- 엔트로피는 확률이 모두 같은 경우에 최대화

### Example

말 8마리가 있고, 이들이 경주하여 이기는 확률변수 $x$는 동일한 확률값을 갖는다고 가정하자. 그렇다면 확률변수 $x$에 대한 정보와 엔트로피는 다음과 같다.

$$
I(x _i) = I(x _j) = - \log _2 p(x) = \log _2 8 = 3 \textrm { bits}
$$

$$
H(x) = -\sum _x p(x) \log p(x) = -8 \times \frac{1}{8} \log \frac{1}{8} = 3 \textrm { bits}
$$

각 말이 이길 확률이 균일하므로, 이의 평균값이 같은 것을 확인할 수 있다.

이번에는 다음과 같은 표처럼 각 확률변수가 서로 다른 확률값을 갖는 경우를 살펴보자.

| 확률변수| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
 :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|확률 | $\frac{1}{2}$ | $\frac{1}{4}$ | $\frac{1}{8}$ | $\frac{1}{16}$ | $\frac{1}{64}$ | $\frac{1}{64}$ | $\frac{1}{64}$ | $\frac{1}{64}$ |

이 경우 엔트로피는 다음과 같다.

$$
\begin{align}
H(x) &= -\sum _x p(x) \log p(x) \\
& = -\frac{1}{2} \log \frac{1}{2} -\frac{1}{4} \log \frac{1}{4} -\frac{1}{8} \log \frac{1}{8} -\frac{1}{16} \log \frac{1}{16} -\frac{4}{64} \log \frac{1}{64} = 2 \textrm { bits}
\end{align}
$$

이를 통해 uniform distribution의 엔트로피가 그렇지 않은 경우보다 더 크다는 사실을 확인할 수 있다.
이는 엔트로피를 **무질서의 척도**로 해석하는 것과 연관이 있다.
즉, 균일한 경우, 어떠한 관측을 하더라도 얻는 정보량이 작은 반면, 균일하지 않은 경우 낮은 확률의 사건을 관측하는 경우에는 얻는 **정보의 량이 증가**한다는 것이다.
이는 앞서 정의한 정보의 양과 같다.

직관적으로 생각해보면 어떤 균일 분포 $P(X)$에서 어떠한 관측을 통해 $P(X=x _i)=1/N$임을 밝혀냈다고 하자. 이를 통해 다음 시행에서 어떠한 결과가 나올지 얻을 수 있는 예측하기는 매우 힘들 것이다. 그러나 베르누이 분포 $p(x) = 0.99$라는 결과를 얻었다고 하자. 이는 $
1 - p = 0.01$임을 의미하고, 이를 통해 다음 시행을 예측하기 굉장히 쉬워진다. 즉, **불확실성(무질서)이 줄었다**고 볼 수 있다.

## Joint Entropy

**결합 엔트로피(Joint Entropy)**는 결합확률분포(joint distribution)을 사용하여 정의한 엔트로피를 말한다. 

$$
H(x, y) = - \sum _x \sum _y p(x)p(y)\log _2 p(x, y)
$$

## Conditional Entropy

또한, 조건부 확률 $p(y \rvert x)$에 대한 조건부 엔트로피(Conditional Entropy)는 다음과 같이 정의할 수 있다.

$$
\begin{align}
H(y|x) & = - \sum _x \sum _y p(x, y) \log \frac {1}{p(y \rvert x)} \\
& = - \sum _x \sum _y p(y \rvert x) p(x) \log \frac {1}{p(y \rvert x)}
\end{align}
$$

이는 확률변수 $X$가 가질 수 있는 모든 경우에 대해 $H(Y \rvert X = x)$를 가중평균 한 것이다.
조건부 엔트로피는 **$x$가 알려졌을 때 $y$에 대한 불확실성**으로 볼 수 있다.

## Cross Entropy

앞서 정보이론에서 엔트로피란 $p(x)$에 대한 가능한 **최적의 coding scheme을 사용한다고 가정**했을 때, $X$로 부터 취한 값을 통신하기 위해 필요한 bit의 기댓값이라고 했다.
만약 **최적의 coding scheme**을 사용하지 않는다면 어떻게 될까?

확률변수 $X$의 분포가 이전과는 다르게 $q(x)$라 하자. 
그러나 우리는 여전히 $p(x)$를 위한 최적의 coding scheme을 사용하고 있고, 이는 $q(x)$에서 최적이 아니다. 
즉, 우리의 모델은 **데이터가 $p(x)$를 따른다고 생각하는데, 실제로는 $q(x)$를 따르는 것이다.**
이 때, **교차엔트로피(cross entropy)**는 차선으로 최적화된 $p(x)$를 coding scheme으로 사용했을 때 확률변수 $X$로부터 취한 값과 통신하기 위해 필요한 bit의 기대값으로 볼 수 있다.

머신러닝의 목표는 우리가 알지 못하는 확률 분포를 모델링하는 것이다(density estimation). 이러한 관점에서 생각해볼 때, 교차 엔트로피는 true distribution $q(x)$에 대해 머신러닝 모델이 근사(approximation)한 distribution $p(x)$가 **얼마나 차이가 나는지**를 설명해준다.


두 확률분포  $p, q$에 대한 교차엔트로피 $H(p, q)$는 다음과 같이 정의한다.

$$
H(p, q) = - \sum _x p(x) \log _2 q(x)
$$

## Kullback-Leibler Divergence

쿨백-라이블러 발산 (Kullback-Leibler Divergence, KL Divergence)는 **두 확률분포 $p(x), q(x)$가 얼마나 다른지** 나타내는 지표이다.

정보이론의 관점으로 보면 차선으로 최적화된 $p(x)$를 coding scheme으로 사용했을 때 확률변수 $X$로부터 취한 값과 통신하기 위해 **추가적으로** 필요한 bit의 기댓값으로 볼 수 있다. 쿨백-라이블러 발산은 $D _{KL}$ 혹은 $KL$로, 다음과 같이 표현한다.

$$
\begin{align}
D _{KL} (p \rvert \rvert q) = KL(p \rvert \rvert q) & = H(p, q) - H(p)
= \sum _x p(x) log _2 \frac{p(x)}{q(x)}
\end{align}
$$

쿨백-라이블러 발산의 주요 특징은 다음과 같다:
- $D _{KL} (p \rvert \rvert q)$는 항상 양수
- $D _{KL} (p \rvert \rvert q) \neq D _{KL} (q \rvert \rvert p)$
- $D _{KL} (p \rvert \rvert q)=0$이면, $p$와 $q$는 같은 분포
- $p$가 높으나 $q$가 낮은 경우, 값이 커짐
- $p$가 낮으면 큰 신경을 쓰지 않음



### Relation Between KL Divergence and Entropy

쿨백-라이블러 발산은 또한 상대 엔트로피 (relative entropy)로도 부르는데, 이는 $p$에 대한 $q$의 *상대적인 엔트로피*로 볼 수 있기 때문이다. 
즉, **엔트로피를 KL 발산 관점에서** 생각해보면, 이는 우리의 분포 $p(x)$가 **uniform distribution으로부터 얼마만큼 떨어져있는지**에 대한 지표로 볼 수 있다. 즉, uniform에서 크게 벗어날 수록, 엔트로피는 증가한다 (uniform과의 KL 발산이 커진다).

그외에도 KL 발산은 다음과 같이 해석할 수 있다.
- **베이지안 관점에서** 이전의 확률분포 $q(x)$에서 사후 확률 분포 $p(x)$로 belief를 수정했을 때 얻을 수 있는 정보의 양
    - 달리 표현하면, $p$대신 $q$가 쓰였을 때 얻는 information gain
- **머신러닝 관점에서** true distribution $p$를 근사하기 위해 $q$를 사용했을 때 잃는 정보의 양


이를 딥러닝 관점에서 생각하면, true distribution은 $y$이고, 이에 대한 estimation은 $\hat y$가 된다. 따라서 KL 발산은 다음과 같은 식이 될 것이다.

$$
-(\sum y _j \log \hat y) - ( \sum y \log y )
$$

여기서 두번째 항은 데이터에 의해 정해지므로, 첫번째 항인 **우도(likelihood)**에 영향을 미치지 않는다. 즉, 모델의 파라미터와 무관하다. 따라서 **KL 발산을 최소화하는 것은 cross entropy와 negative log likeihood를 최소화하는 것과 같다.** 이는 딥러닝에서 교차 엔트로피가 손실함수(loss function)으로 쓰이는 주된 이유이다.

## Mutual Information

상호정보량(mutual information)은 결합분포(joint distribution)과 주변분포(marginal distribution) 사이의 상대 엔트로피로, 다음과 같이 계산된다.

$$
\begin{align}
I(x, y) & = \sum _x \sum _y p(x, y) \log [\frac{p(x, y)}{p(x)p(y)}] \\
& = D _{KL}(p(x, y) \rvert \rvert p(x)p(y)) \\
& = \mathbb E _{p(x, y)} [\log [\frac{p(x, y)}{p(x)p(y)}]]
\end{align}
$$

결합분포 $p(x, y)$는 $x$와 $y$가 독립이라면 $p(x)p(y)$가 된다. 따라서 두 분포가 독립하는 경우, 상호정보량의 값은 0이 된다.

상호정보량은 $y$에 대한 정보를 통해 줄일 수 있는 $x$의 불확실성을 의미하기도 한다.
즉, 다음과 같다.

$$
I(x, y) = H(x) - H(x \rvert y)
$$



## Summary
- 1 bit = the amount of information gained by observing an event of probability $\frac{1}{2}$
- 1 bit = distinct between two states
- 1 nat = the amount of information gained by observing an event of probability $\frac{1}{e}$
- Entropy = Average information