---
title:  "머신러닝을 위한 수학 정리: Probability and Distributions"
toc: true
toc_sticky: true
permalink: /project/mml/Probability-and-Distributions/
categories:
  - Mathmatics
  - Machine Learning
tags:
  - probability
  - distributions
use_math: true
last_modified_at: 2021-06-02
---

넓게보면 확률은 불확실성에 대한 연구라 할 수 있다. 확률은 어떤 사건이 일어나는 비율 혹은 사건에 대한 믿음(belief)의 정도라 할 수 있다. 그러면 이러한 확률을 이용하여 어떤 실험(experiment)에서 무엇인가가 일어날 기회를 측정할 수 있다. 앞선 챕터들에서 언급했듯, 데이터 내의 불확실성과 머신러닝 모델 내의 불확실성, 모델의 결과에 대한 불확실성을 측정할 때가 있다. 이를 측정하는 것은 **random variable(확률변수)**에 대한 개념을 필요로 한다. 확률변수란 임의의 실험에 대한 결과를 우리가 관심있는 성질들의 집합으로 연결하는 것이다. 이러한 확률변수와 연관있는 것은 특정한 결과가 일어날 확률을 측정하는 함수로, 우리는 이를 **확률 분포**라 부른다.

확률분포는 다른 개념을 위한 빌딩 블록으로, probabilistic modeling (Section 8.4), graphical models (Sec-tion 8.5), model selection (Section 8.6)에서 사용된다. 다음 챕터에서는 확률 공간에서 정의되는 세 개의 개념 (sample space, events, probability of an event)을 살펴볼 것이며, 이들이 확률변수와 어떻게 연관이 있는지 살펴볼 것이다. 여기서의 설명은 고의적으로 *대충 (hand wavy)* 설명하고 있는데, 이를 엄밀하게 설명하면 개념을 직관적으로 이해하는데 방해가 되기 때문이다. 이에 대한 전체적인 아웃라인은 아래 그림과 같다.

> An EXPERIMENT is any activity with an observable result. Tossing a coin, rolling a die or choosing a card are all considered experiments.

![image](https://user-images.githubusercontent.com/47516855/120493417-e74e3c00-c3f5-11eb-9503-32c2e7eae815.png){: .align-center}{:width="500"}

## Construction of a Probability Space

확률이론은 수학적 구조를 정의하여 실험을 통해 나오는 임의의 결과를 정의하기 위해있는 것이다. 예를들어, 한 개의 동전을 던졌을 때 그 결과를 결정할 수 없지만 엄청나게 많은 수의 동전을 던짐으로서 결과값을 평균적인 측면에서 관찰하여 규칙성을 관찰할 수 있다. 확률의 이러한 수학적인 구조를 이용하여 자동화된 추론(automated reasoning)을 수행하는 것을 목표로 하고, 이러한 측면에서 확률은 논리적인 추론(logical reasoning)을 일반화한다.

### Philosophical Issues

자동화된 추론 시스템을 구축할 때, 클래식한 Boolean logic(불 논리)로는 그럴듯한 추론(plausible reasoning)의 형태를 표현할 수 없다. 다음과 같은 시나리오를 고려해보자.

친구를 기다리고 있는데, 가능한 경우의 수는 세 가지이다.
- H1: 제 시각에 온다.
- H2: 막혀서 늦는다.
- H3: 외계인한테 납치당했다.

친구가 제 시간에 오지 않았을 때 H1은 논리적으로 기각한다. 또한, 우리는 H2가 가장 그럴듯할 것이라고 생각하는데, 논리적으로 이렇게 생각할 이유는 없다. H3가 일어날 것이라고 생각할 수도 있지만, 이럴 확률은 낮다고 치부해버린다.

여기서 우리는 어떻게 H2가 가장 그럴듯한 정답이라고 결론을 내릴 수 있었을까? 이러한 측면에서 확률이론은 불 논리의 일반화로 생각할 수 있다. 머신러닝에서는 자동화된 추론 시스템의 디자인을 형식화하는데 이러한 방법이 적용된다.

*Remarks*. 머신러닝과 통계에서 확률을 바라보는 두 개의 주요한 해석이 있다. 이는 베이지안과 빈도주의적 해석으로, 베이지안 해석은 확률을 어떤 사건에 대해 사람이 갖고 있는 불확실성의 정도를 측정하는 이용한다. 이는 가끔 "주관적 확률" 혹은 "믿음의 정도(degree of belief)"로 일컫기도 한다. 빈도주의적 해석은 사건이 발생하는 전체 횟수에 대해 관심있는 사건에 대한 상대적 빈도수로 생각할 수 있다. 사건의 확률은 무한히 많은 데이터가 있을 때 사건이 상대적으로 발생하는 정도로 정의한다.
{: .notice}

확률 모델을 다루는 머신러닝 책에서는 이상한 표현(lazy notation)과 전문용어를 통해 혼란을 주곤 한다. 이 책에도 예외는 없다. 서로 다른 많은 표현들을 통틀어 "확률 분포"로 호칭할 것이다. 독자들은 맥락만을 통해 이에 대해 구분해야 할텐데, 한가지 줄 수 있는 팁은 이산확률변수인지, 연속확률변수인지를 구분하여 이에 대해 구분하는 것이다.

### Probability and Random Variables

확률에 대해 이야기할 때 세 가지 개념들이 종종 혼동되곤 한다. 첫번째로는 probability space(확률공간)으로, 확률에 대한 아이디어를 정량화하게 해주는 개념이다. 그러나 우리는 이러한 확률공간을 직접적으로 다루는 대신 두번째 개념인 확률변수를 이용한다. 확률변수는 더욱 편리하고 (종종 수치적인) 공간으로 확률을 변환하게 해준다. 세번째 개념은 확률변수에 대한 규칙 혹은 분포이다. 세번째 개념은 다음챕터에서 살펴보도록 한다.

현대의 확률은 sample space, event space, probability measure 이 세 개의 개념을 소개한 Kolmogorov에서 제시하는 공리의 집합에 기반한다. 확률공간은 임의의 경우(outcome or sample)를 내는 현실세계의 프로세스를 모델링한다 (이는 실험이라고 칭하기도 한다).

- **The sample space(표본공간) $\Omega$**  
  - **The sample space(표본공간)**은 실험을 통해 얻을 수 있는 모든 가능한 경우(outcome)에 대한 집합으로, 주로 $\Omega$로 표현한다. 예를들어 두 개의 연속적인 동전 던지기를 통하여 얻는 확률공간은 ${\text{hh}, \text{tt}, \text{th}, \text{ht}}$가 된다.
- **The event space(사건공간) $\mathcal A$**  
  - **The event space(사건공간)**은 실험을 통해 얻을 수 있는 잠재적인 결과들에 대한 공간이다. 만약 실험의 결과로 관측하는 특정한 결과 $\omega \in \Omega$가 $A$(사건)에 있으면 (즉, 사건이 발생하면), 표본공간 $\Omega$의 부분집합 $A$은 event space $\mathcal A$내에 있다. Event space $\mathcal A$는 $\Omega$의 부분집합을 모음으로써 얻을 수 있다. Discrete probability distribution의 경우 $\mathcal A$는 종종 $\Omega$의 *멱집합(power set)*이 된다.

  > An EVENT is a subset of the sample space.  
  > 멱집합(power set)은 주어진 집합의 모든 부분 집합들로 구성된 집합이다.
- **The probability(확률)** $P$
  - 각 사건 $A \in \mathcal A$에 대해, $P(A)$라는 숫자를 통해 어떤 사건이 일어날 확률 혹은 믿음의 정도를 측정한다. $P(A)$는 $A$의 **확률**이라고 한다.

> 뭔 말인지 하나도 모르겠고, 정의가 너무 많이나와 정리한다.
> - Experiment, trial
>   - 관측 가능한 결과를 갖는 어떠한 행위. 동전던지기 등등
> - Outcome(경우) (or Sample point(표본점), sample(표본), 근원사건):
>   - 실험의 결과. 실험에 대한 가능한 모든 outcome의 집합은 Sample space라 불린다.
> - Event(사건, $\omega$)
>   - Sample space의 subset이다.
> 
> 주사위를 굴리는 실험에 대해,
> - Sample space: $S=\\{1, 2, 3, 4, 5, 6\\}$
> - Sample(\omega):  $1, 2, 3, 4, 5, 6$
> - Event space: $\Sigma = \text{Even}, \text{Odd}, \text{Prime}, \text{Divisible by 2} , \text{etc.}$
> - Event: $\varnothing, \\{1\\}, \\{2\\}, \cdots $

> ![](https://i.imgur.com/pkyG0xQ.png){: .align-center}{:width="600"}
>
> 확률변수는 확률에 따라 변하는 값으로, sample space를 domain으로, 실수를 range로 갖는 함수이다. $X$를 사용할 때 확률 $P(X)$를 구할 수 있다면, $X$는 확률변수라 할 수 있다.
>
> [출처: 확률 변수](https://adioshun.gitbooks.io/statics-with-r/content/probability/random-variance.html)

하나의 사건에 대한 확률은 반드시 구간 $[0, 1]$안에 있어야하고, sample space $\Omega$내에 모든 outcome에 대한 확률의 합은 반드시 1이 되야한다. 즉, $P(\Omega)=1$가 되야한다. 우리는 주어진 확률 공간 $(\Omega, \mathcal A, P)$에 대해, 이를 통해 현실세계의 현상을 모델링해야 한다. 머신러닝에서는 종종 명시적으로 확률 공간을 표현하는 대신 관심있는 것에 대한 확률을 측정하길 원한다. 이를 $\tau$라 표현하겠다. 이 책에서 $\tau$를 **target space**라 부를 것이고, $\tau$의 원소를 상태(states)라 부를 것이다.

우리는 어떤 함수 $X: \Omega \to \tau$를 도입할 것이다. 이 함수는 $\Omega$의 원소(outcome)를 취하여 우리의 관심사 $x$($\tau$내 원소)의 양(quantity)을 반환한다. 이러한 맵핑을 **random variable(확률변수)**라 부른다.

*random variable이라는 말은 큰 오해를 불러일으킨다. 이는 랜덤하지도 않고, 변수도 아니기 때문이다. 이는 그저 함수일 뿐이다.*

두 동전을 던져 앞면의 갯수를 세는 경우를 생각해보자. 확률변수 $X$는 세 가지 결과가 가능하다.
- $X(\text{hh})=2$
- $X(\text{ht})=1, X(\text{th})=1$
- $X(\text{tt})=0$

이 경우 $\tau =\\{0, 1, 2\\}$고, 이는 곧 $\tau$의 원소에 대한 확률로, 우리의 관심사가 된다. 유한한 차원의 sample space $\Omega$와 유한한 집합 $\tau$에 대해 확률변수에 해당하는 함수는 lookup table이 된다. 어떠한 subset $S \subseteq \tau$에 대해 확률 $P _X(S) \in [0, 1]$를 이에 해당하는 확률변수인 특정한 사건이 일어나는 것으로 연결시킨다.

> 본 책의 예제는 번역하지 않지만, 원할한 이해를 돕기 위해 Example 6.1을 번역하고 살펴보도록 하겠다.

<div class="notice--success" markdown="1">

**Example 6.1**

가방안에서 반복적으로 두 개의 동전을 뽑는 게임을 고려해보자. 이는 통계적 실험으로 생각할 수 있다. 가방안에는 미국 동전(\\$)과 영국동전(£)이 있다. 우리는 두번 뽑을 것이기 때문에 총 4개의 outcome이 나오게된다. 이의 state space 혹은 sample space $\Omega$는 그러면 (\\$, \\$), (\\$, £), (£, \\$), (£, £)가 된다. 또한, \\$를 뽑을 확률이 0.3으로 주어진다고 하자.

우리가 관심있는 사건은 \\$가 중복해서 나오는 횟수이다. 이제 확률변수 $X$를 정의해보자. 이는 sample space $\Omega$로부터 $\tau$로 맵핑하는 역할을 한다. $\tau$는 \\$를 가방으로부터 뽑았을 때 등장하는 횟수이다. 이 sample space로부터 우리는 $\tau=\\{0, 1, 2\\}$임을 알 수 있다. 확률변수(함수 혹은 lookup table) $X$는 다음과 같이 테이블 형태로 표현할 수 있다.

$$
\begin{align}
X((\$, \$))=2 \tag{6.1} \\ 
X((\$, £))=1 \tag{6.2} \\ 
X((£, \$))=1 \tag{6.3} \\ 
X((£, £))=0 \tag{6.4} \\ 
\end{align}
$$

두번째 동전을 뽑기 전에 첫번째 동전을 뽑기 때문에 두 뽑기 사이에는 독립적인 관계가 성립함을 알 수 있다. 이는 추후에 살펴보도록 하겠다. 여기서 두개의 outcome이 있고, 이는 같은 event로 맵핑함을 주목하자. 그렇기때문에 둘 중 하나만이 \\$를 내놓는 것으로도 볼 수 있다. 따라서 $X$의 probability mass function(확률질량함수)는 다음과 같이 주어진다.

$$
\begin{align}
P(X=2) &= P((\$, \$)) \\
      &=P(\$) \cdot P(\$) \\
      &= 0.3 \cdot 0.3 = 0.09 \tag{6.5} \\ 
P(X=1) &= P((\$, £) \cup P((£, \$)) \\
      &= P((\$, £)) + P((£, \$)) \\
      &= 0.3 \cdot (1-0.3) + (1-0.3) \cdot 0.3 = 0.42 \tag{6.6} \\ 
P(X=0) &= P((£, £)) \\
      &= P(£) + P(£) \\
      &= (1-0.3) \cdot (1-0.3) = 0.49 \tag{6.7} \\ 
\end{align}
$$

</div>

이 계산에서 우리는 $X$의 outcome에 대한 확률과, $\Omega$내 sample의 확률이라는 두 개의 다른 개념을 동일시하여 사용하였다. 예를들어, (6.7)에서 $P(X=0) = P((£, £))$로 표현하였다. 어떤 확률변수 $X: \Omega \to \tau$와 subset $S \subseteq \tau$를 생각해보자. $S$는 예를들어 두 개의 동전을 던졌을 때 얻을 수 있는 앞면의 갯수와 같은 것이다. $X^{-1}(S)$를 $X$에 의한 $S$의 pre-image(원상)이라고 해보자 (즉, $\Omega$의 원소의 집합이 $X$에 의해 $S$로 맵핑되는 것을 의미한다: $\\{\omega \in \Omega: X(\omega) \in S \\}$). 확률변수 $X$를 통해 $\Omega$내의 사건으로부터의 확률의 전이를 이해하는 방법 중 하나는 $S$의 pre-image로부터의 확률과 이를 연결시키는 것이다. $S \subseteq \tau$에 대해, 다음과 같은 notation을 얻는다.

$$
P _X(S) = P(X \in S) = P(X^{-1}(S)) = P(\{\omega \in \Omega: X(\omega) \in S \}) \tag{6.8}
$$

위 식의 좌변은 가능한 결과(e.g. # of \\$=1)의 집합에 대한 확률이고, 우리의 관심사가 된다. state로부터 outcome을 맵핑하는 확률변수 $X$를 통해 우변의 식이 state($\Omega$ 내)의 집합에 대한 확률이며, 이러한 state가 성질을 갖음을 볼 수 있다 (e.g., \\$£, £\\$). 우리는 확률변수는 특정한 확률분포 $P _X$에 따라 분포되어 있다고하며, 이는 event와 확률변수의 outcome의 확률 사이의 확률맵핑을 정의한다. 다른말로, 함수 $P _X$ 혹은 이와 동일한 $P \circ X^{-1}$은 확률변수 $X$의 **law** 혹은 **distribution**이라고 한다.

*Remark*. 확률변수 $X$의 range $\tau$인 target space는 확률공간(i.e., $\tau$ 확률변수)과 같은 것을 가르키는데 사용한다. $\tau$가 유한하거나 셀 수 있는 무한집합인 경우, 이는 이산확률변수라 부른다. 연속확률변수로는 $\tau = \mathbb R$ 혹은 $\tau = \mathbb R^D$인 경우이다.
{: .notice}

### Statistics

확률이론과 통계는 종종 함께 묶이는 개념이지만 불확실성의 다른 측면을 다룬다. 이 둘을 비교하는 방법 중 하나는 이들이 다루는 문제를 살펴보는 것이다. 확률을 이용하여 어떠한 프로세스의 모델을 다룰 수 있으며, 이의 불확실성은 확률변수에 의해 측정된다. 그리고 확률의 규칙을 이용하여 어떠한 일이 일어나는지를 살펴볼 수 있다. 통계에서 우리는 이미 일어나는 일을 다루며, 관측값을 설명할 수 있는 기저의 프로세스를 밝혀내려고 노력한다. 머신러닝의 목표가 데이터를 생성하는 프로세스를 적절하게 표현하는 모델을 생성한다는 측면에서 머신러닝은 통계와 가깝다고 할 수 있다. 또한, 확률의 규칙을 사용하여 어떠한 데이터에 대해 "best-fitting"하는 모델을 얻어낼 수도 있다.

머신러닝 시스템의 또다른 측면으로는 우리가 일반화 오류(Chapter 8)에 관심이 있다는 것이다. 이 뜻은 모델의 성능이 미래에 관측할 데이터에 관한 것으로, 이 데이터는 과거에 우리가 관측한 데이터와는 차이가 있다. 이러한 미래의 성능에 대한 분석은 확률과 통계에 의존하고 있으며, 대부분은 본 챕터에서 다루는 내용을 넘어선다. 이에 대해 흥미로움을 느끼는 독자들은 Boucheron et al. (2013)과 Shalev-Shwartz and Ben-David (2014)를 읽어보길 바란다. 통계에 대해서는 Chapter 8에서 추후에 더 살펴보도록 하겠다.

## Discrete and Continuous Probabilities

앞서 소개된 event의 확률을 묘사하는 방법에 대해 더욱 집중해보도록 하자. Target space가 연속적이냐 이산적이냐에 따라 분포를 다루는 방법이 달라지게 된다. 이산적일 경우 확률변수 $X$가 특정한 값 $x \in \tau$를 취하는 확률을 기술할 수 있다. 이는 $P(X=x)$로 표현한다. 이산확률분포 $X$에 대한 표현법 $P(X=x)$는 **probability mass function(확률질량함수)**라고 부른다.

반대로 연속적일 경우, 확률변수 $X$의 확률이 구간내에 있다고 기술하는 것이 자연스럽다. 이는 $a < b$에 대해 $P(a \leq X \leq b)$로 표현한다. 편의를 위해 확률변수 $X$의 확률이 특정한 값 $x$보다 작다고 기술하기도 하며, 이는 $P(X \leq x)$로 표현한다. 이 표현법은 **cumulative distribution function(누적분포함수)**라 알려져있다.

*Remark*. **univariate** distribution를 하나의 확률변수에 대한 분포로 사용할 것이다 (state는 non-bold $x$로 나타낸다). 하나 이상의 확률변수에 대한 분포는 **multivariate** 분포라 부를 것이고, 확률변수를 벡터로 사용할 것이다 (state가 bold $\boldsymbol x$)로 표현한다).
{: .notice}

### Discrete Probabilities

Target space가 이산일 경우, 여러개의 확률변수의 확률분포를 multidimensional array를 채우는 것으로 생각해볼 수 있다. 아래그림은 이에 대한 예시이다.

![image](https://user-images.githubusercontent.com/47516855/121380135-999a7c00-c97f-11eb-9de1-5384d21467e2.png){: .align-center}{:width="600"}

**joint probability(결합확률)**의 target space는 각 확률변수의 target space에 대한 카테시안 곱이 된다. 이는 다음과 같이 두 값을 동시에 넣는 것으로 정의한다.

$$
P(X=x _i, Y=y _j) = \frac{n _{ij}}{N} \tag{6.9}
$$

$n _{ij}$는 state $x _i, y _j$에 대한 event의 갯수이고, $N$은 전체 event의 갯수이다. 결합확률은 두 사건에 대한 교집합의 확률로, $P(X=x _i, Y = y _j)=P(X = x _i \cup Y = y _j)$가 된다. 앞서 본 그림 6.2는 확률질량함수 pmf를 묘사한 것으로, 두 개의 확률변수 $X, Y$에 대해 $X = x$와 $Y = y$일 때의 확률 값이다. 이는 $p(x, y)$로 쓰기도 한다. 또한 확률을 어떤 함수로, state $x, y$를 취하여 실수를 반환하는 것으로 생각해볼 수 있다. 이는 우리가 $p(x, y)$로 쓰는 이유가 되기도 한다. 확률변수 $X$가 $Y$ 값에 무관한 값 $x$를 취하는 **marginal distribution(주변확률)**는 $p(x)$로 쓴다. $X \sim p(x)$는 확률변수 $X$가 $p(x)$를 따라 분포함을 표현하기 위해 사용한다. 만일 $X=x$인 경우만 고려하고 나머지 확률분포를 고정하는 것을 **conditional probability(조건부확률)**이라 하고 $p(y \rvert x)$로 쓴다.

### Continuous Probabilities

본 장에서 우리는 real-valued random variable만 고려한다. 즉, target space가 실선 $\mathbb R$의 구간으로 주어지는 경우이다. 이 책에서 우리는 유한한 state를 갖는 이산확률공간이있는 것처럼, real random variable에 대해 연산을 수행 할 수 있다고 가정한다. 그러나 이런식으로 일반화하는 것은 무언가를 무한히 반복할 때와 구간에서 어떤 점을 뽑는 상황에서 정확해지지 않는다. 무언가를 무한히 뽑는 상황은 머신러닝에서 generalization error를 다룰 때 발생한다 (Chapter 8). 구간에서 점을 뽑는 것은 연속분포에서 발생한다. 그러나 앞서 언급한 것처럼 쉽게 가정하는 것은 확률에 대한 개념을 쉽게 소개하게 해준다.

*Remark* 연속적인 공간에서는 비직관적인 두 개의 추가적인 규칙이 있다. 우선, 모든 subset에 대한 집합(앞서 event space $\mathcal A$를 정의하는데 사용)은 충분히 잘 동작하지 않는다. $\mathcal A$는 여집합, 교집합, 합집합에 대해 잘 동작하도록 제한될 필요가 있다. 두번째로, 집합(이산공간에서는 원소를 세는 것으로 얻어짐)의 크기가 tricky하게 바뀐다는 것이다. 집합의 크기는 이의 **measure(측도)**라고 불린다. 예를들어, 이산집합의 cardinality라던가, $\mathbb R$의 구간의 길이, $\mathbb R^d$의 영역의 volume$은 모두 measure가 된다. 집합에 대한 연산에 대해 잘 동작하고, topology를 갖고 있는 경우엔 **Borel $\sigma$-algebra**라고 부른다.
{: .notice}

<div class="notice--warning" markdown="1">

**Definition 6.1** (Probability Density Function). 다음을 만족하는 어떤 함수 $f: \mathbb R^D \to \mathbb R$를 **probability density function(확률밀도함수) (pdf)**라 한다.

1. $\forall \boldsymbol x \in \mathbb R^D: f(\boldsymbol x) \geq 0$
2. 이의 적분이 존재하고,

$$
\int _{\mathbb R^D} f(\boldsymbol x)d \boldsymbol x = 1 \tag{6.15}
$$

</div>

이산확률변수에 대한 probability mass function (pmf, 확률질량함수)을 다룰 때 식 (6.15)는 summation으로 바뀌게된다.

pdf는 어떠한 함수라도 non-negative하며 적분의 결과가 1이되면 된다. 확률변수 $X$를 이 함수 $f$를 다음과 같이 연결시킬 수 있다.

$$
P(a \leq X \leq b) = \int^b _a f(x)dx \tag{6.16}
$$

$a, b \in \mathbb R$이고 $x \in \mathbb R$은 연속확률변수 $X$의 outcome이다. State $\boldsymbol x \in \mathbb R^D$는 $x \in \mathbb R$의 vector와 비슷하게 정의된다. (6.16)은 확률분포 $X$의 **law** 혹은 **distribution(분포)**라고 한다.

*Remark.* 이산확률분포와 대조적으로 특정한 값 $P(X=x)$를 취하는 연속확률변수 $X$의 확률은 0이다. 이는 (6.16)의 구간을 $a=b$로 설정하는 것과 같다.

<div class="notice--warning" markdown="1">

**Definition 6.2** (Cumulative Distribution Function). State $\boldsymbol x \in ]\mathbb R^D$를 갖는 multivariate real-valued random variable $X$의 **Cumulatvie distribution function (cdf)**는 다음과 같이 주어진다.

$$
F _X(\boldsymbol x) = P(X _1 \leq x _1, \cdots, X _D \leq x _D) \tag{6.17}
$$

여기서 $X=[X _1, \cdots, X _D]^\intercal, \boldsymbol x = [x _1, \cdots, x _D]^\intercal$이 되고, 우변은 확률변수 $X _i$가 $x _i$이하의 값을 취할 확률을 나타낸다.


</div>

또한 cdf는 pdf $f(\boldsymbol x)$의 integral로 표현할 수 있다.

$$
F _X(\boldsymbol x) = \int^{x _1} _{-\infty} \cdots \int^{x _D} _{-\infty} f(z _1, \cdots, z _D)\text{d}z _1 \cdots \text{d}z _D \tag{6.18}
$$

*대응하는 pdf가 없는 cdf도 있다*

*Remark.* 사실은 두 개의 다른 분포가 있다는 것을 되풀이하겠다. 첫번째는 pdf($f(x)$)에 대한 것으로, nonnegative function이며 이의 합은 1이 된다. 두번째는 확률변수 $X$의 law로, 이는 확률변수 $X$와 pdf $f(x)$의 결합이다.
{: .notice}

### Contrasting Discrete and Continuous Distributions

앞서 [Probability and Random Variables](#probability-and-random-variables)에서 확률은 양수이며, 전체 확률의 합은 1이라고 하였다. 이는 이산확률변수에 대해서는 각 state의 확률이 반드시 구간 [0, 1] 사이에 있어야 한다는 것을 뜻한다. 그러나 연속확률변수에 대해서는 normalization은 (식 6.15 참고) 밀도의 값이 1이하의 값만을 가져야한다는 것을 의미하진 않는다. 아래 그림은 균등분포를 통해 둘 모두를 묘사하고 있다.

![image](https://user-images.githubusercontent.com/47516855/121543540-f52e3d80-ca43-11eb-8576-476afe46a807.png){: .align-center}{:width="500"}

*Remark*. 추가적으로 이산확률분포에 관련된 교묘함이 숨겨져있다. State $z _1, \cdots, z _d$는 원칙적으로 어떠한 구조도 갖지 않는다. 즉, $z _1=\text{red}, z _2=\text{green}, z _3=\text{blue}$와 같이 비교할 수 있는 방법이 없다는 것이다. 그러나 머신러닝에서 이산적인 state는 숫자값을 취하게 된다 (e.g. $z _1=-1.1, z _2=0.3, z _3=1.5$). 그러면 우리는 이들 사이에 순서를 비교할 수 있게된다. 숫자값을 가정하는 Discrete states는 특히 유용한데, 이는 확률변수의 기댓값을 고려할 수 있게되기 때문이다.
{: .notice}

불행하게도 머신러닝 문헌들은 sample space $\Omega$, target space $\tau$, 확률변수 $X$를 구분하는 notation이나 명명법을 사용하지 않는다. 확률변수 $X$에 대해 가능한 모든 outcome의 집합의 원소 $x$에 대해(즉, $\x \in \tau$), $p(x)$는 확률변수 $X$가 outcome $x$를 갖을 확률을 표현한 것이다. 이산확률변수에 대해서 이는 $P(X=x)$로 쓰고, 이는 pmf로도 알려져있다. pmf는 종종 "분포"로 일컫어진다. 연속확률변수에 대해서, $p(x)$는 pdf라 불린다(종종 density로도 불림). 여기서 더 헷갈리게도, cdf $P(X \leq x)$는 종종 "분포"라 불린다.

본 챕터에서 우리는 notation $X$를 univariate/multivariate random variable을 가르키는데 사용할 것이고, state를 $x, \boldsymbol x$로 쓸 것이다. 이러한 명명법은 아래의 테이블에 요약해놓았다.

![image](https://user-images.githubusercontent.com/47516855/121548388-eea1c500-ca47-11eb-9b5f-7b9bffed13ab.png){: .align-center}{:width="600"}

*Remark.* 비록 틀린표현일지라도 "확률분포"라는 표현을 pmf뿐만이 아니라 연속적인 pdf를 나타내는데도 사용할 것이다. 대부분의 머신러닝 문헌이 그렇듯 이를 구분하는 것은 맥락에 맡길 것이다.
{: .notice}

## Sum Rule, Product Rule, and Bayes’ Theorem

확률이론은 logical reasoning의 연장선으로 생각할 수 있다. 확률모형은(Section 8.4) ML methods를 설계하는데 있어 원칙적 기반을 제공한다. 한번 우리가 데이터에 대한 불확실성에 대응하는 확률분포와 우리의 문제를 정의하고 나면, 곧 이에는 오직 두 가지 기본적인 규칙밖에 없다는 것이 밝혀지게 된다. 이는 sum rule과 product rule이다.

식 (6.9)를 다시 살펴보도록 하자. $p(\boldsymbol x, \boldsymbol y)$는 두 확률변수의 joint distribution이다. 분포 $p(\boldsymbol x)$와 $p(\boldsymbol y)$는 이에 해당하는 marginal distribution이 되고, $p(\boldsymbol y \rvert \boldsymbol x)$는 conditional distribution이 된다. 앞서 살펴봤던 연속/이산확률변수의 주변/조건부 확률의 정의로부터 확률이론에서 기본적인 규칙이 되는 두 가지 규칙을 보일 수 있게된다.

$$
p(\boldsymbol x) = 
\begin{cases}
    \sum _{\boldsymbol y \in \mathcal Y} p(\boldsymbol x, \boldsymbol y)       & \quad \text{if } \boldsymbol y \text{ is discrete}\\
    \int _{\mathcal Y} p(\boldsymbol x, \boldsymbol y) \text{d}\boldsymbol y  & \quad \text{if } \boldsymbol y \text{ is continuous}
\end{cases}
\tag{6.20}
$$

$\mathcal Y$는 확률변수 $Y$의 target space에 대한 state이다. 이 뜻은 우리가 확률변수 $Y$의 state $\boldsymbol y$의 집합을 모두 더한다는 뜻이다. Sum rule은 또한 **marginalization property**로 알려져 있다. Sum rule은 joint distribution을 marginal distribution으로 연결하는 역할을 한다. 일반적으로 joint distribution이 두 개 이상의 확률변수를 포함하고 있을 경우, sum rule은 확률변수의 어떠한 부분집합에도 적용할 수 있으며, 이 결과로 잠재적으로 하나 이상의 확률변수에 대한 marginal distribution을 얻게 된다. 좀 더 구체적으로, $\boldsymbol x = [x _1, \cdots, x _D]^\intercal$일 경우, sum rule을 반복하여 다음과 같은 marginal을 얻는다.

$$
p(x _i) = \int p(x _1, \cdots, x _D) \text{d}\boldsymbol x _{\setminus i} \tag{6.21}
$$

이때 sum/integrate는 모든 $x _i$를 제외한 모든 확률변수에 대해 진행한다. 식의 $\setminus i$가 바로 이를 나타내고, 이는 "$i$를 제외한 모든"으로 해석하면 된다.

두번째 규칙은 **product rule**로, joint distribution과 conditional distribution을 연결하는 것이다.

$$
p(\boldsymbol x, \boldsymbol y) = p(\boldsymbol y \rvert \boldsymbol x)p(\boldsymbol x) \tag{6.22}
$$

Product rule은 "모든 두 개의 확률변수로 이루어진 모든 joint distribution은 서로 다른 두개의 분포로 factorize가 가능하다"로 해석할 수 있다. 두 factor는 첫번째 확률변수 $p(\boldsymbol x)$의 marginal distribution과, 첫번째 확률변수가 주어졌을 때의 두번째 확률변수의 conditional distribution $p(\boldsymbol y \rvert \boldsymbol x)$이다. 여기서 확률변수의 순서는 임의로 주어졌기 때문에, product rule은 반대로도 작용하므로 $pp(\boldsymbol x, \boldsymbol y) = p(\boldsymbol x \rvert \boldsymbol y)p(\boldsymbol y)$ 역시 성립함을 알 수 있다. 연속확률변수의 경우, product rule은 pdf의 형태로 표현되게 된다.

머신러닝과 베이지안 통계학에서 관측된 확률변수가 주어질 때 (즉, 데이터), 종종 관측되지 않은(latent) 확률변수를 추론해야할 때가 있다. 우리에게 관측되지 않은 확률변수 $\boldsymbol x$에 대한 어떤 사전지식(prior knowledge) $p(\boldsymbol x)$과 확률변수 $\boldsymbol x$와 $\boldsymbol y$사이에서 관측된 어떤 관계 $p(\boldsymbol y \rvert \boldsymbol x)$가 있다고 가정하자. 만일 우리가 $\boldsymbol y$를 관측하면, 베이즈 정리를 이용하여 관측값 $\boldsymbol y$가 주어졌을 때에  $\boldsymbol x$에 대한 결론을 얻을 수 있다. **Bayes' theorem(베이즈 이론)**은 (6.22)로부터 다음과 같은 식을 바로 도출할 수 있다.

$$
\overbrace{p(\boldsymbol x, \boldsymbol y)}^{\text{posterior}} = \frac{\overbrace{p(\boldsymbol y \rvert \boldsymbol x)}^{\text{likelihood}} \overbrace{p(\boldsymbol x)}^{\text{prior}}}{\underbrace{p(\boldsymbol y)} _{\text{evidence}}} \tag{6.23}
$$

(6.23)에서 $p(\boldsymbol x)$는 **prior(사전확률)**로, 어떠한 데이터를 관측하기도 전에 latent variable $\boldsymbol x$에 대한 우리의 주관적인 사전지식을 내포한다. 우리는 어떠한 그럴듯한 prior라도 선택할 수 있지만, 가능한 모든 $\boldsymbol x$에 대해 prior가 nonzero pdf(or pmf)를 갖게해야 한다.

**Likelihood(우도)** $p(\boldsymbol y \rvert \boldsymbol x)$는 $\boldsymbol x$와 $\boldsymbol y$가 얼마나 연관되어있는지를 나타내고, 이산확률분포의 경우 잠재 변수 $\boldsymbol x$를 알고 있을 때 data $\boldsymbol y$의 확률이 된다. 여기서 한가지 집고 넘어갈 것은 우도는 $\boldsymbol x$에 대한 분포가 아니라 오직 $\boldsymbol y$의 영향을 받는 분포라는 것이다. $p(\boldsymbol y \rvert \boldsymbol x)$는 "($\boldsymbol y$가 주어졌을 때) $\boldsymbol x$의 우도"로 부르거나, $\boldsymbol x$가 주어졌을 때 $\boldsymbol y$의 확률이라 부르지만, 절대로 $\boldsymbol y$의 우도라고 부르지는 않는다.

*우도는 또한 종종 "measurement model"이라고 불리기도 한다.*

**Posterior(사후확률)** $p(\boldsymbol x \rvert \boldsymbol y)$는 베이즈 통계학에서 *quantity of interest*라 불리는 것인데, 이는 정확히 우리가 관심있는 것을 표현한 것이기 때문이다. 이는 곧 $\boldsymbol y$를 관측하고나서 $\boldsymbol x$에 대해 알게되는 것이 된다.

다음과 같은 quantity는 **marginal likelihood/evidence**라 불린다.

$$
p(\boldsymbol y) := \int p(\boldsymbol y \rvert \boldsymbol x)p(\boldsymbol x) \text{d}\boldsymbol x = \mathbb E _X [p(\boldsymbol y \rvert \boldsymbol x)] \tag{6.27}
$$

우변은 다음장에서 정의할 기댓값을 사용한 것이다. 정의에 의해, marginal likelihood는 (6.23)의 분자를 잠재변수 $\boldsymbol x$에 대해 적분한다. 그러므로 marginal likelihood는 $\boldsymbol x$에 대해 독립적이고, posterior $p(\boldsymbol x \rvert \boldsymbol y)$가 normalize된다. 또한, 이는 prior $p(\boldsymbol x)$에 대한 expected likelihood로 해석할 수 있다. Posterior의 정규화를 넘어서 이는 추후 Section 8.6에서 배울 베이지안 모델 선택에서 중요한 역할을 한다. 여기에서는 적분 연산을 하게 되는데, 이로 인해 evidence가 계산되기 어려워진다.

<div class="notice" markdown="1">

*Remark.* 베이즈 통계학에서 사후분포는 quantity of interest로, 데이터와 사전확률로부터 가능한 모든 정보를 담고 있다. 사후확률을 따르는 대신, 사후확률의 최대값과 같은 통계량에 관심을 갖는 것도 가능하다. 그러나 이는 정보의 손실을 야기한다. 더 큰 맥락을 생각해보면, 사후확률은 의사결정 시스템에서 사용할 수 있으며, 완전 사후확률을 갖는 것은 매우 유용하며 robust한 결정을 내릴 수 있게된다.

예를 들어 강화학습에서 Deisenroth et al. 2015는 plausible transition functions의 완전 사후확률분포가 매우 빠른 학습(data/sample efficient)을 가능케 했으며, 사후확률의 최댓값을 이용하는 방법은 일관되게 안 좋은 결론을 도출하였다. 따라서 완전 사후확률분포는 downstream task에서 매우 유용하다고 할 수 있다. 이는 Chapter 9의 linear regression의 맥락에서 다시 살펴보도록 하겠다.

</div>

## Summary Statistics and Independence

우리는 종종 확률변수를 요약하거나 확률변수의 pairs를 비교하는 것에 관심을 갖게된다. 확률변수의 statistic(통계량)은 확률변수의 deterministic function이고, 분포의 summary statistics(요약 통계량)은 확률변수가 어떻게 행동하는지에 대해 유용한 관점을 제공하고, 이름이 나타내는 것처럼 분포의 특징과 요약을 제공한다. 우리는 평균과 분산이라는 잘 알려진 두 개의 요약 통계량에 대해 살펴볼 것이다. 그다음 두 확률변수를 비교하는 방법으로 독립인지 아닌지 판별하는 방법과, 이들 사이의 inner product를 계산하는 것이다.

### Means and Covariances

평균과 (공)분산은 확률분포의 성질을 기술하는데 유용하게 사용된다(기댓값(expected value)과 퍼짐의 정도). 추후에 Section 6.6에서 살펴보겠지만, exponential family(지수족)는 확률변수의 통계량이 가능한 모든 정보를 포착한다.

기댓값의 개념은 머신러닝의 핵심이며, 확률 그 자체의 핵심적인 개념 또한 기댓값을 통해서 유도할 수 있다.

<div class="notice--warning" markdown="1">

**Definition 6.3** (Expected Value). 단변량 연속 확률변수 $X \sim p(x)$에 대한 어떤 함수 $g: \mathbb R \to \mathbb R$의 **Expected value(기댓값)**은 다음과 같이 주어진다.

$$
\mathbb E _X[g(x)] = \int _\mathcal X g(x)p(x)\text{d}x \tag{6.28}
$$

마찬가지로 이산 확률변수 $X \sim p(x)$에 대한 함수 $g$의 기댓값은 다음으로 주어진다.

$$
\mathbb E _X[g(x)] = \sum _{x \in \mathcal X} g(x)p(x) \tag{6.29}
$$

$\mathcal X$는 확률변수 $X$의 가능한 모든 outcome의 집합(target space)이 된다.

</div>


이 장에서 이산 확률변수는 숫자로 이루어진 outcome으로 간주한다. 이는 함수 $g$가 실수를 input으로 취하는 것을 관측하여 알 수 있다. 

<div class="notice" markdown="1">

*Remark*. 우리는 다변량 확률변수 $X$를 단변량 확률변수의 유한 벡터 $[X _1, \cdots, X _D]^\intercal$로 생각할 수 있다. 다변량 확률변수에 대해 element-wise expected value는 다음과 같이 정의한다.

$$
\mathbb E _X[g(\boldsymbol x)] = 
\begin{bmatrix}
  \mathbb E _{X _1}[g(x _1)] \\
  vdots \\
  \mathbb E _{X _D}[g(x _D)]
\end{bmatrix} \in \mathbb R^D \tag{6.30}
$$

여기서 아랫첨자 $\mathbb E _{X _d}$는 벡터 $\boldsymbol x$ $d$번째 원소에 대한 기댓값을 취한 것을 나타낸다.

</div>

Definition 6.3은 notation $\mathbb E _X$를 pdf(연속분포)에 대한 적분 혹은 모든 state에 대한 합(이산분포)을 하는 연산으로 정의하고 있다. Mean에 대한 정의(Definition 6.4)는 기댓값에 대한 특수한 경우로, $g$가 identity function인 경우에 얻어진다.

<div class="notice" markdown="1">

**Definition 6.4** (Mean). State $\boldsymbol x \in \mathbb R^D$에 대한 확률변수 $X$의 **mean**은 average가 되고, 다음과 같이 정의한다.

$$
\mathbb E _X[\boldsymbol x] = 
\begin{bmatrix}
  \mathbb E _{X _1}[x _1] \\
  \vdots \\
  \mathbb E _{X _D}[x _D]
\end{bmatrix} \in \mathbb R^D \tag{6.31}
$$

여기서 $\mathbb E _{X _d}[x _d]$는 $d=1, \cdots, D$에 대해 다음과 같이 정의된다.

$$
\mathbb E _{X _d}[x _d] := 
\begin{cases}
    \int _{\mathcal X} x _d, p(x _d) \text{d}x _d  & \quad \text{if } X \text{ is a continuous random variable} \\
    \sum _{x _i \in \mathcal X} x _i p(x _d = x _i)       & \quad \text{if } X \text{ is discrete random variable}
\end{cases} \tag{6.32}
$$

아랫첨자 $d$는 $\boldsymbol x$의 차원을 의미한다. 적분과 합은 확률변수 $X$의 target space 내 state $\mathcal X$에 대해 시행한다.

</div>

차원이 하나인 경우, "average"에 대한 직관적인 표현법은 mean 외에도 **median**과 **mode**가 있다. Median은 값들을 정렬했을 때 "중앙"에 있는 값이다. 이러한 개념은 연속적인 값일 때 cdf(Definition 6.2)에서는 0.5가 된다. asymmetric하거나 긴 꼬리가 달린 분포의 경우 median은 mean값보다는 사람의 직관에 가까운 값을 제공해준다. 또한, 이상치에 대해 덜 민감하다. 더 높은 차원에서 median을 측정하는것은 값을 정렬한다는 개념이 없기 때문에 자명하지가 않다(non-trivial). 

Mode는 가장 빈번하게 등장하는 값으로, 이산확률변수에서는 등장횟수로, 연속확률변수에서는 density $p(\boldsymbol x)$의 값이 튀는(peak) 것으로 정의된다. 특정한 density $p(\boldsymbol x)$는 하나 이상의 mode값을 갖을 수 있으며, 또한 차원이 높을 경우 매우 큰 mode를 갖을 수 있다. 따라서 어떤 분포의 모든 mode를 찾는 것은 계산하기 어려울 수 있다.

<div class="notice" markdown="1">

*Remark.* 기댓값(Definition 6.3)은 linear operator이다. 예를 들어 real-valued function $f(\boldsymbol x) = ag(\boldsymbol x) + bh(\boldsymbol x) ~\text {where}~a,b \in \mathbb R ~\text{and}~ \boldsymbol x \in \mathbb R^D$에 대해, 우리는 다음과 같은 기댓값을 얻는다.

$$
\begin{align}
\mathbb E _{X}[f(\boldsymbol x)] &= \int f(\boldsymbol x)p(\boldsymbol x)\text{d} \boldsymbol x \tag{6.34a} \\
&= \int [a g(\boldsymbol x) + b h(\boldsymbol x)] p(\boldsymbol x) \text{d} \boldsymbol x \tag{6.34b} \\
&= a \int g(\boldsymbol x) p(\boldsymbol x) \text{d} \boldsymbol x + b \int h(\boldsymbol x) p(\boldsymbol x)\text{d} \boldsymbol x \tag{6.34c} \\
& = a \mathbb E _{X}[g(\boldsymbol x)] + b \mathbb E _{X}[h(\boldsymbol x)] \tag{6.34d}
\end{align}
$$

</div>

두개의 확률 변수간에 의존성을 파악하고 싶은 경우가 있다. Covariance는 확률변수들이 다른 확률변수와 얼마나 의존되었는지에 대한 직관적인 개념을 제공한다.

<div class="notice--warning" markdown="1">

**Definition  6.5** (Covariance  (Univariate)) 두 단변량 확률변수 $X, Y \in \mathbb R$에 대한 **Covariance**는 이들 각각의 평균과 개별값의 차의 곱의 평균과 같다. 즉,

$$
\text{Cov} _{X, Y}[x, y] := \mathbb E _{X, Y}[(x - \mathbb E _X [x])(y - \mathbb E _Y[y])] \tag{6.35}
$$

</div>

*Remark.* 공변량과 기댓값을 표현할 때 이를 표현하는 확률변수가 명확하다면 아랫첨자는 생략하는 경향이 있다. 즉, $\mathbb E _X [x] = \mathbb E[x]$가 된다.
{: .notice}

기댓값의 linearity 성질을 이용하여, 앞서 살펴보면 Definition 6.5는 다음과 같이 바꿔 쓸 수 있다.

$$
\text{Cov}[x, y] = \mathbb E[xy] - \mathbb E[x] \mathbb E[y] \tag{6.36}
$$

자기자신에 대한 공분산 $\text{Cov}[x, x]$는 **variance**라 부르고, $\mathbb V _X [x]$로 쓴다. 분산의 제곱근은 **standard deviation**이라 부르고, $\sigma(x)$로 쓴다. 공분산은 다변량 확률변수에서도 적용할 수 있다.

<div class="notice--warning" markdown="1">

**Definition  6.6** (Covariance (Multivariate)) State $\boldsymbol x \in \mathbb R^D, \boldsymbol y \in \mathbb R^E$를 갖는 두 다변량 확률변수 $X, Y \in \mathbb R$에 대한 **Covariance**는 다음과 같이 정의된다.

$$
\text{Cov}[\boldsymbol x, \boldsymbol y] := \mathbb E _[\boldsymbol x \boldsymbol y^\intercal] - \mathbb E _[\boldsymbol x]\mathbb E _[\boldsymbol y]^\intercal = \text{Cov}[\boldsymbol y, \boldsymbol x]^\intercal \in \mathbb R^{D \times E}
\tag{6.37}
$$

</div>

Definition  6.6은 같은 확률변수에 대해서도 적용할 수 있으며, 이는 직관적으로 확률변수의 "퍼짐"을 나타내는 유용한 개념을 나타낼 수 있다. 다변량 확률변수에 대해, 분산은 확률변수의 개개의 차원간의 관계를 나타낸다.

<div class="notice--warning" markdown="1">

**Definition  6.7** (variance) State $\boldsymbol x \in \mathbb R^D$와 mean vector $\boldsymbol \mu \in \mathbb R^D$를 갖는 다변량 확률변수 $X$에 대한 **variance**는 다음과 같이 정의된다.

$$
\begin{align}
\mathbb V _X[\boldsymbol x] &= \text{Cov} _X[\boldsymbol x, \boldsymbol x] \tag{6.38a} \\
&= \mathbb E _X [(\boldsymbol x - \boldsymbol \mu)(\boldsymbol x - \boldsymbol \mu)^\intercal] = \mathbb E _X[\boldsymbol x \boldsymbol x^\intercal] - \mathbb E _X [\boldsymbol x] \mathbb E _X[\boldsymbol x]\mathbb E _X[\boldsymbol x]^\intercal \tag{6.38b} \\
&= 
\begin{bmatrix}
  \text{Cov}[x _1, x _1] & \text{Cov}[x _1, x _2] & \cdots & \text{Cov}[x _1, x _D] \\
  \text{Cov}[x _2, x _1] & \text{Cov}[x _2, x _2] & \cdots & \text{Cov}[x _2, x _D] \\
  \vdots & \vdots & \ddots & \vdots \\
  \text{Cov}[x _D, x _1] & \cdots & \cdots & \text{Cov}[x _D, x _D] \\
\end{bmatrix} \tag{6.38c}
\end{align}
$$

</div>

(6.38c)의 $D \times D$ 행렬은 다변량 확률변수 $X$의 **covariance matrix**라 부른다. 공분산 행렬은 symmetric, positive, semidifinite하고, 데이터의 퍼짐의 정도를 말해준다. 대각성분에 대해서는 marginal에 대한 variance를 포함한다.

$$
p(x _i) = \int p(x _1, \cdots, x _D)\text{d}x _{\subset i} \tag{6.39}
$$

비대각성분은 **cross-covariance** 성분으로, $\text{Cov}[x _i, x _j] ~ \text{for} ~ i, j = 1, ..., D, i \neq j$로 나타낸다.

<div class="notice--warning" markdown="1">

**Definition  6.8** (Correlation) 두 개의 확률변수 $X, Y$ 사이의 **correlation**은 다음과 같이 주어진다.

$$
\text{corr}[x, y] = \frac{\text{Cov}[x, y]}{\sqrt{\mathbb V[x] \mathbb V[y]}} \in [-1, 1] \tag{6.40}
$$

</div>

Correlation matrix는 standardized random variable $x/\sigma(x)$의 covariance matrix이다. 다른말로하면, 각 확률변수가 correlation matrix안의 각각의 표준편차로 나누어진 것이다.

Covariance/correlation은 두 개의 확률변수가 어떻게 연관되었는지를 나타낸다.

![image](https://user-images.githubusercontent.com/47516855/122921463-7b7c4500-d39d-11eb-9f5a-df904133bb1c.png){: .align-center}{:width="800"}


### Empirical Means and Covariances

앞장에서 본 정의들은 종종 **population mean and covariance**라 불린다. 이는 모집단에 대한 실제 통계량를 가르키기 때문이다. 머신러닝에서는 데이터의 경험적인 관측으로부터 배워야 한다. 어떤 확률변수 $X$를 고려해보자. 모집단의 통계량으로부터 경험적 통계량의 실현값(realization)으로 가기 위해서는 두 가지 개념적인 단계가 있다. 우선, 유한한 데이터셋($N$ 사이즈)가 있어서 유한한 갯수의 동일한 확률변수 $X _1, \cdots, X _N$에 대한 함수인 경험적인 통계량을 만드는 것이다. 두번째로 데이터를 관측하고, 각 확률변수에 대한 실현 $x _1, \cdots, x _N$을 보고 경험적 통계량을 적용하는 것이다.

구체적으로 mean(Definition 6.4)에 대해, 특정한 데이터셋이 주어졌을 때 mean에 대한 추정치(estimate)를 얻을 수 있고, 이는 **empirical mean** 혹은 **sample mean**이라 부른다. 이는 covariance에 대해서도 똑같이 적용된다.

> 현실 세계의 데이터는 확률변수가 가진 확률분포에 따라 실수 표본공간에서 선택된 표본이다. 이렇게 확률분포함수에 따라 표본공간의 표본이 현실 세계의 데이터로 선택되는 것을 **실현(realization)** 혹은 **표본화(sampling)**라고 한다.
>
> 출처: [데이터 사이언스 스쿨 - 7.1 확률적 데이터와 확률변수](https://datascienceschool.net/02%20mathematics/07.01%20%ED%99%95%EB%A5%A0%EC%A0%81%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%99%80%20%ED%99%95%EB%A5%A0%EB%B3%80%EC%88%98.html)

<div class="notice--warning" markdown="1">

**Definition 6.9** (Empirical Mean(표본평균) and Covariance(표본분산)). **Empirical mean** vector는 각 확률변수에 대한 관측치의 산술평균으로 이루어지고, 다음과 같이 정의된다.

$$
\bar{\boldsymbol {x}} := \frac{1}{N} \sum^N _{n=1} \boldsymbol x _n \tag{6.41}
$$

여기서 $\boldsymbol x _n \in \mathbb R^D$이다.

표본평균과 비슷하게, **empirical covariance** matrix는 $D \times D$ matrix로, 다음과 같이 정의된다.

$$
\boldsymbol \Sigma := \frac{1}{N} \sum^N _{n=1} (\boldsymbol x _n - \bar{\boldsymbol x})(\boldsymbol x _n - \bar{\boldsymbol x})^\intercal \tag{6.42}
$$

</div>

특정 데이터셋에 대해 통계량을 계산하기 위해 실현값(관측값) $\boldsymbol x _1, \cdots, \boldsymbol x _N$을 이용하고, 식 (6.41)과 (6.42)를 이용하면 된다. 표본분산행렬은 [symmetric, positive semidefinite](/project/mml/Analytics-Geometry/#Symmetric,-Postivie-Definite-Matrix)하다.

### Three Expressions for the Variance

이제 하나의 확률변수 $X$에 초점을 맞추고 앞장의 공식들을 이용해 다음의 세 가지 공식을 유도해보자. 다음의 유도식은 적분에 주의를 기울인다는 점을 제외하면 모분산과 동일하다. 분산의 일반적인 정의는 확률변수 $X$와 기댓값의 차이의 제곱근으로 정의된다. 즉,

$$
\mathbb V _X [x] := \mathbb E _X [(x - \mu)^2] \tag{6.43}
$$

위 식의 기댓값과 mean $\mu = \mathbb E _X (x)$는 식 (6.32)를 이용하여 계산되며, $X$가 이산확률변수인지 연속확률변수인지에 따라 달라지게 된다. 이는 새로운 확률변수 $Z := (X - \mu)^2$의 기댓값과도 같다.

식 (6.43)의 분산을 경험적으로 추정할 때, 두 단계를 resort할 필요가 있다. 첫번째는 식 (6.41)을 이용하여 mean $\mu$를 계산하기 위한 것이고, 두번째는 이 추정치 $\hat{\mu}$를 이용하여 분산을 계산하는 것이다. 이러한 단계는 term을 재정렬하여 하나로 줄일 수 있는데, (6.43)에 있는 식을 이용하여 흔히말하는 **raw-score formula for variance**로 변환하는 것이다.

$$
\mathbb V _X [x] = \mathbb E _X [x^2] - (\mathbb E _X [x])^2 \tag{6.44}
$$

위 식은 $x _i$와 $x^2 _i$를 한번에 계산할 수 있기 때문에 위와는 달리 한번에 계산할 수 있다. 그러나 이러한 방법을 사용하면 numerically unstable하다. raw-score formula for variance는 머신러닝에서 bias-variance trade off를 계산하는데 유용하게 사용된다.

분산을 이해하는 세번째 방법은 관측의 모든 쌍의 차이를 더하는 것이다. 확률변수 $X$의 실현인 어떤 샘플 $x _1, \cdots, \x _N$를 생각해보자. 그리고 이에대한 $(x _i, x _j)$쌍의 차이의 제곱을 계산해보자. 제곱을 펼쳐보면 $N^2$의 쌍의 차이의 합이 관측치의 표본분산을 이루는 것을 확인할 수 있다.

$$
\frac{1}{N^2} \sum^N _{i, j=1} (x _i - x _j)^2 = 2 \left [ \frac{1}{N} \sum^N _{i=1} x^2 _i - (\frac{1}{N} \sum^N _{i=1} x _i)^2 \right ] \tag{6.45}
$$

이 식을 살펴보면 아까 보았던 raw-score formula (6.44)의 2배가 되는 것을 볼 수 있다. 이는 pairwise distance ($N^2$개)의 합을 평균과의 거리의 합($N$개)으로 표현할 수 있다는 뜻이다. 기하학적으로 이는 pairwise distance와, 점들의 집합의 평균으로부터 점들까지의 거리가 같다는 뜻이다. Computation의 측면으로는, mean을 계산(합 안에 $N$개의 term)한 다음 분산을 계산(다시 합 안에 $N$개의 term)하여 $N^2$ term을 갖는 식((6.45)의 좌변)을 얻을 수 있다는 뜻이다.

### Sums and Transformations of Random Variables

textbook distributions에 의해 잘 설명되지 않는 현상을 모델링하고 싶을 수 있다. 따라서 확률변수의 조작을 통해 이를 수행하는 것을 생각해볼 수 있다.

states $\boldsymbol x, \boldsymbol y \in \mathbb R^D$를 갖는 확률변수 $X, Y$를 생각해보자. 그러면,

$$
\begin{align}
\mathbb E[\boldsymbol x + \boldsymbol y] = \mathbb E[\boldsymbol x] + \mathbb E[\boldsymbol y] \tag{6.46} \\
\mathbb E[\boldsymbol x - \boldsymbol y] = \mathbb E[\boldsymbol x] - \mathbb E[\boldsymbol y] \tag{6.47} \\
\mathbb V[\boldsymbol x + \boldsymbol y] = \mathbb V[\boldsymbol x] + \mathbb V[\boldsymbol y] + \text{Cov}[\boldsymbol x, \boldsymbol y] + \text{Cov}[\boldsymbol y, \boldsymbol x] \tag{6.48} \\
\mathbb V[\boldsymbol x - \boldsymbol y] = \mathbb V[\boldsymbol x] + \mathbb V[\boldsymbol y] - \text{Cov}[\boldsymbol x, \boldsymbol y] - \text{Cov}[\boldsymbol y, \boldsymbol x] \tag{6.49}
\end{align}
$$

mean과 (co)varaince는 확률변수의 affine transformation을 수행할 때 유용한 성질들을 유지한다. mean $\boldsymbol {\mu}$와 covariance matrix $\boldsymbol{\Sigma}$, $\boldsymbol x$의 (deterministic) affine transformation $\boldsymbol y = \boldsymbol A \boldsymbol x + \boldsymbol b$를 갖는 확률변수 $X$를 생각해보자. 그러면, $\boldsymbol y$는 확률변수로, 다음과 같은 mean vector와 covariance matrix를 갖는다.

$$
\begin{align}
\mathbb E _Y[\boldsymbol y] = \mathbb E _X[\boldsymbol A \boldsymbol x + \boldsymbol b] =  \boldsymbol A \mathbb E _X[\boldsymbol x] + \boldsymbol b = \boldsymbol A \boldsymbol \mu + \boldsymbol b \tag{6.50} \\
\mathbb V _Y[\boldsymbol y] = \mathbb V _X[\boldsymbol A \boldsymbol x + \boldsymbol b] =  \mathbb V _X[\boldsymbol A \boldsymbol x] = \boldsymbol A \mathbb V _X[ \boldsymbol x] \boldsymbol A^\intercal =
\boldsymbol A \boldsymbol \Sigma + \boldsymbol A^\intercal \tag{6.51} \\
\end{align}
$$

또한,

$$
\begin{align}
\text{Cov}[\boldsymbol x, \boldsymbol y] = \mathbb E [\boldsymbol x (\boldsymbol A \boldsymbol x + \boldsymbol b)^\intercal] - \mathbb E [\boldsymbol x] \mathbb E [\boldsymbol A \boldsymbol x + \boldsymbol b]^\intercal \tag{6.52a} \\
= \mathbb E [\boldsymbol x] \boldsymbol b^\intercal + \mathbb E [\boldsymbol x \boldsymbol x^\intercal] \boldsymbol A^\intercal - \boldsymbol \mu \boldsymbol b^\intercal - \boldsymbol \mu \boldsymbol \mu^\intercal \boldsymbol A^\intercal \tag{6.52b} \\
\boldsymbol \mu \boldsymbol b^\intercal - \boldsymbol \mu \boldsymbol b^\intercal + (\mathbb E [\boldsymbol x \boldsymbol x^\intercal] - \boldsymbol \mu \boldsymbol \mu^\intercal) \boldsymbol A^\intercal \tag{6.52c} \\
\stackrel{6.38}{=} \boldsymbol \Sigma \boldsymbol A^\intercal \tag{6.52d}
\end{align}
$$

여기서 $\boldsymbol \Sigma=\mathbb E [\boldsymbol x \boldsymbol x^\intercal] - \boldsymbol \mu \boldsymbol b^\intercal$는 $X$의 covariance가 된다.

### Statistical Independence

<div class="notice--warning" markdown="1">

**Definition 6.10** (Independence). 두 확률변수 $X, Y$는 **통계적으로 독립**임과 다음은 동치이다.

$$
p(\boldsymbol x, \boldsymbol y) = p(\boldsymbol x)p(\boldsymbol y) \tag{6.53}
$$

</div>

직관적으로 $\boldsymbol y$ (한번 알려지면)의 값이 $\boldsymbol x$에 대해 어떠한 추가적인 정보도 주지 못하면 두 확률변수는 독립이다 (반대도 마찬가지). 만약 $X, Y$가 통계적으로 독립이면 다음을 만족한다.

- $p(\boldsymbol y \rvert \boldsymbol x) = p(\boldsymbol y)$
- $p(\boldsymbol x \rvert \boldsymbol y) = p(\boldsymbol x)$
- $\mathbb V _{X, Y} [\boldsymbol x + \boldsymbol y] = \mathbb V _X[\boldsymbol x] + \mathbb V _Y[\boldsymbol y]$
- $\text{Cov} _{X, Y} [\boldsymbol x, \boldsymbol y] = \boldsymbol 0 $

마지막 항목의 반대는 성립하지 않을 수 있다. 즉, 두 확률변수가 0의 covariance를 갖더라도 통계적으로 독립하지 않을 수 있다. 이를 이해하기 위해서는 covariance는 오직 linear dependence만을 측정한다는 사실을 떠올릴 필요가 있다. 그러므로, nonlinearly dependent random variable은 0의 covariance를 갖을 수 있다.

머신러닝에서는 종종 **independent and identically distributed(i.i.d.)**인 확률변수로 모델링할 수 있는 문제들만 다루게 된다. 두개 이상의 확률변수에 대해서는 "independent"라는 단어는 주로 mutually indenpendent random variable만 다루게 된다. 이는 모든 부분집합이 independent한 경우이다. "identically distributed"라는 말은 모든 확률변수가 같은 분포에서 왔다는 뜻이 된다.

머신러닝에서 중요한 또 다른 개념은 conditional independence이다.

<div class="notice--warning" markdown="1">

**Definition 6.11** (Conditional Independence). 확률변수 $X, Y$가 주어진 $Z$에 대해 **conditinally independent**하다는 것은 다음과 동치이다 (iff).

$$
p(\boldsymbol x, \boldsymbol y \rvert \boldsymbol z) = p(\boldsymbol x \rvert \boldsymbol z) p(\boldsymbol y \rvert \boldsymbol z) \quad \text{for all } \boldsymbol z \in \mathbb Z \tag{6.55}

$$

여기서 $\mathbb Z$는 확률변수 $Z$의 state에 대한 집합이다. $Z$가 주어졌을 때 $X$가 $Y$에 대해 conditionally independent하다는 것은 $X ⫫ Y \rvert Z$로 쓴다.

</div>

Definition 6.11은 (6.55)에 나와있는 관계가 $\boldsymbol z$의 모든 값에 대해 참이 성립해야 한다. 식 (6.55)의 해석은 "$\boldsymbol z$에 대한 지식이 주어졌을 때, 분포 $\boldsymbol x$와 $\boldsymbol y$가 분해된다"로 이해할 수 있다. 독립은 $X ⫫ Y \rvert \varnothing$로 쓸 경우 conditional independence의 특수한 경우로 생각할 수 있다. 확률의 곱셈규칙 (6.22)를 이용하여 (6.55)의 좌변을 전개하여 다음과 같은 식을 얻을 수 있다.

$$
p(\boldsymbol x, \boldsymbol y \rvert \boldsymbol z) = p(\boldsymbol x \rvert \boldsymbol y, \boldsymbol z)p(\boldsymbol y \rvert \boldsymbol z) \tag{6.56}
$$

(6.56)과 (6.55)의 우변을 비교해보면, $p(\boldsymbol y \rvert \boldsymbol z)$가 둘 다 등장하는 것을 알 수 있다. 따라서,

$$
p(\boldsymbol x \rvert \boldsymbol y, \boldsymbol z) = p(\boldsymbol x \rvert \boldsymbol z)p(\boldsymbol y \rvert \boldsymbol z) \tag{6.57}
$$

위 식은 조건부 독립에 대해 또 다른 표현인 $Y ⫫ X \rvert Z$가 된다.

### Inner Products of Random Variables

지난 시간에 봤던 [inner product](/project/mml/Analytics-Geometry/Inner-Product)의 정의를 떠올려보자. 여기서는 두 확률변수의 내적을 살펴볼 것이다. 만약 두개의 확률변수 $X, Y$가 uncorrelate하다면, 다음이 성립한다.

$$
\mathbb V[x + y] = \mathbb V[x] + \mathbb V[y] \tag{6.58}
$$

분산은 제곱을 통해 측정되므로, 이는 피타고라스 정리 $c^2 = a^2 + b^2$와 비슷하게 생기게 된다.

확률변수는 벡터공간 내 벡터로 생각할 수 있으며, 기하학적 성질을 얻기 위해 내적을 정의할 수 있다. zero mean을 갖는 두 확률변수 $X, Y$에 대해, 아래와 같이 정의하면 내적을 얻을 수 있다.

$$
\langle X, Y \rangle:= \text{Cov}[x, y] \tag{6.59}
$$

covariance는 symmetric, positive definite하고, 각 argument에 대해 linear함을 알 수 있다.

$$
\begin{align}
\text{Cov}[x, x] = 0 \iff x =0 \\
\text{Cov}[\alpha x + z, y] = \alpha \text{Cov}[x, y] + \text{Cov}[z, y] ~ \text{for } \alpha \in \mathbb R
\end{align}
$$

확률변수의 길이는 다음과 같다.

$$
\| X \| = \sqrt{\text{Cov}[x, x]} = \sqrt{\mathbb V[x]} = \sigma[x] \tag{6.60}
$$

즉, 이의 표준편차가 된다. 확률변수가 "더 길수록", 더욱 불확실해진다. 따라서 길이가 0이라면 deterministic하다.

두 확률변수 $X, Y$사이의 각도 $\theta$에 대해서는,

$$
\cos \theta = \frac{\langle X, Y \rangle}{\| X \| \| Y \|} = frac{\text{Cov}[x, y]}{\sqrt{\mathbb V[x] \mathbb V[y]}} \tag{6.61}
$$

를 얻는다. 이는 correlation이 된다. 이는 즉, 상관관계를 기하학적으로 두 확률변수에 대한 각도로 볼 수 있다는 뜻이다. 지난시간에 $X \perp Y \iff \langle X, Y \rangle = 0$임을 살펴보았다. 이 뜻은 확률변수에서 $X$와 $Y$가 orthogonal함과 $\text{Cov}[x, y]=0$은 동치라는 뜻이다. 즉, 이는 uncorrelated를 의미한다. 아래 그림에 이에 대한 설명이 나와있다.

![image](https://user-images.githubusercontent.com/47516855/122956858-58ad5900-d3bc-11eb-8ecc-6f5eb00702fd.png){: .align-center}{:width="500"}

## Gaussian Distribution

Gaussian distribution은 가장 널리 연구된 확률분포로, 연속확률변수에 대한 분포이다. 이는 종종 **normal distribution**으로 부르기도 한다. 정규분포가 중요한 이유는 정규분포가 계산적으로 용이한 특성을 많이 갖고 있기때문이다. 특히, linear regression을 위한 likelihood와 prior를 정의하거나 (Chapter 9), density estimation에서 Gaussian mixture model을 고려할 때 사용한다 (Chapter 11).

또한, 머신러닝의 많은 영역에서 가우시안 분포를 사용함으로써 얻는 이점이 있는데, 대표적으로 Gaussian processes, variationalinference, reinforcement learning이 있다. 또한, signal processing (e.g., Kalman filte), control (e.g.,linear quadratic regulator), statistics (e.g., hypothesis testing)과 같은 다른 영역에서도 널리 사용된다.

단변량 확률변수에 대해 가우스 분포는 다음과 같은 밀도함수를 갖는다.

$$
p(x \rvert \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left ( - \frac{(x-\mu)^2}{2 \sigma^2} \right ) \tag{6.62}
$$

**multivariate  Gaussian  distribution**는 **mean** vector $\boldsymbol \mu$와 **covariance matrix** $\boldsymbol \Sigma$를 통해 특성화되고, 다음과 같이 정의된다.

$$
p(\boldsymbol x \rvert \boldsymbol \mu, \boldsymbol \Sigma) = (2 \pi)^{-\frac{D}{2}} \lvert \boldsymbol \Sigma \rvert^{-\frac{1}{2}} \exp (- \frac{1}{2} (\boldsymbol x - \boldsymbol \mu)^\intercal \boldsymbol \Sigma^{-1} (\boldsymbol x - \boldsymbol \mu)) \tag{6.63}
$$

여기서 $\boldsymbol x \in \mathbb R^D$이다. $p(\boldsymbol x)=\mathcal N(\boldsymbol x \lvert \boldsymbol \mu, \boldsymbol \Sigma)$ 혹은 $X \sim \mathcal N(\boldsymbol \mu, \boldsymbol \Sigma)$로 쓴다. Figure 6.7은 bivariate Gaussian (mesh)이며, 이에 따른 등고선 그래프이다.

![image](https://user-images.githubusercontent.com/47516855/123113776-c8354e00-d479-11eb-9566-64a40dd8943a.png){: .align-center}{:width="700"}

mean이 0이고 동일한 covariance를 갖을 경우 ($\boldsymbol x=0, \boldsymbol \Sigma = \boldsymbol I$) 이를 **standard normal distribution**이라고 한다.

가우스 분포는 marginal/conditional distribution에서 closed-form으로 표현할 수 있으므로 statistical estimation과 machine learning에서 널리 사용된다. Chapter 9에서는 closed-form을 통해 linear regression을 표현할 것이다. 가우시안 확률변수를 모델링하는 것의 주요 장점은 variable transformations (Section 6.7)이 필요하지 않다는 것이다. 가우스 분포는 mean과 covariance만으로 완벽히 표현할 수 있기 때문에 확률변수의 mean과 covariance에 transformation을 적용하여 transformed distribution을 얻을 수 있다.

### Marginals and Conditionals of Gaussians are Gaussians

이제 다변량 확률변수의 일반적인 경우에 대해 marginalization과 conditioning을 만들어보자. 이에 대해 처음접하게 되어 헷갈리는 경우, 두 개의 단변량 확률변수를 대신하여 고려할 것을 권한다. $X, Y$가 서로 다른 차원을 갖는다고 가정하자. 확률의 합법칙과 조건부에 대한 영향을 파악하기 위해, 명시적으로 concatenated states $[\boldsymbol x^\intercal, \boldsymbol y^\intercal]$에 대한 가우스분포를 다음과 같이 표현하겠다.

$$
p(\boldsymbol x, \boldsymbol y) = \mathcal N \left ( 
  \begin{bmatrix}
    \boldsymbol \mu _x \\ \boldsymbol \mu _y 
  \end{bmatrix}
,
  \begin{bmatrix}
    \boldsymbol \Sigma _{xx} & \boldsymbol \Sigma _{xy} \\
    \boldsymbol \Sigma _{yx} & \boldsymbol \Sigma _{yy}
  \end{bmatrix}
\right ) \tag{6.64}
$$

여기서 $\boldsymbol \Sigma _{xx} = \text{Cov}[\boldsymbol x, \boldsymbol x]$와 $\boldsymbol \Sigma _{yy} = \text{Cov}[\boldsymbol y, \boldsymbol y]$는 $\boldsymbol x, \boldsymbol y$의 marginal covariance matrix이다. $\boldsymbol \Sigma _{xy} = \text{Cov}[\boldsymbol x, \boldsymbol y]$는 $\boldsymbol x, \boldsymbol y$의 cross-covariance matrix이다.

조건부 분포 $p(\boldsymbol x, \boldsymbol y)$ 또한 Gaussian이고 (Figure 6.9의 c), 다음과 같이 주어진다 (Section 2.3과 Bishop, 2006을 통해 유도)

$$
\begin{align}
p(\boldsymbol x, \boldsymbol y) &= \mathcal N (\boldsymbol \mu _{x \rvert y}, \boldsymbol \Sigma _{x \rvert y}) \tag{6.65} \\
\boldsymbol \mu _{x \rvert y} &= \boldsymbol \mu _{x} + \boldsymbol \Sigma _{xy}\boldsymbol \Sigma^\intercal _{yy} (\boldsymbol y - \boldsymbol \mu _y) \tag{6.66} \\
\boldsymbol \Sigma _{x \rvert y}) &= \boldsymbol \Sigma _{xx} - \boldsymbol \Sigma _{xy} \boldsymbol \Sigma _{yy}^\intercal \boldsymbol \Sigma _{yx} \tag{6.67}
\end{align}
$$

![image](https://user-images.githubusercontent.com/47516855/123122253-d2a71600-d480-11eb-919e-f524f4fe448a.png){: .align-center}{:width="600"}

식 (6.66)에서 mean의 계산할 때 $\boldsymbol y$-value는 더 이상 random한것이 아닌 어떤 관측값이라는 것에 주의하자. 

Joint Gaussian distribution $p(\boldsymbol x, \boldsymbol y)$(식 6.64 참고)의 marginal distribution $p(\boldsymbol x)$은 
그 자체로 가우시안 분포이며, 식 (6.20)의 합법칙을 적용하여 계산한다.

$$
p(\boldsymbol x) = \int p(\boldsymbol x, \boldsymbol y) \text{d} \boldsymbol y = \mathcal N(\boldsymbol x \lvert \boldsymbol \mu _x, \boldsymbol \Sigma _{xx}) \tag{6.68}
$$

이는 $p(\boldsymbol y)$에 대해서도 똑같이 적용된다. 직관적으로 (6.64)의 joint distribution을 보면 관심이 없는 변수는 전부 무시한다. 이는 위 Figure 6.9의 (b)에 나와있다.

### Product of Gaussian Densities

linear regression (Chapter 9)에서 Gaussian likelihood를 계산할 필요가 있다. 또한,  Gaussian prior (Section 9.3)를 가정해야 할 때도 있다. 베이즈 이론을 이용하여 posterior를 계산하면 두 가우스 분포의 곱으로 나타나게 된다. 두 가우스 분포의 **product** $\mathcal N(\boldsymbol x \rvert \boldsymbol a, \boldsymbol A) \mathcal N(\boldsymbol x \rvert \boldsymbol b, \boldsymbol B)$는 $c \in \mathbb R$로 인해 스케일된 가우스 분포로, $c \mathcal N(\boldsymbol x \rvert \boldsymbol c, \boldsymbol C)$로 주어지며,

$$
\begin{align}
\boldsymbol C &= ( \boldsymbol A^{-1} + \boldsymbol B^{-1})^{-1} \tag{6.74} \\
\boldsymbol c &= \boldsymbol C (\boldsymbol A^{-1} \boldsymbol a + \boldsymbol B^{-1} \boldsymbol b) \tag{6.75} \\
c &= (2 \pi)^{-\frac{D}{2}} \lvert \boldsymbol A + \boldsymbol B \rvert^{-\frac{1}{2}} \exp (- \frac{1}{2} (\boldsymbol a - \boldsymbol b)^\intercal (\boldsymbol A + \boldsymbol B)^{-1} (\boldsymbol a - \boldsymbol b)) \tag{6.76}
\end{align}
$$

scaling  constant c는 $\mathcal N(\boldsymbol a \rvert \boldsymbol b, \boldsymbol A + \boldsymbol B) = \mathcal N(\boldsymbol b \rvert \boldsymbol a, \boldsymbol A + \boldsymbol B)$이다.

### Sums and Linear Transformations

$X, Y$가 독립인 가우스 확률변수이고 (즉, $p(\boldsymbol x, \boldsymbol y)=p(\boldsymbol x)p(\boldsymbol y)$로 주어진 결합분포), $p(\boldsymbol x) = \mathcal N(\boldsymbol x \rvert \boldsymbol \mu _x, \boldsymbol \Sigma _x)$, $p(\boldsymbol y) = \mathcal N(\boldsymbol y \rvert \boldsymbol \mu _y, \boldsymbol \Sigma _y)$이면, $\boldsymbol x + \boldsymbol y$ 또한 다음과 같은 가우스 분포로 주어진다.

$$
p(\boldsymbol x + \boldsymbol y) = \mathcal N(\boldsymbol \mu _x + \boldsymbol \mu _y, \boldsymbol \Sigma _x + \boldsymbol \Sigma _y) \tag{6.78}
$$

$p(\boldsymbol x + \boldsymbol y)$가 가우스 분포임을 알고 있으므로, mean과 covariance matrix는 식 (6.46)부터 (6.49)를 이용하여 즉시 구할 수 있다. 이 성질은  linear  regression  (Chapter 9)처럼 확률변수에 대해 i.i.d.인 Gaussian noise acting을 다룰 때 중요해진다.

Theorem 6.12에서 확률변수 $x$는 두 density $p _1(x), p _2(x)$의 조합과 가중치 $\alpha$를 통해 얻어진다. 이 정리는 다변량 확률변수에서도 적용되는데, 이는 기댓값의 선형성이 여전히 만족하기 때문이다. 그러나 확률변수의 제곱에 대해서는 $\boldsymbol x \boldsymbol x^\intercal$로 변환하여 사용해야 한다.

<div class="notice--info" markdown="1">

**Theorem 6.12.** 두 단변량 가우시안 분포의 조합을 생각해보자.

$$
p(x) = \alpha p _1(x) + (1 - \alpha) p _2(x) \tag{6.80}
$$

여기서 스칼라 $0 < \alpha < 1 $은 mixture weight, $p _1(x), p _2(x)$는 서로 다른 파라미터를 갖는 univariate Gaussian densities이다 (식 (6.62)). 그러면 mixture density $p(x)$의 mean은 각 확률변수의 weighted sum형태로 주어진다.

$$
\mathbb E[x] = \alpha \mu _1 + (1 - \alpha) \mu _2 \tag{6.81}
$$

mixture density $p(x)$의 covariance는 다음과 같다.

$$
\mathbb V[x] = [\alpha \sigma^2 _1 + (1 - \alpha) \sigma^2 _2] + ([\alpha \mu^2 _1 + (1 - \alpha) \mu^2 _2] - [\alpha \mu _1 + (1 - \alpha) \mu _2]^2) \tag{6.82}
$$

</div>

> 위 정리에 대한 증명은 생략한다.

*Remark.* 위 유도는 어떠한 density에 대해서도 성립하지만 가우스 분포만이 mean과 variance만으로 결정되고, density의 혼합이 closed form으로 결정된다.
{: .notice}

mixture  density에서 개별 density는 conditional distribution으로 간주된다.

예제 6.17을  bivariate  standard  Gaussian  random variable $X$에 linear transformation $\boldsymbol A \boldsymbol x$를 적용한 것으로 간주할 것이다. 그 결과는 zero mean에 covariance $\boldsymbol A \boldsymbol A^\intercal$를 갖는 가우시안 확률 변수가 된다. 여기서 constance vector $\boldsymbol \mu$를 더하면, $\boldsymbol x + \boldsymbol \mu$는 분산에는 영향이 없고, mean $\boldsymbol \mu$, 그리고 identity covariance를 갖게된다. 따라서 어떠한 linear/affine transformation을 수행하더라도 가우스 확률변수는 여전히 가우스 분포이다.

$X \sim \mathcal N (\boldsymbol \mu, \boldsymbol \Sigma)$인 어떤 확률변수를 생각해보자. 적절한 형태의 $\boldsymbol A$에 대해, $\boldsymbol x$가 변환된 $\boldsymbol y = \boldsymbol A \boldsymbol x$가 확률변수 $Y$라 해보자. 식 (6.50)에서 처럼 기댓값이 linear operator임을 이용하여 $\boldsymbol y$의 mean을 계산하면, 

$$
\mathbb E[\boldsymbol y] = \mathbb E[\boldsymbol A \boldsymbol x] = \boldsymbol A \mathbb E[\boldsymbol x] = \boldsymbol A \boldsymbol \mu \tag{6.86}
$$

이 된다. 이와 비슷하게 $\boldsymbol y$의 분산도 구할 수 있다 (식 (6.51) 참고).

$$
\mathbb V[\boldsymbol y] = \mathbb V[\boldsymbol A \boldsymbol x] = \boldsymbol A \mathbb V[\boldsymbol x] \boldsymbol A^\intercal = \boldsymbol A \boldsymbol \Sigma \boldsymbol A^\intercal \tag{6.87}
$$

이는 확률변수 $\boldsymbol y$가 다음 분포를 따른다는 뜻이다.

$$
p(\boldsymbol y) = \mathcal N (\boldsymbol y \rvert \boldsymbol A \boldsymbol \mu, \boldsymbol A \boldsymbol \Sigma \boldsymbol A^\intercal) \tag{6.88}
$$

이제 reverse transformation을 생각해보자. 즉, 확률변수가 다른 확률변수의 linear transformation으로 이루어진 mean을 갖는 경우이다. Full rank matrix $\boldsymbol A \in \mathbb R^{M \times N}$, $M \geq N$에 대해, $\boldsymbol y \in \mathbb R^M$를 mean $\boldsymbol A \boldsymbol x$를 갖는 가우시안 확률변수라 하자. 즉,

$$
p(\boldsymbol y) = \mathcal N (\boldsymbol y \rvert \boldsymbol A \boldsymbol x, \boldsymbol \Sigma) \tag{6.89}
$$

이에 해당하는 확률분포 $p(\boldsymbol x)$는 무엇일까? $\boldsymbol A$가 invertible하다면, $\boldsymbol x = \boldsymbol A^{-1} \boldsymbol y$로 쓰고 이를 적용할 수 있을 것이다. 그러나 대부분 아니기 때문에 pseudo-inverse (3.57)을 이용해야 한다. 이는 양변에 $\boldsymbol A^\intercal$을 곱하고, $\boldsymbol A^\intercal \boldsymbol A$의 역행렬을 구하는 것이다. 이는 symmetric, positive definite하다.

$$
\boldsymbol y = \boldsymbol A \boldsymbol x \iff (\boldsymbol A^\intercal \boldsymbol A)^{-1} \boldsymbol A^\intercal \boldsymbol y = \boldsymbol x \tag{6.90}
$$

따라서 $\boldsymbol x$는 $\boldsymbol y$의 선형변환이며, 다음을 얻는다.

$$
p(\boldsymbol x) = \mathcal N (\boldsymbol x \rvert (\boldsymbol A^\intercal \boldsymbol A)^{-1} \boldsymbol A^\intercal \boldsymbol y, (\boldsymbol A^\intercal \boldsymbol A)^{-1} \boldsymbol A^\intercal \boldsymbol \Sigma \boldsymbol A (\boldsymbol A^\intercal \boldsymbol A)^{-1}) \tag{6.91}
$$

### Sampling from Multivariate Gaussian Distributions

multivariate Gaussian의 경우 이 과정은 다음의 세 단계를 거친다.
1. [0, 1] 사이의 균일분포를 제공하는 난수
2. Box-M ̈uller transform와 같은 non-linear transformation을 통해 단변량 가우스분포를 sample
3. 다변량 standard normal로부터 sample을 얻기 위해 이러한 sample을 대조

일반적인 경우 (mean이 nonzero이고 covariance가 identity matrix가 아닐 때) 가우스 확률변수의 linear transformation 성질을 이용한다. $\boldsymbol \mu$와 covariance matrix \Sigma$를 갖는 다변량 가우스 분포로부터 어떤 샘플 $\boldsymbol x _i, i=1, \dotsc, n$을 생성하는데 관심이 있다고 하자. 그러면 multivariate standard normal로부터 샘플을 생성하여 이를 구성할 것이다.

multivariate normal $\mathcal N(\boldsymbol \mu, \boldsymbol \Sigma)$로부터 샘플을 얻기 위해, 가우스 확률변수에 대한 linear transformation의 성질을 이용할 수 있다. 만약 $\boldsymbol x \sim \mathcal N(\boldsymbol 0, \boldsymbol I)$이라면, $\boldsymbol y = \boldsymbol A \boldsymbol x$이고, 이는 mean $\boldsymbol \mu$와 공분산 행렬 $\boldsymbol A \boldsymbol A^\intercal = \boldsymbol \Sigma$를 갖는 가우스 분포임을 알 수 있다. $\boldsymbol A$는 공분산 행렬의 Cholesky decomposition (Section 4.3)을 이용하면 쉽게 구할 수 있다. Cholesky decomposition의 장점은 $\boldsymbol A$가 삼각행렬로, 연산에 있어 용이하다는 점이다.




\boldsymbol \mu
\mathbb E
\mathbb V

\mathcal N(\boldsymbol \mu, \boldsymbol \Sigma)

\boldsymbol \Sigma

p(\boldsymbol x, \boldsymbol y)

{: .align-center}{:width="400"}

<div class="notice--warning" markdown="1">


</div>

> 다음은 고전적 통계학과 베이즈 통계학의 차이를 나타낸 표이다.
>
> ![](https://i.imgur.com/xc23ScH.png){: .align-center}{:width="400"}
>
> [출처: 베이지안 통계 정의 및 비교](https://adioshun.gitbooks.io/statics-with-r/content/bayesian/basics-comparison.html)

