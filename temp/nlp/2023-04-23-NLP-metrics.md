---
title:  "자연어처리 평가지표에 대해 알아보자"
toc: true
toc_sticky: true
permalink: /project/nlp/metrics
categories:
  - NLP
use_math: true
last_modified_at: 2023-04-23
---

## 들어가며


## Precision(정밀도)

**Precision (정밀도)**은 모델이 True라고 분류한 것 중 실제로 True인 것들의 비율이다.

$$
\text{Precision} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}
$$

## Recall(재현율)

**Recall(재현율)**은 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율이다.

$$
\text{Recall} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}
$$

예를 들어 모델이 모든 input에 대해 True를 내보내게 되어있다고 생각해보자.
이 경우 False Negative는 작고 True Positive와 False Positive가 크게된다.
따라서 precision은 낮지만 recall은 100%가 된다.

반대로 확실하지 않은 경우 예측을 보류하여 FP의 경우의 수를 줄이면 어떻게 될까?
이 경우엔 False Positive가 줄고 False Negative가 크게 되어 precision은 100%가 되며 recall은 작아진다.

Accuracy는 TP와 TN을 둘 다 고려하여 모델의 성능을 직관적으로 표현하는 지표처럼 보인다.
그러나 domain의 bias를 고려할 수 없다는 문제가 생긴다.
만일 데이터의 레이블이 불균형하다면 특정 사건을 예측하는 성능이 매우 낮을 수 밖에 없다.

이를 보완하는 것이 다음에 볼 F1 score이다.

## F1-Score

**F1-score**는 레이블이 불균형할 때 모델의 성능을 정확하게 평가할 수 있다.
F1 score는 Precision과 Recall의 _조화평균_ 으로 다음과 같이 계산된다.

$$
\text{F1-score} = 2 \times \frac{1}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}}
$$

조화평균을 사용하는 이유는 다음과 같다.

> 조화 평균은 표본들이 비율이나 배수이지만 각 표본값은 독립적이고 표본끼리 곱한 값이 의미가 없을 때, 효율이나 속도 처럼 역수가 의미가 있을 때, 각 표본들이 비중이 같을 때 주로 쓰인다. 
> 이런 표본값은 그냥 산술평균을 하면 값이 큰 쪽이 작은 쪽보다 부당하게 높은 비중을 차지하는 것을 시정하고 공정한 평균을 낼 수 있다. 
> 성능이나 효율 속도 시간당 진도 통계 등에 그런 통계가 유효할 때가 많다. 
> 예를 들어 여러 은행의 평균 이자율 이라든지 주식의 평균 주가수익률 이라든지 같은 것을 계산할 때 쓰는 게 좋다. 
> 각 표본값들이 비중이 다를 때는 가중조화평균을 사용해야 한다.
> 
> 출처: [나무위키](https://namu.wiki/w/%ED%8F%89%EA%B7%A0?from=%EC%A1%B0%ED%99%94%20%ED%8F%89%EA%B7%A0#s-2.3)

아래 그림을 보면 설명한 바와 같이 **큰 쪽이 작은 쪽보다 부당하게 높은 비중을 차지하는 것**을 방지하는걸 볼 수 있다.


![Fig.1-harmonic-mean-of-precision-and-recall]({{site.url}}{{site.baseurl}}/assets/posts/nlp/NLP-metrics-Fig.1.png){: .align-center}{: width="600"}


### Micro F1

자주 등장하는 레이블에 대한 성능을 추적한다.

### Macro F1

**Macro F1-score**는 빈도를 무시하고 모든 레이블에 대한 성능을 평균내어 구한다.
0과 1사이의 값을 가지며, 1에 가까울수록 좋은 성능을 보인다.

Macro-F1의 경우 모든 class의 값에 동등한 중요성을 부여한다. 
즉, 비교적 적은 클래스에서 성능이 좋지 않다면, Macro-F1의 값은 낮게 나타날 것이며
Macro-F1은 먼저 class와 label의 각각 F1-score를 계산한 뒤 평균내는 방식으로 동작한다.

클래스별/레이블별 F1-score의 평균으로 정의된다.

## mAP (mean Average Precision)

- Recall @ k (r@k)를 보조하는 지표

## EM (Exact Match)

- 예측과 정답에 있는 문자가 정확히 일치하면 EM=1이고 아니면 0
- QA에서 사용


## Natural Language Generation

## BLEU (Bilingual Evaluation Understudy Score)

- 사용처:
  - Machine Translation
- 특징:
  - n-gram의 precision을 통해 값을 계산
  - 블루라 읽음
  - 0에서 1사이의 값을 갖음
  - 값이 높을수록 두 문장이 가까움
- 장점:
  - 언어에 구애받지 않음
  - 빠른 계산 속도
- 단점:
  - 동의어 고려 X
  - 토큰을 입력으로 받기 때문에 tokenization이 달라지면 결과가 달라짐

BLEU는 machine translation에서 사용하는 평가지표로, n-gram precision을 근간으로 한다.
BLEU는 두 텍스트를 비교하여 정답 텍스트에 있는 단어가 생성된 텍스트에 얼마나 자주 등장하는지 카운트한 뒤 이를 생성된 텍스트의 길이로 나누어 얻어진다.

$$
\begin{aligned}
\text{BLEU} &= \text{BP} \times (\prod^N _{n=1} p _n)^{1/N}
\end{aligned}
$$

먼저 두번째 항인 $p_n$은 살펴볼 것인데, 그 전에 notation을 정리하도록 하자.

- $\hat y$: candidate string. 생성된 문장을 의미한다.
- $y$: reference string. 참조 문장을 의미한다. 일반적으로는 여러개가 될 수 있다.
- $\hat{S}:= (\hat y^{(1)}, \cdots, \hat y^{(M)})$: candidate corpus로 **생성된 문장 corpus**를 의미한다.
- $S_i := (y^{(i, 1)}, ..., y^{(i, N_i)})$: reference candidate strings의 리스트로 i번째 생성 문장 $y_i$에 대한 **정답문장의 리스트**를 의미한다.
- $S = (S_1, \cdots, S_M)$: reference candidate corpus로 **정답 문장들의 corpus**을 의미한다.

$p_n$은 modified n-gram precision으로 n-gram 단위의 precision을 의미하며, 이에 대한 기하평균으로 구성되어 있다.

$$
\begin{aligned}
p_n := \frac{\sum_{i=1}^M\sum_{s\in G_n( \hat y^{(i)})} \min(C(s, \hat y^{(i)}), \max_{y\in S_i} C(s, y))}{\sum_{i=1}^M\sum_{s\in G_n( \hat y^{(i)})} C(s, \hat y^{(i)})}
\end{aligned}
$$

여기서 각 항목은 다음과 같다.
- $G_n(y)$: $y$에 대한 n-gram
- $C(s, y)$: $s$가 $y$의 substring으로서 등장한 횟수

복잡하므로 분모부터 살펴보자.

$$
\begin{aligned}
{\sum_{i=1}^M\sum_{s\in G_n( \hat y^{(i)})} C(s, \hat y^{(i)})}
\end{aligned}
$$

$1$부터 $M$개의 문장에 대해, $\hat y^{(i)}$의 n-gram $G_n( \hat y^{(i)})$ 내 원소 $s$와 $y$

$$
\begin{aligned}
\sum_{s\in G_n(\hat y)}  C(s, y) = \text{number of n-substrings in } \hat y\text{ that appear in } y
\end{aligned}
$$


그 다음 첫번째 항인 $\text{BP}$는 **brevity penalty**라 부르는 것으로 생성된 텍스트가 짧을수록 precision이 더 높아 BLEU가 높아지는 것에 대한 패널티 값이다.

$$
\begin{aligned}
\text{BP} = \min (1, e^{1 - \ell _\text{r} / \ell _\text{c}})
\end{aligned}
$$
- $r$: reference string (정답 문장) $\hat y$의 길이
- $c$: candidate string (생성 문장)의 길이

이를 해석하면 아래와 같게된다.
- $r \leq c$이면 $\text{BP} = 1$ 이다. 따라서 긴 $c$는 제외하고 짧은 경우에만 penalty를 준다.
- $r > c$이면 지수항의 ${1 - \ell _\text{r} / \ell _\text{c}}$이 작아져 BLEU 점수를 기하급수적으로 작게 만든다.


[딥 러닝을 이용한 자연어 처리 입문 14-03강](https://wikidocs.net/31695)과 [위키피디아 페이지](https://en.wikipedia.org/wiki/BLEU#Mathematical_definition)를 보면 어떤 과정을 통해 BLEU를 계산하게 되는지 차근차근 설명하고 있으므로 이를 확인해보자.

- 두 텍스트를 비교할 때 정답 텍스트에 있는 단어가 생성된 텍스트에 얼마나 자주 등장하는지 카운트한 뒤 생성된 텍스트 길이로 나눔
- 반복적인 생성에 보상을 주지 않도록 분자의 count를 clip
  - 생성된 문장에서 ngram 등장 횟수는 참조 문장에 나타난 횟수보다 작게 됨
- 재현율을 고려하지 않기 때문에 짧지만 정밀하게 생성된 시퀀스가 긴 문장보다 유리
  - 이를 방지하기 위해 brevity penalty를 통해 보정
- 마지막 항은 1에서 N까지 n-gram에서 수정 정밀도의 기하 평균
- BLUE-4 점수가 실제로 많이 사용


**transformers에서 사용하기**

과거엔 `datasets.Metric`을 통해 사용하였으나 `datasets==2.5.0` 이후로 deprecated 되었으니 앞으로는 `evaluate`를 통해 평가하자.
사용법은 기존과 비슷한 것으로 보인다.

아래는 `evauluate`를 활용한 예시이다.

```py
import evaluate

bleu_metric = evaluate.load("bleu")
# .add()의 경우 동작하지 않는데 그 원인을 모르겠음
# bleu_metric.add(prediction="the the the the the the", reference=["the cat is on the mat"])
# ValueError: Evaluation module inputs don't match the expected format.
# Expected format: {'predictions': Value(dtype='string', id='sequence'), 'references': Value(dtype='string', id='sequence')},
# Input predictions: the the the the the the,
# Input references: ['the cat is on the mat']

bleu_metric.add_batch(predictions=["the the the the the the"], references=["the cat is on the mat"])
bleu_metric.compute()
# 혹은 `compute()`에 바로 인자를 넣어 계산할 수도 있음
# bleu_metric.compute(predictions=["the the the the the the"], references=["the cat is on the mat"])
```

결과:

```pycon
{'bleu': 0.0,
 'precisions': [0.3333333333333333, 0.0, 0.0, 0.0],
 'brevity_penalty': 1.0,
 'length_ratio': 1.0,
 'translation_length': 6,
 'reference_length': 6}
```

## SacreBLEU

- BLEU에 토큰화 단계를 내재화해 문제 해결

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- 루주([ro͞oZH])라 발음
- 높은 재현율이 정밀도보다 훨씬 더 중요한 요약 등에서 사용
- 생성된 텍스트와 정답에서 여러 가지 n-gram이 얼마나 자주 등장하는지 비교하는 점에서 BLEU와 유사
  - 그러나 정답에 있는 n-gram이 생성된 텍스트에 얼마나 많이 등장하는지도 확인한다는 점에서 차이점을 보임
- 정밀도를 완전히 제거하면 부정적인 영향이 커지기 때문에 


{: .align-center}{: width="300"}
