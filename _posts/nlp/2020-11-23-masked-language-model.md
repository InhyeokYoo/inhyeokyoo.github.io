---
title:  "Masked Language Modeling (작성 중)"
excerpt: "BERT에서 쓰인 masked LM에 대해 알아보자"
toc: true
toc_sticky: true
categories:
  - NLP
tags:
  - Language Modeling
  - Masked Language Modeling

use_math: true
last_modified_at: 2020-11-23
---

기존의 language model은 AR(Auto-Regressive)을 통해 다음 단어를 예측한다. 
그러나 BERT의 등장과 함께 AR의 시대는 저물고 AE(Auto-Encoder) 시대가 오게되었다.
본 포스트에서는 BERT에서 사용한 masked language model에 대해 살펴보고, 이에 대한 근거를 진행하고자 한다.

[BERT review 보러가기](/project/nlp/bert-review/)

# Masked Language Model in BERT

BERT는 Masked Language Model (MLM) pretraining objective를 사용하여 Transformer encoder를 학습시킨다. 
이는 [Cloze task (test)](https://en.wikipedia.org/wiki/Cloze_test)에서 영감을 받은 것으로, 이 cloze task는 학생들에게 빈 칸을 채우도록 한 다음 이들의 언어 능력을 평가하는 테스트이다. Cloze test를 잘 수행하기 위해서는 context를 이해하고 단어를 잘 이해하는 능력이 필요하다.

![Cloze task](https://miro.medium.com/max/620/1*2X0uYNinK7KOQLtNknQPsg.png){: .align-center}{: width="400"}

Masked language model은 기존 unidirectional language model이 갖는 문제를 해결하기 위해 등장하게 되었다. language model은 문장의 단어를 왼쪽에서 오른쪽으로 읽으며 학습하는데, 실제로 언어를 이해하기 위해서는 역방향(backward) 또한 고려를 해야한다 (bidirectional). 그러나 이런 bidirectional은 또 다른 문제를 낳는데, **바로 단어를 예측할 때 자기 자신을 볼 수 있기 때문이다. ** 본문에서는 다음과 같이 표현하고 있다.

> Unfortunately, standard conditional language models can **only be trained left-to-right or right-to-left**, since bidirectional conditioning would allow each word to indirectly **“see itself”**, and the model could trivially predict the target word in a multi-layered context.

아래 그림은 CS224N 슬라이드에서 추출한 [자료](https://nlp.stanford.edu/seminar/details/jdevlin.pdf)이다. 보다시피 bidirectional의 경우 high layer에서 단어들이 자기 자신을 참고하는 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/47516855/100091582-23797080-2e98-11eb-828b-021cb10aa565.png){: .align-center}{: width="800"}

![bidirectional conditioning would allow each word to indirectly “see itself”](https://qph.fs.quoracdn.net/main-qimg-514d3773b91bbd2c1a72ca4e3d83f707){: .align-center}{: width="800"}

이에 대한 솔루션으로 단어의 15%를 [mask] 토큰으로 대체하게 된다. 15%에 대한 근거는 찾기가 힘들었는데, 다만 너무 적게할 경우 학습하는데 비용이 많이 들고 (즉, 충분히 학습하기가 힘들어서 더 많은 학습이 필요함), 너무 많이 할 경우 문맥(context)를 충분히 주기가 힘들어 학습하기가 어렵다고 한다.

이는 deep learning 관점에서 생각해보면 denoising auto-encoder를 사용하는 것과 같다. denoising auto-encoder에서 collapse된 부분이 mask를 씌운 것과 동일하다고 보는 것이다. 아래는 BART논문에서 발췌한 내용이다.

> The most successful approaches have been variants of masked language models, which are denoising auto encoders that are trained to reconstruct text where a random subset of the words has been masked out.

BERT에서 denoising auto-encoder를 언급하는 부분은 아래와 같다.

> **2.1  Unsupervised Feature-based Approaches**
......
To  train  sentence  representations,  prior work  has  used  objectives  to  rank  candidate  next sentences  (Jernite  et  al.,  2017;  Logeswaran  and Lee, 2018),  left-to-right  generation  of  next  sentence words given a representation of the previous sentence  (Kiros  et  al.,  2015),  or  **denoising  auto-encoder derived objectives [(Hill et al., 2016)](https://www.aclweb.org/anthology/N16-1162/)**.

> **2.2 Unsupervised Fine-tuning Approaches**
......
Left-to-right  language  modeling and auto-encoder  objectives  have  been  used for pre-training such models Howard and Ruder, 2018; Radford et al., 2018; **[Dai and Le, 2015](https://arxiv.org/pdf/1511.01432.pdf)**)

> **3.1  Pre-training BERT - Task #1: Masked LM**
......
In contrast to denoising auto-encoders (Vincent et al., 2008), we only **predict the masked words rather than reconstructing the entire input**.

이 중에서 denoising auto encoder를 사용한 것은 Hill et al., 2016과 Dai and Le, 2015의 연구이다.

# 