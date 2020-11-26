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
본 포스트에서는 BERT에서 사용한 masked language model에 대해 살펴보고, 이에 대한 근거를 살펴보고자 하낟.

[BERT review 보러가기](/project/nlp/bert-review/)

# Masked Language Model in BERT

BERT는 Masked Language Model (MLM) pretraining objective를 사용하여 Transformer encoder를 학습시킨다. 
이는 [Cloze task (test)](https://en.wikipedia.org/wiki/Cloze_test)에서 영감을 받은 것으로, 이 cloze task는 학생들에게 빈 칸을 채우도록 한 다음 이들의 언어 능력을 평가하는 테스트이다. Cloze test를 잘 수행하기 위해서는 context를 이해하고 단어를 잘 이해하는 능력이 필요하다. 따라서 머신이 인간처럼 잘 학습하기 위해서는 **인간이 언어를 이해하는 방법**을 잘 따라하는 것이 필요하고, 이는 Cloze task의 목표와도 잘 부합한다고 볼 수도 있을 것 같다.

![Cloze task](https://miro.medium.com/max/620/1*2X0uYNinK7KOQLtNknQPsg.png){: .align-center}{: width="400"}

Masked language model은 기존 unidirectional language model을 bidirectional로 변경하면서 갖는 문제를 해결하기 위해 등장하게 되었다. language model은 문장의 단어를 왼쪽에서 오른쪽으로 읽으며 학습하는데, 실제로 언어를 이해하기 위해서는 역방향(backward) 또한 고려를 해야한다 (bidirectional). 그러나 대부분의 연구는 unidirectional에 치중되어 있고, 설령 ELMo처럼 bidirectional하더라도 shallow concatenation을 사용할 정도로 소극적인데, 이에는 다음과 같은 문제가 있다.
- 어차피 단어의 분포를 생성하려면 unidirectional 해야 한다
- **단어를 예측할 때 자기 자신을 볼 수 있다** 

특히 중요한 것은 두 번째인데, 본문에서는 다음과 같이 표현하고 있다.

> Unfortunately, standard conditional language models can **only be trained left-to-right or right-to-left**, since bidirectional conditioning would allow each word to indirectly **“see itself”**, and the model could trivially predict the target word in a multi-layered context.

![bidirectional conditioning would allow each word to indirectly “see itself”](https://qph.fs.quoracdn.net/main-qimg-514d3773b91bbd2c1a72ca4e3d83f707){: .align-center}{: width="800"}

이에 대한 솔루션으로 단어의 15%를 [mask] 토큰으로 대체하게 된다. 15%에 대한 근거는 찾기가 힘들었는데, 다만 너무 적게할 경우 학습하는데 비용이 많이 들고 (즉, 충분히 학습하기가 힘들어서 더 많은 학습이 필요함), 너무 많이 할 경우 문맥(context)를 충분히 주기가 힘들어 학습하기가 어렵다고 한다.

이를 deep learning 구조에서 생각해보면 denoising auto-encoder를 사용하는 것과 같다. denoising auto-encoder에서 collapse된 부분이 mask를 씌운 것과 동일하다고 보는 것이다. 아래는 BART논문에서 발췌한 내용이다.

> The most successful approaches have been variants of masked language models, which are denoising auto encoders that are trained to reconstruct text where a random subset of the words has been masked out.

BERT에서 (denoising) auto-encoder를 언급하는 부분은 아래와 같다.

> **2.1  Unsupervised Feature-based Approaches**
......
To  train  sentence  representations,  prior work  has  used  objectives  to  rank  candidate  next sentences  (Jernite  et  al.,  2017;  Logeswaran  and Lee, 2018),  left-to-right  generation  of  next  sentence words given a representation of the previous sentence  (Kiros  et  al.,  2015),  or  **denoising  auto-encoder derived objectives [(Hill et al., 2016)](https://www.aclweb.org/anthology/N16-1162/)**.

> **2.2 Unsupervised Fine-tuning Approaches**
......
Left-to-right  language  modeling and auto-encoder  objectives  have  been  used for pre-training such models Howard and Ruder, 2018; Radford et al., 2018; **[Dai and Le, 2015](https://arxiv.org/pdf/1511.01432.pdf)**)

> **3.1  Pre-training BERT - Task #1: Masked LM**
......
In contrast to denoising auto-encoders (Vincent et al., 2008), we only **predict the masked words rather than reconstructing the entire input**.

이 중에서 (denoising) auto encoder를 사용한 것은 Dai and Le, 2015와 Hill et al., 2016의 연구이다. 간략하게 한 번 살펴보자.

# Semi-supervised Sequence Learning (Dai and Le, 2015)

본 논문은 auto encoder를 활용하여 pre-trained model을 만들고 성능을 평가했다. 본 논문에선 unlabeled data를 이용하여 RNN sequence learning을 향상시키기 위한 두 가지 방법을 비교하고 있다. 이 두 알고리즘은 일종의 **pre-trained** 단계에서 사용되어 supervised learning을 거치게 된다. 두 방법론은 다음과 같다.
- 일반적인 LM: $p(x _t \rvert x _1, ..., x _{t-1})$
- sequence autoencoder: input sequence를 읽어 이를 벡터로 만든 후에 다시 input sequence를 생성

이 두 가지 방법은 random initializing을 통해 모델을 end-to-end로 학습하는 것보다 더 좋았다고 한다. 또 한가지 중요한 결과는 관련된 task의 unlabeled data를 활용했을 때 generalization이 더 좋았다는 점이다.

위에서 언급한 sequence autoencoder는 seq2seq과 비슷한 구조를 갖고 있지만 unsupervised learning이라는 점에서 차이점이 있다. Sequence autoencoder는 데이터를 입력받고 encoding한 후, decoding과정에서 **원본 데이터의 복원**을 objective로 삼는다. 

![Sequence autoencoder](https://user-images.githubusercontent.com/47516855/100340636-d83d9a00-301e-11eb-9b74-00b719c46d87.png){: .align-center}{: width="800"}

이렇게 얻은 weight는 다른 supervision task에서 initialization으로 사용할 수 있다. 

# Sequential (Denoising) Autoencoder, Hill et al., 2016

본 논문에서는 DAE를 활용한 representation learning objective를 사용한다. 본래 DAE는 고정된 길이를 갖는 이미지에 적용하는데, 여기서는 가변 길이를 갖는 문장에 적용하도록 noise function $N(S \rvert p _0, p _x)$의 평균값을 이용한다. $p _0, p _x$는 0과 1사이의 값는 확률 값으로, 각 단어 $w$는 독립 확률 $p _0$를 통해 삭제된다. 그리고 문장 안에 서로 겹치지 않는 bigram $w _iw _{i+1}$에 대해 $N$은 $p _x$확률로 이를 $w _i$와 $w _{i+1}$로 바꾼다. 그후 LSTM 기반의 encoder-decoder를 통해 원래 문장을 예측하도록 한다. 즉, 원래의 source sentence는 ground truth로, input은 $N(S \rvert p _0, p _x)$가 되는 것이다. 이러면 novel word sequence를 distributed representation으로 표현할 수 있게 된다. 만일 $p _0, p _x$가 0이 되면, 앞서 언급한 sequence autoencoder와 동일한 objective가 된다.

사실 여기까지만 봐서는 MLM이 학습하는 것이 무엇인지, 왜 잘되는 것인지를 알기는 힘들다. 따라서 BERT 이후의 후속논문을 통해 MLM에 대해서 더 이해보도록 하자.

[Why Do Masked Neural Language Models Still Need Common Sense Knowledge?](https://arxiv.org/pdf/1911.03024.pdf)는 MNLM (Masked Neural Language Model)을 common sense knowledge 측면에서 분석한 논문이다.

