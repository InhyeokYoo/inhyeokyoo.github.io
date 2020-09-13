---
title:  "Deep contextualized word representations (ELMo) 구현 Issue (작성 중)"
excerpt: "PyTorch ELMo 구현 이슈 정리"
toc: true
toc_sticky: true

categories:
  - NLP
  - PyTorch
tags:
  - Language Modeling

use_math: true
last_modified_at: 2020-09-11
---

# Intro.

[지난번](https://inhyeokyoo.github.io/nlp/ELMO-Paper/)엔 ELMo에 대해 알아보았으니, 이제는 구현을 할 차례이다.
본 포스트에서는 ELMo를 구현하며 궁금한 점과 issue를 정리해보았다.

# Character CNN Embedding

처음에 읽을 때 뭐 이런 논문이 다 있나 싶었는데, 모델 구조도 정확하게 안 나와있고, 다른 논문의 citation에 의존하고 있어서 굉장히 당황스러웠다.
읽을 땐 그냥 그랬는데, 이를 막상 구현하자니 이렇게 막막할 수가... 

우선 ELMo는 [Jozefowicz et al. (2016)](https://arxiv.org/abs/1602.02410)와 [Kim et al. (2015)](https://arxiv.org/pdf/1508.06615.pdf)에 기반하고 있다고 밝히고 있으니, 이를 필히 읽어야 한다.

> The pre-trained biLMs in this paper are similar tothe  architectures in  J ́ozefowicz  et al.  (2016) andKim  et  al.  (2015)  
...  
we halved all embedding and hidden dimensions from the singlebest model CNN-BIG-LSTM in Jozefowicz et al.(2016).

#### Kim et al. (2015)

- $\mathcal C$: vocaburalry of chraceters
- $d$: the  dimensionality of character embeddings
  - 여기서 $d < \lvert \mathcal C \rvert$
- $\mathbf Q \in \mathbb R^{d \times \lvert \mathcal C \rvert}$: character embedding의 matrix
- 이러한 word embedding $\mathbf Q$와 width w를 갖는 *filter (or kernel)* $\mathbf H \in  \mathbb R^{d \times w$ 사이의 narrow convoultion을 적용
- 이후 bias와 non-linearity를 추가하여 feature map을 얻음


$e _w$: word embedding for $w$

흠.. 구현이슈 정리인데 하다보니 논문정리가 되는 듯 하다.

## Embedding layer의 vocab size는 어떻게 되는가?

character만 하므로, 26개의 alphabet만 하면 되는가 싶다가도, space라던가, apostrophe, puncation 등은 어떻게 처리하나 궁금해졌다.
