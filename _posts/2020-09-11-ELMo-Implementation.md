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

> The pre-trained biLMs in this paper are similar to the  architectures in  J ozefowicz  et al. (2016) and Kim  et  al.  (2015)  
...  
we halved all embedding and hidden dimensions from the singlebest model CNN-BIG-LSTM in Jozefowicz et al.(2016).

## 전처리과정

biLM을 돌리기 위해서는, 
- sequence 내에 word를 그대로 받은 이후에 (tensor X)
- 각 word를 iterate하며 character를 모아야 함

말이야 쉽지 이미 전처리 라인을 `Field`와 `Vocab`으로 구성해놓은 걸 어느 세월에 character level로 바꾸나 싶어서 당황스러웠다. 또한, language modeling에서의 전처리 과정에서 특수문자는 어떻게 처리하는지도 모르겠다.

또한, WikiText2는 영어 외에도 일본어같은 다양한 언어가 들어있다. 따라서 이를 적절하게 처리할 방법이 필요하다.

## WikiText2에서 batch_size로 만드는 방법

`BPTTIterator`를 이용해서 iterator를 만들어보았는데, 분명 batch size 옵션을 넣었는데도 불구하고 `[1, seq_len]`의 tensor를 반환한다. 뭔가 이상하다 싶어서 알아봤는데, `bptt_len`옵션을 넣어줘야 batch로 반환하는 것으로 보인다. 아래와 같이 작성하면 잘 작동한다.

<script src="https://gist.github.com/InhyeokYoo/827545227b081452cd2345010e23aff8.js"></script>

## LM의 전처리 과정

character를 embedding 하므로, 26개의 alphabet만 하면 되는가 싶다가도, space라던가, apostrophe, puncation 등은 어떻게 처리하나 궁금해졌다. 또한, 앞서 언급했듯, WikiText2는 영어 외에도 일본어같은 다양한 언어가 들어있다. 따라서 이를 적절하게 처리할 방법이 필요할 것 같다.
