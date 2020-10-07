---
title:  "Deep contextualized word representations (ELMo) 구현 Issue (작성 중)"
excerpt: "PyTorch ELMo 구현 이슈 정리"
toc: true
toc_sticky: true

categories:
  - NLP
  - PyTorch
  - Fucking Lazy
tags:
  - Language Modeling

use_math: true
last_modified_at: 2020-09-25
comments: true
---

# Intro.

[지난번](https://inhyeokyoo.github.io/nlp/ELMO-Paper/)엔 ELMo에 대해 알아보았으니, 이제는 구현을 할 차례이다.
본 포스트에서는 ELMo를 구현하며 궁금한 점과 issue를 정리해보았다. 완성본은 [다음 repo](https://github.com/InhyeokYoo/NLP/tree/master/papers/4.%20ELMo)을 참고하자.

# Character CNN Embedding

처음에 읽을 때 뭐 이런 논문이 다 있나 싶었는데, 모델 구조도 정확하게 안 나와있고, 다른 논문의 citation에 의존하고 있어서 굉장히 당황스러웠다.
읽을 땐 그냥 그랬는데, 이를 막상 구현하자니 이렇게 막막할 수가... 

우선 ELMo는 [Jozefowicz et al. (2016)](https://arxiv.org/abs/1602.02410)와 [Kim et al. (2015)](https://arxiv.org/pdf/1508.06615.pdf)에 기반하고 있다고 밝히고 있으니, 이를 필히 읽어야 한다.

> The pre-trained biLMs in this paper are similar to the  architectures in  J ozefowicz  et al. (2016) and Kim  et  al.  (2015)  
...  
we halved all embedding and hidden dimensions from the singlebest model CNN-BIG-LSTM in Jozefowicz et al.(2016).

아래는 이를 종합적으로 정리한 내용이다 (cs224n)

![CNN Embedding](https://user-images.githubusercontent.com/47516855/94338680-2b9b6800-002f-11eb-83b3-34f3f5df884d.png)

## torch text를 활용한 chracter embedding 방법

biLM을 돌리기 위해서는, 
- sequence 내에 word를 그대로 받은 이후에 (tensor X)
- 각 word를 iterate하며 character를 모아야 함

말이야 쉽지 이미 전처리 라인을 `Field`와 `Vocab`으로 구성해놓은 걸 어느 세월에 character level로 바꾸나 싶어서 당황스러웠다. 
우선 찾다보니 torchtext 깃허브에 [이슈](https://github.com/pytorch/text/issues/834)로 등록된 글을 찾았는데, contributor중 하나가 고맙게도 [gist](https://gist.github.com/akurniawan/30719686669dced49e7ced720329a616)로 코드를 작성해주었다. 물론 NMT 데이터 셋에 대한 것이고, 잘 작동하지 않아서 좀 고쳐야 하지만, 최소한 시작점 위에는 올라선 셈.

## WikiText2에서 batch_size로 만드는 방법

`BPTTIterator`를 이용해서 iterator를 만들어보았는데, 분명 batch size 옵션을 넣었는데도 불구하고 `[1, seq_len]`의 tensor를 반환한다. 뭔가 이상하다 싶어서 알아봤는데, `bptt_len`옵션을 넣어줘야 batch로 반환하는 것으로 보인다. 아래와 같이 작성하면 잘 작동한다.

<script src="https://gist.github.com/InhyeokYoo/827545227b081452cd2345010e23aff8.js"></script>

## LM의 전처리 과정

character를 embedding 하므로, 26개의 alphabet만 하면 되는가 싶다가도, space라던가, apostrophe, puncation, capital 등은 어떻게 처리하나 궁금해졌다. 또한, ikiText2는 영어 외에도 일본어같은 다양한 언어가 들어있다. 따라서 이를 적절하게 처리할 방법이 필요하다.

## Filter map size

논문에 보면 Jozefowicz et al. (2016)의 CNN에서 사이즈를 반토막 낸다고 되어 있는데(4096 -> 2048), filter map의 사이즈가 정확하게 안 나와있다.

그러나 [이기창님의 자료](https://github.com/ratsgo/embedding/blob/master/models/bilm/training.py#L114)를 보면, filter size가 나와있으므로, 이를 활용하면 될듯 싶다.

## CNN parallel

ELMo 구조를 보면 여러 filter map size에 대해 convolution 연산을 하기 때문에, 이를 병렬로 처리하여 loop 구조를 탈피해야만 했다.
그러나 여러방면으로 노력해도 이에 대한 자료를 찾을 수는 없었다. 각 in_channels에 대해 convolution 연산을 처리할 순 있으나, 우리의 in_channels는 embedding dimension이므로 이 조차도 실패.

그리고 애초에 [AllenAI](https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L410)조차도 iterative 돌리고 있으므로 이대로 issue closed.

# BiLM

전반적으로 내용을 정리하면, CNN을 통해 Char embedding의 결과인 2048 (halved from 4096) 차원의 word embeddings에 대해, $L=2$인 biLSTM을 통과하고, 첫번째 레이어부터 2번째 레이어까지의 residual connection 하나와 LSTM 사이에 projection layer 하나가 있다.

## Residual connection

본문에는 *add a residual connection between LSTM layers*라고 되어 있는데, 정확히 어떻게 되는건지 모르겠다. 첫 번째 LSTM의 시작(input)에서 두 번째 LSTM의 시작(input)으로 residual connection을 연결한다는 것인지, 첫 번째 LSTM의 결과(output)와 두 번째 LSTM의 결과(output)을 연결한다는 것인지 헷갈린다. 

딱히 중요한 것은 아니므로, 둘 다 시도해서 성능을 비교하는 것도 좋아보인다.

