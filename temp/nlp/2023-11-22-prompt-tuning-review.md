---
title:  "The Power of Scale for Parameter-Efficient Prompt Tuning (Prompt-tuning) review"
toc: true
toc_sticky: true
permalink: /project/nlp/prompt-tuning-review/
categories:
  - NLP
  - Paper Review
tags:
  - Large Language Model
  - PEFT
use_math: true
last_modified_at: 2023-11-22
---

## Introduction

간략한 개요와 참고 리소스 등을 적는다.

ICLR 2022
https://arxiv.org/pdf/2110.04366.pdf

### Summary

전체 내용을 요약한다

- 문제는 뭐였고
- 이렇게 해서
- 저렇게 해결한다
- 결론은 어떻게 나온다.

## Challenges

널리 알려져있다시피 일반적으로는 PLM의 파라미터 전부를 fine-tune한다 (full fine-tuning). 
그러나 이러한 방법을 큰 모델에 적용하기엔 부담이 심하다.
최근 좋은 성능은 유지한채로 작은 양의 (추가) 파라미터만을 fine-tuning하는 parameter-efficient transfer learning method가 제안됨.
성능은 효과적이나 왜 잘되는지에 대한 연구는 아직 적은 상황임.


## Contributions

SOTA parameter-efficient transfer learning method의 디자인을 분석하고 이들 사이의 연결점을 보여주는 통합된 프레임워크를 제안.
이러한 방법론들을 PLM의 특정 hidden state를 변경하는 것으로 재구성하며, 변경을 계산하는 함수와 이를 적용할 위치와 같이 다양한 방법이 달라지는 일련의 디자인 차원을 정의한다.

## Related Work

내가 모르는 연구 배경을 적는다

## Method

본 논문에서는 PEFT/PETL 방법론들 사이의 갭을 연결하는 역할을 한다 (3.BRIDGING THE GAP – A UNIFIED VIEW).
이를 위해 먼저 prefix tuning의 equivalent form을 유도하여 adapter와의 접점을 찾는다.
그 후 PEFT/PETL에 대한 unified framework을 제안한다.

### A CLOSER LOOK AT PREFIX TUNING



## Experiment

간략하게 실험에 대한 overview를 적는다

### Detail-Experiment

자세한 프로세스를 적는다.
제목은 논문에서 따온다.

## Conclusion

논문의 결론과 개인적인 소감, 아쉬운 점, 응용법 등을 정리한다.

![Fig.1-add-caption-here]({{site.url}}{{site.baseurl}}/assets/posts/CATEGORY/POST-NAME-Fig.1.png){: .align-center}{: width="600"}

![Caption](URL){: .align-center}{: width="600"}

