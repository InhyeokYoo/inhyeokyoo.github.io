---
title:  "NLP 모델 distillation하기"
toc: true
toc_sticky: true
categories:
  - NLP
tags:
  - Distillation
use_math: true
last_modified_at: 2023-05-07
---

## 들어가며


## Evaluation

보통은 latency와 메모리를 줄이기 위해 작은 모델을 고른다.
Y. Kim and H. Awadella의 FastFormers에 따르면 teacher와 student가 동일한 종류의 모델일 때 distillation이 잘된다고 한다.
이는 모델 종류가 다를 때 출력 임베딩 공간이 달라져 student가 teacher를 모방하는데 방해가 되기 때문이다.


{: .align-center}{: width="300"}
