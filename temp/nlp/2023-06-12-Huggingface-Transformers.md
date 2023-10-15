---
title:  "🤗 Huggingface: Transformers"
toc: true
toc_sticky: true
permalink: /project/nlp/Hugging-Face/Transformers
categories:
  - NLP
  - Hugging Face
tags:
  - Transformers
use_math: true
last_modified_at: 2023-06-12
---

## 들어가며

## `pipeline`

Transformers에서는 데이터를 파인튜닝된 모델의 결과로 사용하기 위해 모든 단계를 추상화하는 `pipeline`을 사용한다.

이를 위해서는 `pipeline()`함수를 호출하면서 태스크 이름을 전달하여 객체를 생성한다.
텍스트 분류에 대한 예제를 살펴보자

```py
from transformers import pipeline

classifier = pipeline("text-classification")
```


{: .align-center}{: width="300"}
