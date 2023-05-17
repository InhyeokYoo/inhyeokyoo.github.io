---
title:  "NLP-with-transformers: 개요"
toc: true
toc_sticky: true
permalink: /project/nlp/transformers/Trainer
categories:
  - NLP
  - transformers
tags:
  - Trainer
  - Model
  - 
use_math: true
last_modified_at: 2023-03-26
---

TODO-list:
- Trainer
- TrainingArguments
- 

## TrainingArguments

`Trainer`는 fine-tuning 시 labels 이름의 열을 찾는다.
이는 `TrainingAurguments`의 `label_names`를 사용해 오버라이드가 가능하다.

## `pipeline`

사용법:

```py
from transformers import pipeline

pipe = pipeline("fill-mask", model="bert-base-uncased")
pipe(example["text"], all_labels, multi_label=True)
```

task 종류:
- `fill-mask`: MLM에 사용. `transformers.FillMaskPipeline` 객체를 반환한다.
- `zero-shot-classification`: NLI에 대해 `ModelForSequenceClassification`을 이용하여 학습한 zero-shot classification으로, `text-classification` 파이프라인과 동일하나 클래스의 갯수를 미리 지정해주느냐 마느냐의 차이가 있다. `zero-shot-classification`은 제로샷에 대해 학습한 것이기 때문에 훨씬 유연하지만 더 느리다는 단점이 있다.



{: .align-center}{: width="300"}