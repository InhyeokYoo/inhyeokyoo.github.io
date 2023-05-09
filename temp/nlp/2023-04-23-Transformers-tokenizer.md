---
title:  "Transformers: models 소개"
toc: true
toc_sticky: true
permalink: /project/nlp/transformers/tokenizer/
categories:
  - NLP
  - transformers
  - Huggingface
tags:
  - tokenizer
use_math: true
last_modified_at: 2023-04-23
---

## 들어가며

`as_target_tokenizer()` 메소드는 디코더 입력에서 사용하는 컨텍스트 매니저로, 디코더 입력 시 필요한 speical token을 만드는데 사용된다.
with문 하에서는 tokenizer가 디코더 입력용으로 사용된다

```py
    input_encodings = tokenizer(example_batch["dialogue"], max_length=1024,
                                truncation=True)
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["summary"], max_length=128,
                                     truncation=True)
```


{: .align-center}{: width="300"}
