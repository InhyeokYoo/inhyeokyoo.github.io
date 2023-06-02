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

인자목록:
- return_overflowing_tokens (`bool`, *optional*, defaults to `False`): overflowing token을 반환할지에 대한 여부. 시퀀스 페어가 `truncation_strategy = longest_first` 나 `True`로 들어오게 될 경우 에러가 raise된다. 리턴값에는 `overflowing_tokens`과 `num_truncated_tokens`가 포함된다.

`as_target_tokenizer()` 메소드는 디코더 입력에서 사용하는 컨텍스트 매니저로, 디코더 입력 시 필요한 speical token을 만드는데 사용된다.
with문 하에서는 tokenizer가 디코더 입력용으로 사용된다

```py
    input_encodings = tokenizer(example_batch["dialogue"], max_length=1024,
                                truncation=True)
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["summary"], max_length=128,
                                     truncation=True)
```

tokenzier를 학습할 때에는 `train_new_from_iterator` 메소드를 사용.

```py
tokenizer.train_new_from_iterator(batch_iterator(), 
                                                  vocab_size=12500,
                                                  initial_alphabet=base_vocab)
```

{: .align-center}{: width="300"}
