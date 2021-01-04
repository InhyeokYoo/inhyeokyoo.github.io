---
title:  "pytorch로 BERT 구현하기 (작성 중)"
excerpt: "BERT 구현 및 issue 정리"
toc: true
toc_sticky: true
permalink: /project/nlp/bert-issue/
categories:
  - NLP
  - PyTorch
tags:

use_math: true
last_modified_at: 2020-12-06
---

[이전 시간](/project/nlp/bert-review/)에는 BERT에 대해 공부해보았다. 이번에는 이를 구현해보도록 하자.

# Pre-processing

BERT는 크게 pre-train과 fine-tuning 두 가지의 task를 하게 된다. 이번 장에서는 데이터를 load하여 DataLoader를 만드는 것을 포함하여 각 task에서 필요로 하는 pre-processing을 다뤄보자.

## Pre-train

Pre-train과정에서는 masked language model과 next sentence prediction을 수행한다. 구체적으로 필요한 요구사항은 다음과 같이 정리할 수 있을 것 같다.
- DataLoader
    - *`torchtext.data.Dataset` vs. `torch.utils.data.Dataset`*
- 학습 데이터에 대해 WordPiece 모델을 통해 tokenizing 하는 기능
    - *직접 만들기는 그렇고 어디선가 가져와야 함*
- 학습 데이터에 대해 `Vocab`으로 단어를 저장하는 기능
    - `torchtext.data.Field`쓰면 됨
- \<CLS>, \<SEP>, \<MASK> special token 추가
    - `torchtext.data.Field`쓰면 됨
- **각 task에 맞는 기능 추가하기**
    - NLM: \<MASK> 토큰 씌우는 기능
    - NSP: 문장 섞어주는 기능. 이러면 *BPTTIterator*를 사용할 필요가 없음

### `torchtext.data.Dataset` vs. `torch.utils.data.Dataset`

우선 데이터를 불러와야 한다. BERT는 BooksCorpus와 wikipedia데이터를 통해 학습한다.

> For the pre-training corpus we use the BooksCorpus (800M  words)  (Zhu  et  al.,2015) and English  Wikipedia (2,500M  words).

BooksCorpus는 [허깅페이스](https://huggingface.co/datasets/bookcorpus)를 통해 다운받을 수 있다.

```shell
!wget https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2
!tar -xvf '/content/drive/MyDrive/Colab Notebooks/datasets/bookcorpus.tar.bz2'
```

데이터를 불러오는 `Dataset`선택지는 크게 두 개가 있다. 하나는 `torchtext.data.Dataset`를 쓰는 것이고, 나머지 하나는 `torch.utils.data.Dataset`를 쓰는 것이다.

`torchtext.data.Dataset`는 parameter로 *examples*와 *fields*를 받고, 자동적으로 vocab 등을 생성해준다는 장점이 있다. 그러나 `torchtext.data.Example`을 만들어줘야 한다. 이는 `torch.utils.data.Dataset`의 자식 클래스이다.

그러나 `torchtext.data.Field`는 곧 deprecation되어 없어질 예정이고, `torchtext.data.Dataset`는 `torch.utils.data.Dataset`와 호환되지 않으므로 `torch.utils.data.Dataset`을 사용하는게 더 좋아보인다. 이는 다음 [torchtext 레포에 남겨진 issue](https://github.com/pytorch/text/issues/936)와 [패치노트](https://github.com/pytorch/text/releases)에서도 확인할 수 있다.

> Several components and functionals were unclear and difficult to adopt. For example, the Field class coupled tokenization, vocabularies, splitting, batching and sampling, padding, and numericalization all together, and was opaque and confusing to users. We determined that these components should be divided into separate orthogonal building blocks. **For example, it was difficult to use HuggingFace's tokenizers with the Field class (issue #609)**. Modular pipeline components would allow a third party tokenizer to be swapped into the pipeline easily.
...
torchtext’s datasets were incompatible with DataLoader and Sampler in torch.utils.data, or even duplicated that code (e.g. torchtext.data.Iterator, torchtext.data.Batch). Basic inconsistencies confused users. For example, many struggled to fix the data order while using Iterator (issue #828), whereas with DataLoader, users can simply set shuffle=False to fix the data order.

여기서 문제는 `torchtext.data.Field`, `torchtext.data.Example`도 같이 없어지기 때문에 이를 대체할 코드가 필요하다는 점이다. 앞선 torchtext repo에는 이와 관련된 튜토리얼 자료를 모두 대체해놨다.


## Masked language model

Issues:
- PAD를 통해 batch로 묶어야 함
    - sentenceA, B가 나뉘어야 하는데, 얘네들은 각 각 256씩으로 해야하나?
    - PAD는 누가하나? collate_fn?
        - 누가 맡지? Dataset? DataLoader?
    - torch를 보면 동시에 학습하지 않는 것처럼 보이는데?