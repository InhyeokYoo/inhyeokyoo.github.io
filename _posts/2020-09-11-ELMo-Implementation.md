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

# Character CNN의 embedding은 어떻게 하는가?

