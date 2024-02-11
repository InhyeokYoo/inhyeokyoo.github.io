---
title:  "CLM 학습하기"
toc: true
toc_sticky: true
categories:
  - NLP
  - Language Modeling
tags:
  - Causal Language Model
use_math: true
last_modified_at: 2023-05-23
---

- max lenght보다 큰건 자름
- 하지만 일부 시퀀스는 이보다 크거나 작을 것임
    - 즉, padding or trunct 필요
    - 리소스가 커짐
- 여러 샘플을 토큰화한 후 EOS 토큰으로 연결해서 긴 시퀀스를 만듬
    - 이후 이를 동일한 크기의 chunk로 나눔
- PAD 토큰을 EOS 토큰으로 변경하는 경우도 있음

Q: 이 경우 EOS 토큰 이후로 문장이 계속 이어지게끔 모델이 학습될텐데 이게 별 문제 없을까?
  - Q: inference가 끝났다는 걸 모델이 어떻게 알 수 있을까?
Q: GPT1의 경우 UNK외에는 speical token이 없음. fine-tuning은 어떻게 하고 어떤 식으로 모델이 inference를 종료하지??
Q: GPT2의 경우 PAD=EOS. 패딩은 마스크 처리가 될텐데 이러면 학습 시 EOS 토큰을 볼 수 없지 않나?
{'bos_token': '<|endoftext|>',
  'eos_token': '<|endoftext|>',
  'unk_token': '<|endoftext|>'}

However, I don't think we plan on adding these tokens automatically when tokenizing an input string because the main use case for GPT2 is open-domain text generation where these tokens should not be added.
I agree that they could /should be added for fine-tuning.
