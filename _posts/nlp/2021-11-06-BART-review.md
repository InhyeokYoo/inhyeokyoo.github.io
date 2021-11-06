---
title:  "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
toc: true
toc_sticky: true
permalink: /project/nlp/BART-review/
categories:
  - NLP
  - Paper Review
tags:
  - Language Modeling
use_math: true
last_modified_at: 2021-11-06
---

## 들어가며


- [원문 보러가기](https://arxiv.org/pdf/1910.13461.pdf)
- [XLNet repository 보러가기](https://github.com/zihangdai/xlnet)


- BERT 이후 denoising auto encoder를 사용하는 것이 표준이 됨.
  - 최근엔 이를 응용한 다양한 연구가 존재
  - 마스크 토큰의 분포
  - 마스크 토큰 순서
  - available context for replacing masked tokens
- 그러나 이러한 방법은 특정한 종류의 end task에만 집중 (generation, span prediction)
- BART (Bidirectional and Auto-Regressive Transformers)는 DAE에 seq2seq을 결합하여 다양한 종류의 end task에 적용가능하게끔 함.
- Pre-training은 다음의 두 스텝
1. 임의의 noise function을 통해 텍스트를 오염
2. seq2seq을 통해 복원
- BERT, GPT, 등과 같이 최근 유행하는 pre-training scheme

- 최대장점은 noising flexibility
  - 임의의 변환이 원본 문서에 적용가능
  - 다양한 noising approach를 테스트
    - 랜덤 셔플링
    - novel in-filling scheme (임의의 길이의 span을 하나의 mask token으로 치환)
      - 이는 BERT의 일반화로 볼수있음
- 생성 task에 유용하지만 comprehension에도 잘 동작
  - GLUE, SQuAD에서 RoBERTa와 비슷
  - abstractive dialogue, question answering, and summarization tasks에서 SOTA
- Fine-tuning의 새로운 메타를 열었음
  - 트랜스포머 위에 BART를 stack하여 NMT를 진행
  - 트랜스포머는 외국어를 오염된 영어로 번역하게끔 학습되고, BART를 pre-trained target-side language model로 사용

## 2. Model

implemented as a sequence-to-sequence model with a bidirectional encoder over corrupted text and a left-to-right autoregressive decoder.

### 2.1 Architecture

- 기존 Transformer와 똑같지만, GPT를 따라서 GELU를 이용. Initial parameter는 $\mathcal N (0, 0.02)$를 채용.
- 모델은 인코더 디코더 6/6, large에선 12/12
- BERT와 거의 똑같지만 아래의 차이점이 있음
  - 인코더의 마지막 레이어에 대해 각 디코더가 추가적으로 cross-attention을 수행 (Transformer처럼)
  - BERT에선 word prediction 이전에 linear layer를 추가하였는데, 여기선 없음
- 대략 BERT에 비해 10% 더 많은 파라미터를 사용

### 2.2 Pre-training BART

BART의 학습은 문서를 오염시킨 뒤 이를 복원하는 reconstruction loss를 최적화함으로 이루어진다. 이는 디코더의 인풋과 아웃풋에 대한 cross entropy가 된다. 기존의 DAE를 이용한 모델들은 특정 noising scheme에 맞춤형(tailored) 모델인 것에 반해 BART는 어떠한 형태의 noising function을 허용한다. 아주 극단적으로 모든 source input을 없앨 경우 BART는 일반적인 language model이 된다.

논문에서 사용한 noise function은 아래에 기술되어 있다.









{: .notice--info}
{: .align-center}{: width="500"}
