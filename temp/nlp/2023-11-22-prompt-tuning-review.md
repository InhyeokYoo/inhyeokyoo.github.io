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


### Summary

전체 내용을 요약한다

- 문제는 뭐였고
- 이렇게 해서
- 저렇게 해결한다
- 결론은 어떻게 나온다.

## Challenges

## Contributions

## Related Work

내가 모르는 연구 배경을 적는다

## Method

실험 옵션은 다음과 같음:
1. Default
  - LM-adaptation을 추가 데이터 100K step에 대해 적용한 버전.
  - Class label 적용 # TODO: Section 3.2 확인
  - Prompt의 길이는 100 token임.
    - Prefix-truning의 10 token보다는 훨씬 길지만, task-specific paramter는 훨씬 적음 (input layer만 학습 진행).
  - T5와 마찬가지로 SuperGlue를 text-to-text 형태로 변형하여 사용하되, input 앞에 붙는 prompt에 SuperGLUE task는 제외하여 어떠한 형태의 데이터인지는 알 수 없게 만듬
2. Standard models
  - SuperGLUE로 학습된 public T5.1.1 checkpoints 사용
  - Hparms: lr=0.001, Adafactor optimizer
    1. Model tuning:
      - 제대로된 비교를 위해 prompt tuning setup으로 **각 task를 따로** 학습
      - Batch size search를 진행하였고, 한 batch 당 $2^16$ 토큰으로 학습
    2. Model tuning (Multi-task)
      - 성능향상을 위해 T5의 **multi-task** 세팅으로 학습
      - T5.1.1로 T5의 multi-task 세팅을 그대로 따라할 순 없으므로 T5논문을 따라 batch 당 $2^20$ 토큰과 DRP 데이터를 추가하여 학습
      - 이 경우엔 태스크의 이름이 prefix로 들어가게 됨


### Dataset

T5와 마찬가지로 SuperGlue를 text-to-text 형태로 변형하여 사용하되, input 앞에 붙는 prompt에 SuperGLUE task는 제외하여 어떠한 형태의 데이터인지는 알 수 없게 만들었다.

### Hyper Paramters

prompt: 30,000 step + CE Loss + learning rate 0.3 + batch size 32 + Adafactor optimizer (weight decay $1e-5$, $\beta _2$ decay 0.8)

Early stopping 적용

TODO: 자세한 내용은 appendix A에있음

## Experiment

### Ablation Study: Prompt Length

Prompt의 길이에 따른 성능 변화를 보기 위해 $\{1, 5, 20, 100, 150\}$ 중 하나를 골라 실험하였다.

아무리 못해도 하나 이상의 prompt를 고르는 것이 좋았으나, XXL 사이즈의 경우 특별히 하나의 토큰으로도 준수한 성능을 보여주었다.
이는 큰 모델일수록 prompt 신호에 덜 영향을 받음을 보여준다. 

### Detail-Experiment

자세한 프로세스를 적는다.
제목은 논문에서 따온다.

## Conclusion

논문의 결론과 개인적인 소감, 아쉬운 점, 응용법 등을 정리한다.

![Fig.1-add-caption-here]({{site.url}}{{site.baseurl}}/assets/posts/CATEGORY/POST-NAME-Fig.1.png){: .align-center}{: width="600"}

![Caption](URL){: .align-center}{: width="600"}

