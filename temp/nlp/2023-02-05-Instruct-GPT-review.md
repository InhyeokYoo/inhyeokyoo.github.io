---
title:  "InstructGPT: Training language models to follow instructions with human feedback review"
toc: true
toc_sticky: true
permalink: /project/nlp/InstructGPT/
categories:
  - NLP
  - Paper Review
tags:
  - Large Language Model
use_math: true
last_modified_at: 2023-03-26
---

## Introduction

언어모델 = Multitask learner


## Related Work

InstructGPT의 사전연구는 크게 RLHF와 instruct를 통한 language model학습, 

### Research on alignment and learning from human feedback

InstructGPT는 RLHF와 같이 사람의 의도에 따라 모델을 정렬시키는 기술을 기반으로 한다.
원래는 시뮬레이션 환경이나 아타리 게임에서의 간단한 로봇을 학습시키는 정도였지만, 최근들어 언어모델을 fine-tuning하여 텍스트를 요약하는 방법론에 적용된바 있으며, InstructGPT 또한 대화(dialogue), 번역, semantic parsing, 이야기 생성(story generation), 리뷰 생성(review generation), evidence extraction 등의 연구에 영향을 받았다고 한다.

[MemPrompt (Madaan et al. (2022))](https://arxiv.org/pdf/2201.06009.pdf)의

동음이의어(同音異義語; homonym)'


## Methodology and Experiment 

### RLHF

1. Pretraining a language model (LM),
2. gathering data and training a reward model, and
3. fine-tuning the LM with reinforcement learning.

For example, suppose that we wanted to use reinforcement learning to train a robot to clean a table or
scramble an egg. It’s not clear how to construct a suitable reward function, which will need to be a
function of the robot’s sensors. We could try to design a simple reward function that approximately
captures the intended behavior, but this will often result in behavior that optimizes our reward
function without actually satisfying our preferences.

Thus, in traditional Reinforcement Learning, the reward function is written by hand. In RLHF, the reward function is learned. Once you have the reward function, the next step is learning a policy to maximize reward.

### PPO

### High-level methodolog

InstructGPT를 학습시키기 위한 준비물은 pretrained language model, 아웃풋 정렬하는데 사용되는 prompts distribution, 훈련된 labler 이 세가지이다.
이를 통해 다음과 같은 단계를 거쳐 학습시킨다.

**1. Collect demonstration data, and train a supervised policy**

TODO: Demonstration data?

Input prompt distribution에 대해 원하는 행동이 무엇인지 labler들이 검증한다.
이후 GPT-3를 fine-tuning한다.

**2. Collect comparison data, and train a reward model**

모델의 아웃풋들간 비교를 위해 데이터를 수집하고 labler로 하여금 인풋에 대해 어떤 아웃풋을 선호하는지 고르도록 한다.
그후 reward model를 학습하여 사람이 선호하는 아웃풋을 예측하도록 한다.

**3. PPO를 이용, reward model에 대해 policy를 최적화 (Optimize a policy against the reward model using PPO)**

Reward model의 아웃풋으로 스칼라값을 사용하고 PPO를 이용하여 이 reward model을 최적화하기 위해 supervised policy를 fine-tune한다.

이때 2단계와 3단계는 계속해서 반복될 수 있다.
현재 최적의 policy에 대한 비교 데이터를 수집한 후 새로운 reward model과 policy를 학습하는데 쓸 수 있다.
실제로 실험에서 대부분의 비교 데이터는 supervised policy로 부터 수집되었으며, 몇 몇만 policy로부터 수집되었다.

demographic criteria: 인종, 종교 등을 기준으로 

### Dataset

(trained via supervised learning on a subset of our demonstration data) 초기 InstructGPT에 대해 OpenAPI로 제출된 프롬프트 위주였는데 

During training and evaluation, our alignment criteria may come into conflict: for example, when a user requests a potentially harmful response. During training we prioritize helpfulness to the user (not 7 doing so requires making some difficult design decisions that we leave to future work; see Section 5.4 for more discussion).
-> criteria가 충돌하는데, 이거 다 만족하기 어려우니까 helpfulness만 우선적으로 두고 학습

Output: completion

## Summary

{: .align-center}{: width="300"}

he maximum distance overhead

gradient checkpointing: https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs
