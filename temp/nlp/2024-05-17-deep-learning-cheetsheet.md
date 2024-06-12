---
title:  "Deep Learning Cheet Sheet"
toc: true
toc_sticky: true
categories:
  - Deep Learning
use_math: true
last_modified_at: 2024-05-28
---

## 들어가며

https://towardsdatascience.com/recent-advances-for-a-better-understanding-of-deep-learning-part-i-5ce34d1cc914

https://www.jeremyjordan.me/nn-learning-rate/

https://arxiv.org/pdf/1908.06477


## Batch size와 learning rate 간의 관계




## BERT fine-tuning 시 baseline

https://chloelab.tistory.com/33

BERT fine-tuning 시 학습 초기에는 gradient vanishing과 학습 후기에는 generalization으로 인해 불안정성이 발생하고, 이로 인해 fine-tuning을 어렵게 만든다.
이를 바탕으로 흔히 사용하는 작은 데이터셋으로 transformer 기반 MLM(Masked language model)을 fine-tuning하는 가이드라인을 알아보자.

- 작은 learning rate와 bias correction을 통해 학습 초기의 gradient vanishing을 방지
- 학습 iteration을 늘려 training loss가 거의 0이 될 때 까지 훈련

이를 토대로 논문에서 구체적으로 제시하는 scheme은 다음과 같다:
1. Bias-correction이 있는 ADAM Optimizer으로, learning rate를 2e-5로 설정
2. Epoch은 20으로
3. Linear scheduler with warm up (10%)를 사용한다.
4. 그 외 hyper parameter는 위에서 실험한 것과 동일하게 진행한다.

TODO: Hparam 사진추가




## 계단 형태의 loss

https://github.com/yuvalkirstain/PickScore/issues/19

## Scheduler

https://kashikarparth.github.io/posts/important-techniques-for-deep-learning

## gradient norm의 변화

TODO: Gradient norm의 정의

학습이 원할하게 진행되는 경우 gradient norm은 줄어들어야 한다.


Neural nets have thousand or millions of parameters so you're solving an optimization problem in a very high dimensional space. Thus the likelihood that you will exactly solve the minimization problem at hand is extremely small. When we say that the network's performance "has converged," we don't generally mean, in the context of neural networks, that the exactly optimal weights have been found. Rather, we mean that the network parameters are in some small neighborhood around a locally optimal solution.

https://stats.stackexchange.com/questions/410440/what-does-it-mean-when-the-global-gradient-norm-keeps-decreasing-while-loss-has

## Convergence

'네트워크가 수렴'하는 경우 'optimal weight값을 발견했다'기보다는 'locally optimal solution 근처에 있다'는 뜻이다.

{: .align-center}{: width="300"}
