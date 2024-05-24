---
title:  "Accelerate 사용법"
toc: true
toc_sticky: true
categories:
  - NLP
use_math: true
last_modified_at: 2024-05-17
---

## 들어가며

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
