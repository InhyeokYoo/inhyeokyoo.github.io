---
title:  "Metric Learning (작성 중)"
excerpt: "Metric Learning을 알아보자"
toc: true
toc_sticky: true
categories:
  - CV
tags:
  - Metric Learning
use_math: true
last_modified_at: 2020-12-31
---

회사에서 사전연구 진행하며 metric learning을 공부할 일이 생겼는데, 연구도 진행할 겸 포스트 글로 정리를 해보았다. 여러가지 metric learning이 있는 것으로 알고 있는데, 본 포스트에서는 딥러닝만을 한정하여 다룬다.

# Metric Learning이란?

Metric learning이란
 목적은 object를 embedded space로 mapping하는 representation function을 찾는 것
즉 feature가 discriminative하게 만드는 것

embedded space에서 거리는 object의 유사도를 잘 보존할 수 있어야 함
비슷한 오브젝트는 가까운 거리에 위치하며, 반대의 경우 먼 거리에 위치

다양한 loss function을 통해 metric learning이 가능


# 참고

- [Digging Deeper into Metric Learning with Loss Functions](https://towardsdatascience.com/metric-learning-loss-functions-5b67b3da99a5)
- [Deep Metric Learning with Angular Loss (Want et al. 2017)](https://arxiv.org/pdf/1708.01682.pdf)