---
title:  "Accelerate 사용법"
toc: true
toc_sticky: true
categories:
  - NLP
  - Huggingface
  - transformers
  - accelerate
tags:
  - Multi-GPU
use_math: true
last_modified_at: 2024-05-08
---

## 들어가며

## `gather_for_metrics`

- `DataLoader`에서 `shuffle=True`하지 않은 이상 순서는 보존됨
- ddp 특성 상 마지막 배치에 모자란 샘플을 추가하게 되는데, `accelerator.gather`을 사용할 경우 이를 확인해야하는 반면 `gather_for_metrics`은 이를 신경쓰지 않아도 됨.

{: .align-center}{: width="300"}
