---
title:  "Accelerate 사용법"
toc: true
toc_sticky: true
categories:
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

## 특정 GPU만 사용하기

CLI에서 accelerate를 사용할 때 `CUDA_VISIBLE_DEVICES` 옵션을 함께 주면 된다.
예를 들면 다음과 같이 사용할 수 있다.

```sh
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 train.py
```

{: .align-center}{: width="300"}
