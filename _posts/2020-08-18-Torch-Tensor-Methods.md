---
title:  "PyTorch tensor의 유용한 메소드에 대해 알아보자"
excerpt: "PyTorch tensor 정리"
toc: true
toc_sticky: true

categories:
  - PyTorch

use_math: true
last_modified_at: 2020-08-18
---

## Introduction

PyTorch을 사용하다 보면 tensor의 유용한 method가 있다.
이를 정리하여 참고 용으로 남겨보자

## scatter_

`torch.Tensor.scatter_(dim, index, src)`는 `src` 텐서의 모든 값을 `index`값을 통해 `self`의 텐서로 변환한다. 즉, `self` tensor의 dim에 대해, index에 위치한 값들을 src의 값들로 변환한다는 뜻이다.

예시를 보자.

```python
>>> x = torch.rand(2, 5)
>>> x
tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
        [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
>>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
        [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
        [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])
```

이 경우, `0`의 dimension에 대해, `self`(이 경우 [3, 5]짜리 zeros가 된다)의 `index` 위치에, 주어진 `[[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]`값을 넣게 된다. 그림으로 보면 좀 더 이해가 빠르다.


![image](https://user-images.githubusercontent.com/47516855/90478303-efd7cd80-e167-11ea-90c4-cab8b102dd08.png){: .align-center}{: width='600'}

`dim=0`이므로 빨간색에 대해 진행하며, 따라서 `[0.3992,  0.2908,  0.9044,  0.4850,  0.6004]`에 한번,
`[ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]`에 한 번 진행하게 된다.
첫 번째 index는 `[0, 1, 2, 0, 0]`이므로, `src`의 첫 번째 원소인 `[0.3992,  0.2908,  0.9044,  0.4850,  0.6004]`를 `torch.zeros(3, 5)`의 `[0, 1, 2, 0, 0]`에 집어넣게 된다. 두 번째는 적혀있지 않지만 같은 논리로 진행되는 것을 파악할 수 있다. 이는 `torch.Tensor.gather`연산의 정반대이기도 하다.


