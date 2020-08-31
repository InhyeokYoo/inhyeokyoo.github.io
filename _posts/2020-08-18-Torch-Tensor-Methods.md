---
title:  "PyTorch tensor method 정리"
excerpt: "PyTorch tensor의 유용한 메소드에 대해 알아보자"
toc: true
toc_sticky: true

categories:
  - PyTorch

use_math: true
last_modified_at: 2020-08-18
---

# Introduction

PyTorch을 사용하다 보면 tensor의 유용한 method가 있다.
이를 빠른 레퍼런스 겸 참고용으로 정리하였다.

# expand/repeat

[다음](https://inhyeokyoo.github.io/pytorch/Expand-Repeat-Compare/)을 참고.

# gather()

![](https://i.stack.imgur.com/nudGq.png){: .align-center}{: width='600'}

`torch.Tensor.gather(input, dim, index)`는 `dim`의 axis를 따라서 값을 모으는 역할을 한다.
따라서 `input` tensor와 `dim`을 제외한 다른 차원이 동일한 tensor를 `index`로 넣어줘야 한다. 백문이 불여일견이라고 직접 예시로 확인해보자.

<script src="https://gist.github.com/InhyeokYoo/a6e426cffdd694815b93f916e5970e43.js"></script>

`dim=0`이므로, 가장 처음의 차원에 대해서 gather을 실시한다. 나의 경우엔 `index` `[[[0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2]]]`를 넣었고, 이의 size는 [1, 5, 4]가 된다. 앞서 언급했듯 `dim=0`의 차원 `1`은 아무렇게나 넣어도 상관없지만, 나머지 `[5, 4]`는 반드시 `input`과 동일해야 한다.

이제 `index`를 보자. `index`는 `[[[0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2]]]`이다. 이를 해석해보면, gather의 결과로 나올 `[1, 5, 4]`짜리 tensor는 `input` tensor에서 이 index를 통해 가져오겠다는 뜻이다. 이에 따라 인덱스의 맨 처음 `[0, 1, 2, 2]`은, `[input[0][0][0], input[1][0][1], input[2][0][2], input[2][0][3]]`을 가져온다.

즉, `[0, 1, 2, 2]`는 `dim=0`의 index를 의미하고, dim2와 dim3은 `[0, 1, 2, 2]`의 위치에 따라 달라진다(e.g. `[0, 1, 2, 2]`는 `index`에서 `[[0][0], [0][1], [0][2], [0][3]]`의 위치에 자리잡고 있으므로, 이 값이 전달된다).


# scatter_

`torch.Tensor.scatter_(dim, index, src)`는 `src` 텐서의 모든 값을 `index`값을 통해 `self`의 텐서로 변환한다. 즉, `self` tensor의 `dim`에 대해, `index`에 위치한 값들을 src의 값들로 변환한다는 뜻이다.

예시를 보자.

<script src="https://gist.github.com/InhyeokYoo/742e26971301916b821f5afe59a891d3.js"></script>

이 경우, `0`의 dimension에 대해, `self`(이 경우 [3, 5]짜리 zeros가 된다)의 `index` 위치에, 주어진 `[[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]`값을 넣게 된다. 그림으로 보면 좀 더 이해가 빠르다.


![image](https://user-images.githubusercontent.com/47516855/90478303-efd7cd80-e167-11ea-90c4-cab8b102dd08.png){: .align-center}{: width='600'}

`dim=0`이므로 빨간색에 대해 진행하며, 따라서 `[0.3992,  0.2908,  0.9044,  0.4850,  0.6004]`에 한번,
`[ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]`에 한 번 진행하게 된다.
첫 번째 index는 `[0, 1, 2, 0, 0]`이므로, `src`의 첫 번째 원소인 `[0.3992,  0.2908,  0.9044,  0.4850,  0.6004]`를 `torch.zeros(3, 5)`의 `[0, 1, 2, 0, 0]`에 집어넣게 된다. 두 번째는 적혀있지 않지만 같은 논리로 진행되는 것을 파악할 수 있다. 이는 `torch.Tensor.gather`연산의 정반대이기도 하다.

또한, `scatter_()`는 autograd를 지원한다. 다만, input tensor에 대해서만 gradient를 계산할 수 있고, index에 대해서는 불가능하다. 아래 예시를 보자.

<script src="https://gist.github.com/InhyeokYoo/f18f867ed542922c15ab909328cf45a6.js"></script>

여기서 `scatter_()`의 경우 in-place 연산이므로 `clone` 한 것을 유의하자.


