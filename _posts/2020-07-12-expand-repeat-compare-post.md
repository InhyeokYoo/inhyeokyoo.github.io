---
title:  "torch.expand() vs. torch.repeat()"
excerpt: "torch.expand()와 torch.repeat()를 비교해보자"
toc: true
toc_sticky: true

categories:
  - PyTorch
tags:
  - torch.repeat
  - torch.expand
use_math: true
last_modified_at: 2020-07-12
---

네트워크에 feed하다보면 차원을 만져줘야 되는 일이 있다. 그때 사용하는 것이 `torch.repeat()`인데, `torch.expand()`랑 구체적으로 어떠한 차이가 있는지 모르겠어서 한번 찾아봤다.

## torch.repeat(\*sizes)

**특정 텐서**의 _sizes_ 차원의 데이터를 **반복**한다. 예시를 통해 이해해보자.

```python
>>> x = torch.tensor([1, 2, 3])
>>> x.repeat(4, 2)
tensor([[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]])
>>> x.repeat(4, 2, 1).size()
torch.Size([4, 2, 3])
```

x는 `[1, 2, 3]`으로, 이를 `dim=0`으로 4, `dim=1`로 2만큼 반복하니, `[4, 6]`의 차원이 나오는 것을 확인할 수 있다. 1-D 텐서의 경우, `[n]`이 아닌 `[1, n]`으로 간주한다. `torch.repeat(*sizes)`의 경우 텐서를 copy한다.

## torch.expand(\*sizes)

마찬가지로, 특정 텐서를 반복하여 생성하지만, 개수가 1인 차원에만 적용할 수 있다.

```python
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
```

`[3, 1]`인 x를 차원의 개수가 1인 `dim=1`에 대해 4번 반복한 모습이다. 만약, `x.expand(3, 4)`에서 첫번째 차원(차원이 1이 아닌)이 3이 아니면 에러가 발생한다. 이는 3-D 텐서에도 마찬가지로 적용할 수 있다. 아래에서 -1 옵션은 차원을 유지하겠다는 의미이다.

```python
a = torch.rand(1, 1, 3)
print(a.size()) # [1, 1, 3]
b = a.expand(4, -1, -1)
print(b.size()) # [4, 1, 3]
```

`torch.expand(*sizes)`의 경우 메모리를 참조하기 때문에, 원본을 참조하게 된다.

```python
a = torch.rand(1, 1, 3)
print(a.size())
b = a.expand(4, -1, -1)
c = a.repeat(4, 1, 1)
print(b.size(), c.size())

a[0, 0 , 0] = 0
print(b, c)
```

```
tensor([[[0.0000, 0.9028, 0.3184]],

        [[0.0000, 0.9028, 0.3184]],

        [[0.0000, 0.9028, 0.3184]],

        [[0.0000, 0.9028, 0.3184]]]) tensor([[[0.9590, 0.9028, 0.3184]],

        [[0.9590, 0.9028, 0.3184]],

        [[0.9590, 0.9028, 0.3184]],

        [[0.9590, 0.9028, 0.3184]]])
```

b의 경우는 원본 a의 변경을 참조하는 것을 확인할 수 있다.