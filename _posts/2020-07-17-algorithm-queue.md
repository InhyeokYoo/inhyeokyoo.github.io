---
title:  "백준 Queue 풀이"
excerpt: "백준 Queue 풀이 모음집"
toc: true
toc_sticky: true

categories:
  - Algorithm
tags:
  - Queue
  - BOJ
  - Python
use_math: true
last_modified_at: 2020-07-17
---

## Introduction

백준에서 Queue 풀이 만을 모아놓았다. 예전에 푼 것은 수정하기가 귀찮아서 그냥 올렸는데 앞으로 푸는 것은 풀이 과정도 정리해서 올릴 예정이다. 

사용언어는 Python이다. 

오른쪽 TOC를 통해 바로가기를 해보자.

**<h3> 알고리즘 바로가기 </h3>**
- [스택](https://inhyeokyoo.github.io/algorithm/Algorithm-Stack/)
- **[큐](https://inhyeokyoo.github.io/algorithm/algorithm-queue/)**
- [순환 큐](https://inhyeokyoo.github.io/algorithm/Algorithm-CircularQueue/)

## 문제 모음

### 10845 큐

- [문제보기](https://www.acmicpc.net/problem/10845)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Queue/10845.py)
- 풀이 과정:
- class로 직접 구현하는 방법과 `collections.deque`를 활용하는 방법이 있다.
- deque를 활용하는 방법은 아래와 같다.

```python
import sys
from collections import deque

q = deque()
num = int(sys.stdin.readline().rstrip())

for i in range(num):
    order = sys.stdin.readline().rstrip().split()

    if order[0] == 'push':
        q.append(int(order[1]))
    elif order[0] == 'front':
        if len(q) != 0:
            print(q[0])
        else:
            print(-1)
    elif order[0] == 'back':
        if len(q) != 0:
            print(q[-1])
        else:
            print(-1)
    elif order[0] == 'size':
        print(len(q))
    elif order[0] == 'empty':
        if len(q) == 0:
            print(1)
        else: print(0)
    elif order[0] == 'pop':
        if len(q) != 0:
            print(q.popleft())
        else:
            print(-1)
```
