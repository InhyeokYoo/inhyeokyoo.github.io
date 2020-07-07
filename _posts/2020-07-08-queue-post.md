---
title:  "1. Data Structure - Queue"
excerpt: "CS 비전공자의 IT 기업 면접 뽀개기"
toc: true
toc_sticky: true

categories:
  - recruiting
tags:
  - CS
  - Queue
  - Data Structure
last_modified_at: 2020-07-08
---

# Queue

Stack과 비슷하지만, 먼저 들어간 자료가 먼저 나오는 선입선출(FIFO) 구조이다. 큐에 새로운 데이터가 들어오면 enqueue, 데이터가 삭제될 때는 dequeue가 된다.

Queue의 종류로는 Circular queue, Priority queue가 있다.

## ADT

- `front(head)`: 데이터를 deque할 수 있는 위치.
- `rear(tail)`: 데이터를 enque할 수 있는 위치
- `is_empty() -> bool` 
- `enqueue(data) -> None`: 삽입
- `dequeue() -> data`: 삭제
- `peak() -> data or 1`: 맨 위의 값을 출력

## Python 구현

파이썬에서는 `collections.deque`로 구현되어 있다. `Queue.queue`도 있지만, 멀티 쓰레드를 위한 환경이기 때문에 매우 느리기에 코딩 테스트에서는 권장하지 않는다.

다음은 List로 구현한 코드이다.

```python
class Queue(list):

    enqueue = list.append

    def dequeue(self):
        return self.pop(0)

    def is_empty(self):
        if not self:
            return True
        else:
            return False

    def peek(self):
        return self[0]
```

그러나 dequeue시에 `pop(0)`를 하면 O(N)의 시간복잡도를 갖아 상당히 느려진다. 반면 `collections.deque`의 경우 O(1)이 된다. 이는 doubly linked list로 이루어져 있기 때문이다.

다음은 linked list를 이용하여 구현한 모습이다.

```python
class Node:
    def __init__(self, data) -> None:
        self.data = data
        self.next = None


class Queue:
    def __init__(self) -> None:
        self.head = None # 선입
        self.tail = None # 후입

    def is_empty(self) -> bool:
        if not self.head:
            return True

        return False

    def enqueue(self, data):
        # Insert
        new_node = Node(data)

        if self.is_empty():
            self.head = new_node
            self.tail = new_node
            return
        # n-1 데이터의 다음 데이터가 n번째 데이터가 됨
        self.tail.next = new_node
        # 새로운 데이터는 LI가 됨
        self.tail = new_node

    def dequeue(self):
        if self.is_empty():
            return None

        ret_data = self.head.data # FIFO
        self.head = self.head.next # 그 다음 데이터가 FI가 됨
        return ret_data

    def peek(self):
        if self.is_empty():
            return None

        return self.head.data
```


## Reference

https://wayhome25.github.io/cs/2017/04/18/cs-21/

https://daimhada.tistory.com/107?category=820522

# Circular Queue

방금전에 보았듯 list로 queue를 구현할 경우 매우 느리다. 이는 데이터가 dequeue 될 경우 데이터의 idx가 한 칸씩 이동하기 때문이다. 이를 극복하는 방법으로는 Circular queue가 있다.

이는 배열의 특성인 제한된 길이에 맞게 고안된 것이다.

## Reference

https://daimhada.tistory.com/168?category=820522

https://hashcode.co.kr/questions/1830/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0%ED%81%90-%EC%99%80-%EC%8A%A4%ED%83%9D%EC%9D%98-%EC%8B%A4%EC%A0%9C-%EC%82%AC%EC%9A%A9%EC%98%88%EB%A5%BC-%EC%95%8C%EA%B3%A0%EC%8B%B6%EC%8A%B5%EB%8B%88%EB%8B%A4

