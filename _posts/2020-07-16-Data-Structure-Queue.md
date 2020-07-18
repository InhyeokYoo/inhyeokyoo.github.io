---
title:  "CS 비전공자의 IT 기업 면접 뽀개기 - (1) Data Structure - Queue"
excerpt: "Queue에 대해 알아보자"
toc: true
toc_sticky: true

categories:
  - IT-Interview
tags:
  - CS
  - Queue
  - Circular queue
  - Priority queue
  - Data Structure
  - Python
last_modified_at: 2020-07-16
---

# IT 면접준비 1. Data Structure - Queue
- [Array, LinkedList](https://inhyeokyoo.github.io/recruiting/Data-Structure-Array-LinkedList/)
- [HashTable](https://inhyeokyoo.github.io/recruiting/Data-Structure-HashTable/)
- [Stack](https://inhyeokyoo.github.io/recruiting/Data-Structure-Stack/)
- **[Queue](https://inhyeokyoo.github.io/recruiting/Data-Structure-Queue/)**
- [Graph]
- [Tree]
- [그래프(Graph)와 트리(Tree)의 차이점]
- [Binary Heap]
- [Red-Black Tree]
- [B+ Tree]


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
        # 새로운 데이터는 Last Input이 됨
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

[초보몽키의 개발블로그](https://wayhome25.github.io/cs/2017/04/18/cs-21/)

[Daim's blog](https://daimhada.tistory.com/107?category=820522)

# Circular Queue

방금전에 보았듯 list로 queue를 구현할 경우 매우 느리다. 이는 데이터가 dequeue 될 경우 데이터의 idx가 한 칸씩 이동하기 때문이다. 이를 극복하는 방법으로는 Circular queue가 있다.

마지막 index에서 다음 index로 넘어가면 IndexError가 발생하게 되는데, 이를 방지하기 위해 `(index + 1) % len(arr)`를 이용하여
index를 순환 시킨다. 따라서 max index와 index가 같게 되면, 다시 0으로 index가 순환된다.
이는 배열의 특성인 제한된 길이에 맞게 고안된 것이다.

- 처음에 공간을 생성한데로 활용할 수 있음
- 따라서 enqueue가 별로 없으면 공간이 낭비됨
- 크기를 늘리기가 힘듬

## Python 구현

다음은 linked list를 활용하여 구현한 모습이다.

생성자에서 max를 받아 queue를 max 길이만큼 초기화한다. 그리고 현재 front/rear의 index를 담을 변수를 초기화한다.
초기 front/rear의 index는 0이다.

index는 앞서 설명했듯 `(index + 1) % len(arr)`씩 움직인다. 따라서 최대 사이즈가 된다면 0으로 다시 돌아가게 된다.

enqueue가 발생하면 rear만 움직인다. 
rear에서 한 칸(`(index + 1) % len(arr)`) 이동한 index에 데이터를 추가한다.
그리고 추가한 데이터는 새로운 rear가 된다.
만일 queue가 꽉 찼을 경우 에러를 발생시킨다.

dequeue가 일어나면 front를 움직인다.
현재 front는 None으로 바꾸고, 
front는 한 칸(`(index + 1) % len(arr)`) 이동한다.
만일 queue가 비었다면 에러를 발생시킨다.

front와 rear가 같으면 큐가 비어있다. 따라서 이를 이용하여 `is_empty()`를 구현한다.

rear의 바로 다음이 front이면 큐는 꽉 찬 것이다. 따라서 이를 이용하여 `is_full()`을 구현한다.
혹은, size를 따로 지정한 후,

```python
class CircularQueue:
    def __init__(self, max=5):
        self.max = max
        self.queue = [None] * self.max
        self.front = 0
        self.rear = 0

    def is_empty(self):
        return self.rear == self.front

    def is_full(self):
        if self.next_index(self.rear) == self.front:
            return True
        else:
            return False

    def next_index(self, idx):
        return (idx + 1) % self.max

    def enqueue(self, data):
        if self.is_full():
            raise Exception("Queue is full")

        if self.rear == None:
            self.rear = 0
        else:
            self.rear = self.next_index(self.rear)

        self.queue[self.rear] = data
        return self.queue[self.rear]

    def deque(self):
        if self.is_empty():
            raise Exception("Queue is empty")
        
        self.queue[self.front] = None
        self.front = self.next_index(self.front)
        return self.queue[self.front]
```

```python
a = CircularQueue()
print("enqueue시에는 rear만 움직임.")
for i in range(4):
    a.enqueue(i)
    print(f"\t{a.queue}: front:{a.front}, rear:{a.rear}")

print("deque시에는 front만 움직임")
for i in range(4):
    a.deque()
    print(f"\t{a.queue}: front:{a.front}, rear:{a.rear}")
```

```
# 결과:
enqueue시에는 rear만 움직임.
    [None, 0, None, None, None]: front:0, rear:1
    [None, 0, 1, None, None]: front:0, rear:2
    [None, 0, 1, 2, None]: front:0, rear:3
    [None, 0, 1, 2, 3]: front:0, rear:4 # (rear + 1) % 5 == 0 이므로 full.
deque시에는 front만 움직임
    [None, 0, 1, 2, 3]: front:1, rear:4
    [None, None, 1, 2, 3]: front:2, rear:4
    [None, None, None, 2, 3]: front:3, rear:4
    [None, None, None, None, 3]: front:4, rear:4 # rear == front면 empty
```

## Reference

[Daim's blog](https://daimhada.tistory.com/168?category=820522)

[Study and Share IT](http://geonkim1.blogspot.com/2019/02/circularqueuepython.html)

[Programming PEACE](https://mailmail.tistory.com/41)

[위키피디아: Circular buffer](https://en.wikipedia.org/wiki/Circular_buffer)


# Priority Queue

우선순위 큐는 FIFO의 특성을 갖는 일반적인 큐와는 다르게 deque시에 우선 순위에 따라 먼저 제거하는 구조이다.
따라서 내부적으로 정렬하는 구조를 갖고 있어야 한다. 그러나 연결리스트나 array로 구현하게 될 경우
정렬을 위해 빈번하게 수정해야 하므로 heap을 이용하여 구현한다.

Priority 큐는 Dijkstra 알고리즘, Huffman 코딩, Prim 알고리즘 등 다양한 알고리즘에서 활용되며,
python에서는 `heapq` 모듈을 통해 구현되어 있다.

## Python 사용

데이터를 삽입할 때 `priority`를 통해 우선순위를 함께 삽입한다.

```python
import heapq

class PriorityQueue:
    def __init__(self) -> None:
        self.queue = []
        self.count = 0

    def enqueue(self, priority, value):
        self.count += 1
        heapq.heappush(self.queue, (priority, value))

    def dequeue(self):
        return heapq.heappop(self.queue)

pq = PriorityQueue()

print("Enqueue")
inputs = [1, 3, 5, 2, 0, 9, 4, 3]
for idx, item in enumerate(inputs):
    pq.enqueue(item, f"test{idx}")
    print(f"\t{item},{idx}: -> {pq.queue}")
```

```
Enqueue
    1,0: -> [(1, 'test0')]
    3,1: -> [(1, 'test0'), (3, 'test1')]
    5,2: -> [(1, 'test0'), (3, 'test1'), (5, 'test2')]
    2,3: -> [(1, 'test0'), (2, 'test3'), (5, 'test2'), (3, 'test1')]
    0,4: -> [(0, 'test4'), (1, 'test0'), (5, 'test2'), (3, 'test1'), (2, 'test3')]
    9,5: -> [(0, 'test4'), (1, 'test0'), (5, 'test2'), (3, 'test1'), (2, 'test3'), (9, 'test5')]
    4,6: -> [(0, 'test4'), (1, 'test0'), (4, 'test6'), (3, 'test1'), (2, 'test3'), (9, 'test5'), (5, 'test2')]
    3,7: -> [(0, 'test4'), (1, 'test0'), (4, 'test6'), (3, 'test1'), (2, 'test3'), (9, 'test5'), (5, 'test2'), (3, 'test7')]
```

```
Dequeue: [(0, 'test4'), (1, 'test0'), (4, 'test6'), (3, 'test1'), (2, 'test3'), (9, 'test5'), (5, 'test2'), (3, 'test7')]
    (0, 'test4') -> [(1, 'test0'), (2, 'test3'), (4, 'test6'), (3, 'test1'), (3, 'test7'), (9, 'test5'), (5, 'test2')]
    (1, 'test0') -> [(2, 'test3'), (3, 'test1'), (4, 'test6'), (5, 'test2'), (3, 'test7'), (9, 'test5')]
    (2, 'test3') -> [(3, 'test1'), (3, 'test7'), (4, 'test6'), (5, 'test2'), (9, 'test5')]
```

## Refrence

[Engineering Blog  by Dale Seo](https://www.daleseo.com/python-priority-queue/)

[Daim's blog](https://daimhada.tistory.com/169?category=820522)
