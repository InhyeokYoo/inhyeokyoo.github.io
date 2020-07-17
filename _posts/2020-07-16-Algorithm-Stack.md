---
title:  "백준 STACK 풀이"
excerpt: "백준 Stack 풀이 모음집"
toc: true
toc_sticky: true

categories:
  - Algorithm
tags:
  - Stack
  - BOJ
  - Python
use_math: true
last_modified_at: 2020-07-17
---

## Introduction

백준에서 stack 풀이 만을 모아놓았다. 예전에 푼 것은 수정하기가 귀찮아서 그냥 올렸는데 앞으로 푸는 것은 풀이 과정도 정리해서 올릴 예정이다. 

사용언어는 Python이다. 

오른쪽 TOC를 통해 바로가기를 해보자.

**<h3> 알고리즘 바로가기 </h3>**
- **[스택](https://inhyeokyoo.github.io/algorithm/Algorithm-Stack/)**
- [큐](https://inhyeokyoo.github.io/algorithm/algorithm-queue/)
- [순환 큐](https://inhyeokyoo.github.io/algorithm/Algorithm-CircularQueue/)

## 문제 모음

### 10828 스택

- [문제보기](https://www.acmicpc.net/problem/10828)   
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/10828.py)
- [풀이보기(class로 구현)](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/10828_class.py)

### 9012 괄호

- [문제보기](https://www.acmicpc.net/problem/9012)   
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/9012.py)

### 10799 쇠막대기

- [문제보기](https://www.acmicpc.net/problem/10799)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/10799.py)

### 1874    스택 수열

- [문제보기](https://www.acmicpc.net/problem/1874)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/1874.py)
- 풀이법:
  - 정답을 출력할 print_list와 수열을 저장할 stack을 준비한다. 정답 수열의 idx를 준비하여 하나씩 탐색할 수 있게 한다.
  - 1 부터 n까지 반복문에서 
      - print_list에 +를 삽입하고 stack에 숫자를 삽입한다. 만일 이 숫자가 정답 배열을 맞출 경우, stack에서 pop하고, idx를 하나 증가한다.
      - 이후 stack에서 하나씩 정답 배열과 비교한다. 만일 이 숫자가 정답 배열을 맞출 경우, stack에서 pop하고, idx를 하나 증가한다.
  - 반복문이 종료되면 남아있는 stack을 전부 다 pop 한다. 이때 정답 배열과 다를 경우 에러메시지를 출력하면 된다.


### 2504	괄호의 값

- [문제보기](https://www.acmicpc.net/problem/2504)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/2504.py)

### 2493	탑

- [문제보기](https://www.acmicpc.net/problem/2493)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/2493.py)
- 풀이법
  - 탑의 배열을 준비한다. 
  - 배열을 iteration하며 stack에 집어넣는다.
  - 만일 stack에 있는 탑의 높이가 현재 iteration 탑의 크기보다 작다면, stack에 있는 탑은 절대 신호를 수신하지 못하므로 `pop`해서 비교한다.

처음에는 바보처럼 탑 배열을 stack으로 넣고 pop하며 크기를 비교하느라 O(N^2)의 시간이 걸렸다.
stack에 대한 이해가 부족하다는걸 느꼈다.
바보처럼 짠 코드는 다음과 같다.
```python
import sys

num = int(sys.stdin.readline().strip()) # [1, 500,000]
stack = list(map(int, sys.stdin.readline().split())) # O(N)

answer = [0 for _ in range(num)] # O(N)
mn_fire = float('inf')
mx_fire = 0
prev_hit_idx = 0

while len(stack) != 1: # O(N)
    fire = stack.pop()

    record_idx = len(stack) # answer에 기록 용도
    hit_idx = len(stack) # hit 확인 용도
    
    if fire < mn_fire or record_idx <= prev_hit_idx:
        while hit_idx > 0: # O(N^2)
            hit_idx -= 1

            if stack[hit_idx] > fire:
                answer[record_idx] = hit_idx + 1 # hit_idx는 index이므로 +1을 해줌.
                mx_fire = stack[hit_idx]
                mn_fire = fire
                prev_hit_idx = hit_idx
                break
    else:
        answer[record_idx] = prev_hit_idx + 1
        
print(*answer)
```

`while`에서 탑의 배열을 한 번, 크기 비교를 위해 남는 stack에서 한 번, 총 2번의 iteration을 도는 것을 확인할 수 있다.

### 5397 키로거

- [문제보기](https://www.acmicpc.net/problem/5397)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/5397.py)

### 6549 히스토그램에서 가장 큰 직사각형

- [문제보기](https://www.acmicpc.net/problem/6549)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/6549.py)
- 풀이법: 
- 길이를 넣다가 더 작은게 들어오면 pop해서 그때 계산하는 방법. 이러려면 idx를 통해 가로길이를 세야 함.
- 0: stack: [(0, 2)]. 처음엔 그냥 넣음.
- 1: stack: [(0, 2)]. (1, 1). 
  - 2 > 1 이므로, 
    - 2를 pop. 넓이: 2 x (1 - 0) = 2. 
    - (1, 1)은 pop된 데이터의 idx로 바꿔줌. (0, 1)append.
- 2: stack: [(0, 1)]. 
  - 1 < 4 이므로 그냥 append.
- 3: stack: [(0, 1), (2, 4)]. 
  - 4 < 5 이므로 그냥 append.
- 4: stack: [(0, 1), (2, 4), (3, 5)]. 
  - 5 > 1 이므로 
    - 5를 pop. 넓이: 5 * (4 - 3)
  - 4 > 1 이므로 
    - 4를 pop. 넓이: 4 * (4 - 2)
  - (4, 1)은 pop된 데이터의 idx로 바꿔줌. (2, 1) append.
- ......

- 문제점:
  - 내림차순으로 들어오면 큰 것들은 이미 pop했기 때문에 계산이 안됨.
    따라서 마지막으로 pop한 (히스토그램에선 맨 앞) 자리에 값을 넣어줘야 함.
  - 마지막은 iteration이 안돌아감. 따라서 배열에 [0]을 넣어줌.
  - `max(area)`에서 런타임 에러 발생. 길이가 0인 경우를 못봄.


### 9935 문자열 폭발

- [문제보기](https://www.acmicpc.net/problem/9935)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/9935.py)
- 풀이법:
  - 그냥 풀게 될 경우 시간복잡도가 매우 크게 증가한다. 따라서 stack을 이용해야 한다.
  - INPUT: string, bomb
  - SET: TEMP
  - for i = 1, ..., length of string
      - string의 i-th 원소를 temp에 push한다.
      - string의 i-th 원소를 폭탄의 마지막 문자와 비교한다.
      - 맞다면,
          - temp의 뒤에서부터 폭탄 문자열의 길이만큼과 폭탄을 비교한다.
              - 맞다면 temp를 pop하여 폭탄을 제거함.
  - end loop

그냥 생으로 풀면 O(N^2) 나오므로 잘 생각해야 한다. 만약 코딩테스트에서 이런 문제를 봤다면, regular expression으로 풀었을 거 같은데, 이 경우에도 시간초과가 발생한다.
```python
import sys
import re

string = sys.stdin.readline().rstrip()
bomb = sys.stdin.readline().rstrip()
n = len(bomb)

p = re.compile(bomb)

loop_ctr = True

while loop_ctr:
    if p.search(string):
        string = p.sub('', string)
    else:
        loop_ctr = False

if string:
    print(string)
else:
    print('FRULA')
```

따라서 다음 방법으로 풀었었는데, kO(N) 정도가 되서 역시 실패하였다.
```python
import sys

string = list(sys.stdin.readline().rstrip()) # O(1,000,000)
bomb = list(sys.stdin.readline().rstrip()) # O(36)
bomb.reverse() # O(36)

n = len(bomb)
temp = []

while string: # O(1,000,000)
    temp.append(string.pop())
    
    if temp[-n:] == bomb: # O(36)
        temp = temp[:-n] # O(36)
else:
    if temp:
        temp.reverse() # O(1,000,000)
        print(*temp)
    else:
        print('FRULA')
```

여기서의 실책은 `while`을 쓰느라고 `reverse`와 slice로 시간 복잡도를 증가한 것이다.
잘 기억해뒀다가 써먹으면 좋을 것 같다.

### 1918 후위 표기식

- [문제보기](https://www.acmicpc.net/problem/1918)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/1918.py)
- 풀이과정:
  - INPUT: STRING
  - SET: ANSWER: stack, STACK: stack, PRIORITY: hashtable
  - 연산자의 순위를 PRIORITY에 초기화함. */가 1순위, +-가 2순위, )가 3순위
  - START for i = 0, 1, ..., len(string)
      - if: string의 i-th element가 )라면,
          - (가 나올 때가지 STACK에서 pop하여 ANSWER에 PUSH
      - else if: string의 i-th element가 알파벳이라면,
          - ANSWER에 PUSH
      - else if: string의 i-th element가 (라면,
          - STACK에 PUSH
      - else if: string의 i-th element가 )라면,
          - (전까지의 stack을 전부 다 pop하고
      - else if: string의 i-th element가 연산자라면,
          - stack내에 자기 이상의 우선순위를 갖는 연산자는 다 뱉음.
          - (는 제외함.
            
문제점:
    input부터 FIFO구존데? 뒤로 쓸 수 있나? 
        -> for문으로 하면 FIFO가능
    어떻게 합치지? extend는 부담스럽고, 또 앞에다가 더하는건 어떻게 하지?
        -> print(end='')를 통해 한 줄로 이어서 출력이 가능.