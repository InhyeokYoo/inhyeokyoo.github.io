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
last_modified_at: 2020-07-08
---

백준에서 stack 풀이 만을 모아놓았다. 예전에 푼 것은 수정하기가 귀찮아서 그냥 올렸는데 앞으로 푸는 것은 풀이 과정도 정리해서 올릴 예정이다. 

사용언어는 Python이다. 

오른쪽 TOC를 통해 바로가기를 해보자.

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

풀이법:

- 정답을 출력할 print_list와 수열을 저장할 stack을 준비한다. 정답 수열의 idx를 준비하여 하나씩 탐색할 수 있게 한다.
- 1 부터 n까지 반복문에서 
    - print_list에 +를 삽입하고 stack에 숫자를 삽입한다. 만일 이 숫자가 정답 배열을 맞출 경우, stack에서 pop하고, idx를 하나 증가한다.
    - 이후 stack에서 하나씩 정답 배열과 비교한다. 만일 이 숫자가 정답 배열을 맞출 경우, stack에서 pop하고, idx를 하나 증가한다.
- 반복문이 종료되면 남아있는 stack을 전부 다 pop 한다. 이때 정답 배열과 다를 경우 에러메시지를 출력하면 된다.

- [문제보기](https://www.acmicpc.net/problem/1874)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/1874.py)

### 2504	괄호의 값

- [문제보기](https://www.acmicpc.net/problem/2504)
- [풀이보기](https://github.com/InhyeokYoo/BOJ_Algorithm/blob/master/Stack/2504.py)

### 2493	탑

- [문제보기](https://www.acmicpc.net/problem/2493)
- [풀이보기]






