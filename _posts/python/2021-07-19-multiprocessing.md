---
title:  "Multiprocessing in Python"
toc: true
toc_sticky: true

categories:
  - Python
use_math: true
last_modified_at: 2021-07-19
---

## 들어가며

https://jinwoo1990.github.io/dev-wiki/python-concept-4/

http://pertinency.blogspot.com/2019/10/join.html

https://www.ellicium.com/python-multiprocessing-pool-process/

https://zzaebok.github.io/python/python-multiprocessing/

https://github.com/remzi-arpacidusseau/ostep-translations/blob/master/korean/README.md


https://tutorialedge.net/python/python-multiprocessing-tutorial/

Celery 는 Python 동시성 프로그래밍에서 가장 많이 사용하는 방법 중 하나이며, 분산 메시지 전달을 기반으로 동작하는 비동기 작업 큐(Asynchronous Task/Job Queue)이다.
이는 Python Framework 라고도 하지만 보통 Worker라고 불린다.
Worker는 웹 서비스에서 Back단의 작업을 처리하는 별도의 프레임이며, 사용자에게 즉각적인 반응을 보여줄 필요가 없는 작업들로 인해 사용자가 느끼는 Delay를 최소하 화기 위해 사용 된다.

예를 들어, 웹 서비스에서 응답 시간은 서비스의 생명과 직결되므로 비동기로 작업을 처리하게 넘기고 바로 응답을 하기 위해 사용 된다.

Celery는 메시지를 전달하는 역할(Publisher)과 메시지를 Message Broker에서 가져와 작업을 수행하는 Worker의 역할을 담당하게 된다.

프로세스는 '실행 중인 프로그램'입니다.

즉, 프로그램이 실행 중이라는 것은 디스크에 있던 프로그램을 메모리에 적재하여 운영체제의 제어를 받는 상태가 되었다는 것입니다.

이는 프로세서를 할당받고, 자신만의 메모리 영역이 있음을 의미하고,
프로그램이 프로세스가 되려면 프로세서 점유 시간, 메모리 그리고 현재의 활동 상태를 나타내는 PC(Program Counter), SR(Status Register) 등이 포함됩니다.

따라서 프로그램은 단지 정적 데이터만을 포함하는 정적인 개체,
프로세스는 현재 활동 상태를 포함하고 있어 동적인 개체라고 볼 수 있습니다.
[출처] 프로세스, 스레드, 프로그램의 차이|작성자 예비개발자


## `multiprocessing.Pool`

core에 작업들을 분배하고, 각 core에서 작업들을 병렬로 처리되는 구조.

## `multiprocessing.Process`

```python
def function1(x, y):
    time.sleep(10)
    print(x+y)
    return x+y

def function2(x, y):
    print(x*y)
    return x*y

t1 = Process(target=function1, args=(1, 2))
t1.start()
t2 = Process(target=function2, args=(1, 2))
t2.start()    
print(True)
```

결과

```
0.060933589935302734
2
3
```

Pool과는 다르게, 하나의 Process가 각자 target함수와 args를 가지고 일을 처리하게 된다.

## `threading`