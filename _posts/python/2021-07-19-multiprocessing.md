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