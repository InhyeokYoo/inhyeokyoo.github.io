---
title:  "Docker 명령어 정리"
excerpt: "Docker 명령어 정리"
toc: true
toc_sticky: true

categories:
  - IT
tags:
  - docker
use_math: true
last_modified_at: 2020-07-12
---

### 1\. 실행 중인 컨테이너 확인

```
docker ps
docker ps -a
```

밑은 중지 중인 컨테이너까지 모두 포함

### 2\. 실행 중인 container에 접속

```
docker attach <컨테이너 이름 혹은 아이디>
```

### 3\. 컨테이너 이름 변경

```
docker rename <옛날 이름> <새 이름>
```

### 4\. 파일 복사

```
docker cp <컨테이너 이름>:<파일> <옮길 주소>
```