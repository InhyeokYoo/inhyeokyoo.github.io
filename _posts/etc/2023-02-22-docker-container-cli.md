---
title:  "Docker container 명령어 정리"
toc: true
toc_sticky: true
categories:
  - IT
tags:
  - docker
use_math: true
last_modified_at: 2023-02-22
---

## `docker create`/`docker run`: 컨테이너 실행

컨테이너 실행을 위해선 `docker create` 혹은 `docker run`을 사용한다.
`docker create`는 run과 달리 container 내부 접근을 하지 않고 스냅샷만 생성한다.

```bash
inhyeok@DESKTOP-QDNEO0C:~$ docker create -it --name container-test1 ubuntu:14.04
95d56402964f969d6205b467dac683134b5e01722d216d94c03ea15b68543288
inhyeok@DESKTOP-QDNEO0C:~$ docker ps -a # STATUS가 start가 아닌 created
CONTAINER ID   IMAGE          COMMAND       CREATED          STATUS          PORTS     NAMES
95d56402964f   ubuntu:14.04   "/bin/bash"   3 seconds ago    Created                   container-test1
inhyeok@DESKTOP-QDNEO0C:~$ docker start container-test1 # 스냅샷 실행
container-test1
inhyeok@DESKTOP-QDNEO0C:~$ docker ps -a
CONTAINER ID   IMAGE          COMMAND       CREATED              STATUS          PORTS     NAMES
95d56402964f   ubuntu:14.04   "/bin/bash"   About a minute ago   Up 4 seconds              container-test1
inhyeok@DESKTOP-QDNEO0C:~$ docker attach container-test1 # 접속
root@95d56402964f:/#
root@95d56402964f:/# exit # 종료
exit
inhyeok@DESKTOP-QDNEO0C:~$ docker rm container-test1 # 삭제
container-test1
```


이를 `docker run`으로 수행하면 다음과 같다.

```bash
inhyeok@DESKTOP-QDNEO0C:~$ docker run -it --name container-test1 ubuntu:14.04 bash
root@13687ba7c93c:/# exit
exit
inhyeok@DESKTOP-QDNEO0C:~$ docker rm container-test1
container-test1
```

또 하나의 차이점은 `docker run`은 호스트 서버에 이미지가 다운로드 되어있지 않아도 로컬에 존재하지 않는 이미지를 자동으로 다운로드 하며, 마지막에 해당 컨테이너에 실행할 명령을 입력하면 컨테이너 동작과 함께 처리된다.

이러한 관계를 정리하면 다음과 같다.

`docker run` = [pull] + create + start [command]
{: .notice--info}

아래는 자주 사용하는 옵션이다.
- `-i`, `--interactive`: 표준 입력(stdin) 사용 (주로 `-t`와 함께 사용)
- `t`: tty(가상 터미널)을 할당. 리눅스에 키보드를 통해 표준 입력(stdin)을 전달할 수 있게한다. (주로 `-i`와 함께 사용)
- `-d`, `--detached=true`: 백그라운드에서 컨테이너 실행 후 컨테이너 ID 등록
- `--name`: 컨테이너에 이름 부여
- `--rm`: 컨테이너 종료 시 삭제
- `--restart[=no|on-failure[:max-retries]|unless-stopped|always]`: 컨테이너 종료 시 재시작 정책 지정
- `--env`: 컨테이너 환경 변수 지정
- `-v`, `--volumne=<host directory>:<container directory>`: 호스트와 컨테이너의 공유 볼륨 설정 (Mount volume)
- `-h`: 컨테이너 호스트명 지정
- `-p <host port>:<container port>`, `--publish`: 호스트 포트와 컨테이너 포트 연결
- `expose`: 호스트 내부의 다른 컨테이너들도 액세스 가능
- `-P`, `--publish-all=[true|false]`: 컨테이너 내부에 노출된(expose) 포트를 임의의 호스트 포트에 게시
- `--link=<container:container_id>`: 동일 호스트 내의 다른 컨테이너와 연결할 때 IP가 아닌 컨테이너 이름을 통해 통신


## 실행 중인 컨테이너 확인

```sh
docker ps # 현재 실행 중인 컨테이너만
docker ps -a # 종료된 컨테이너 포함
```

## docker stop: 컨테이너 정지

```sh
docker stop [OPTIONS] <CONTAINER> [<CONTAINER>...]
```

## docker restart: 컨테이너 재시작

```sh
docker restart [OPTIONS] <CONTAINER> [<CONTAINER>...]
```

## 실행 중인 container에 접속

```sh
docker attach <CONTAINER>
```

## docker rename: 컨테이너 이름 변경

```sh
docker rename <CONTAINER> <NEW_NAME>
```

## docker cp: 파일 복사

```sh
docker cp [OPTIONS] <CONTAINER>:<SRC_PATH> <DEST_PATH> # 컨테이너에서 로컬로 가져올 때
docker cp [OPTIONS] <SRC_PATH> <CONTAINER>:<DEST_PATH> # 로컬에서 컨테이너로 가져올 때
```

## 실행 중인 container에서 bash 실행
```sh
docker exec -t -i <컨테이너 이름> /bin/bash
docker exec -ti <컨테이너 이름> /bin/bash
docker exec -ti <컨테이너 이름> sh
```

## volumn 연결

우선 실행 중인 컨테이너를 commit한 후 volumn 설정을 하면 된다

```
docker commit <컨테이너 이름> <새 이미지 이름>

docker run -ti -v <디렉토리1>:<디렉토리2> <이미지 이름> bash
```