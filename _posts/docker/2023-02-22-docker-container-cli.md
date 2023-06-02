---
title:  "Docker container 명령어 정리"
toc: true
toc_sticky: true
categories:
  - Docker
tags:
use_math: true
last_modified_at: 2023-03-30
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

### docker attach

`attach`는 실행되고 있는 컨테이너에 접속하는 명령어이다.
사용법은 아래와 같다.

```sh
docker attach <CONTAINER>
```

`attach`를 사용하게 되면 로컬의 stdin, stdout, stderr 스트림들이 해당 컨테이너와 연결된다. 
즉, 표준 입출력을 연결시켜 컨테이너 내부 제어가 가능하게 하는 기능이라고 할 수 있다.

`attach`로 컨테이너를 접속할 경우 처음 도커 컨테이너를 run하였을 때의 환경이 포그라운드로 보여진다. 
따라서 도커 컨테이너를 run할 시 `/bin/bash`로 들어간 것이 아니라면 `attach`를 활용해서는 들어갈 수 없다.

### docker exec

`exec`는 실행되고 있는 컨테이너에 새로 명령한다. 
사용법은 다음과 같다.

```sh
docker exec -it <CONTAINER> <COMMAND>
```

`exec`으로 실행한 명령어는 컨테이너에서 이미 실행되고 있던 프로세스가 실행되는 동안에만 실행되며 컨테이너를 다시 시작해도 다시 시작되지 않는다.

`docker exec`가 컨테이너 외부에서 명령을 실행시키기 위한 방법이라면, `docker attach`는 표준 입출력을 컨테이너에 붙여서 직접 명령 및 제어를 하게 만든다.

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

## init 실행 후 종료 방지

Docker run background
docker run -d를 사용하여 entrypoint 또는 cmd가 설정되어 있지 않은 이미지로 컨테이너 실행 시

$ docker run -d --name test ubuntu:20.04
589f7b7cfc6ead8950c92c5e6b09b7e7a4f0bbd89b7557a7b384a2d3c5360a70

$ docker ps -a
CONTAINER ID   IMAGE          COMMAND   CREATED          STATUS                      PORTS     NAMES
589f7b7cfc6e   ubuntu:20.04   "bash"    31 seconds ago   Exited (0) 28 seconds ago             test
위와 같이 컨테이너가 지속적으로 실행시킬 프로그램을 찾지 못하고 자동으로 종료된다.

자동으로 종료되지 않게 설정하기 위해서는 몇 가지 설정이 필요하다.

sleep infinity
docker run 명령어 마지막에 sleep infinity를 추가할 경우 정상적으로 컨테이너가 유지된다.

docker run

$ docker run -d --name test ubuntu:20.04 sleep infinity
347ae2a4d398d5f04cde0df0b5a4243d6ce52e515d1dd6da0e54a7ee1aa3153b

$ docker ps -a
CONTAINER ID   IMAGE          COMMAND            CREATED          STATUS          PORTS     NAMES
347ae2a4d398   ubuntu:20.04   "sleep infinity"   11 seconds ago   Up 11 seconds             test

tail -f /dev/null
이전부터 많이 사용한 방식으로, null device(/dev/null) 라고 불리는 리눅스 특수 장치 파일을 계속 읽음으로써 컨테이너 작업을 유지시키는 방식이다.

필자가 테스트 한 결과, run 시 아래와 같은 오류가 발생하며, docker-compose 및 Dockerfile로는 정상적으로 동작한다.
이미지에 따라 다를 수 있음으로, 위의 sleep infinity를 경우에 따라 혼용하여 사용하면 된다.

docker run

$ docker run -d --name test --entrypoint "tail -f /dev/null" ubuntu:20.04
37a992c25eeab81805a57da3ab08241315ed2745c66b15202235b2037ed4d1c8
docker: Error response from daemon: failed to create shim: OCI runtime create failed: container_linux.go:380: starting container process caused: exec: "tail -f /dev/null": stat tail -f /dev/null: no such file or directory: unknown.