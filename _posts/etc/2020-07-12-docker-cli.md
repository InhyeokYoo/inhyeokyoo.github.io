---
title:  "Docker image 명령어 정리"
toc: true
toc_sticky: true
categories:
  - IT
tags:
  - docker
use_math: true
last_modified_at: 2023-02-22
---

## docker pull: 이미지 내려받기

도커 이미지는 기본적으로 도커 허브 레지스트리로 자동 지정되고, 특정 레지스트리를 수동으로 지정해서 받을 수도 있다.

```sh
docker [IMAGE] pull [OPTIONS] name[:TAG | @IMAGE_DIGEST]
```

실제 예시를 통해 자세히 알아보자.

```bash
inhyeok@DESKTOP-QDNEO0C:~$ docker pull debian
Using default tag: latest # 이미지 명 뒤에 태그가 없으면 latest로 지정됨
latest: Pulling from library/debian # library: 도커 허브의 공식 이미지가 저장되어있는 특별한 네임스페이스
1e4aec178e08: Pull complete # 도커 허브에서 제공된 이미지의 distribution hash.
Digest: sha256:43ef0c6c3585d5b406caa7a0f232ff5a19c1402aeb415f68bcd1cf9d10180af8 # Digest는 원격 도커 레지스트리(도커 허브)에서 관리하는 이미지의 고유 식별값
Status: Downloaded newer image for debian:latest # 다운로드한 이미지 정보가 로컬에 저장
docker.io/library/debian:latest # docker.io는 도커 허브에서 받음을 의미. 나머지는 <NAMESPACE>/<IMAGE_NAME>:<TAG>
```

다음은 옵션이다.
- `-all-tags`, `-a`: 저장소에 태그로 지정된 이미지를 모두 다운로드
- `--disable-content-trust`: 이미지 검증 작업 건너뛰기. Docker Contetns Trust (DCT)를 이용하여 이미지 신뢰성 검증
- `--platform`: 플랫폼 지정, 윈도우 도커에서 리눅스 이미지를 받아야 하는 경우 사용
- `--quite`, `-q`: 이미지 다운로드 과정에서 화면에 나타나는 상세 출력 숨김

## docker image save: 도커 이미지 파일로 관리

`docker image save`/`docker save`는 도커 원본 이미지의 레이어 구조까지 포함한 복제를 수행하여 tar 파일로 이미지를 저장한다.
도커 허브에서 이미지를 받아 내부망으로 이전하는 경우, 신규 애플리케이션 서비스를 위해 Dockerfile로 새롭게 생성한 이미지를 저장 및 배포하는 경우, 컨테이너를 commit하여 생성한 이미지를 저장 및 배포하는 경우, 개발 및 수정한 이미지 등등에서 사용한다.

```console
# 도커 이미지를 tar로 저장
docker image save [OPTIONS] IMAGE [IMAGE...]
docker image load [OPTIONS]
```

예시:

```console
inhyeok@DESKTOP-QDNEO0C:~$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
httpd        latest    3a4ea134cf8e   13 days ago   145MB
debian       latest    54e726b437fb   13 days ago   124MB
mysql        5.7       be16cf2d832a   3 weeks ago   455MB
inhyeok@DESKTOP-QDNEO0C:~$ docker image save mysql:5.7 > test-mysql57.tar
inhyeok@DESKTOP-QDNEO0C:~$ ls
test-mysql57.tar
```

불러올 때는 다음과 같이 `docker image load`를 사용한다.

```console
inhyeok@DESKTOP-QDNEO0C:~$ docker images # image 확인
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
httpd        latest    3a4ea134cf8e   13 days ago   145MB
debian       latest    54e726b437fb   13 days ago   124MB
inhyeok@DESKTOP-QDNEO0C:~$ docker image load < test-mysql57.tar # 불러오기
c233345f327a: Loading layer [==================================================>]    145MB/145MB
9117b1e53ba3: Loading layer [==================================================>]  11.26kB/11.26kB
1256ef6b8ce9: Loading layer [==================================================>]  2.385MB/2.385MB
9e296bbbda1f: Loading layer [==================================================>]  13.95MB/13.95MB
75a9fcfd26c5: Loading layer [==================================================>]  7.168kB/7.168kB
4104fbb529d5: Loading layer [==================================================>]  3.072kB/3.072kB
6740e92960ea: Loading layer [==================================================>]  79.47MB/79.47MB
912dde462543: Loading layer [==================================================>]  3.072kB/3.072kB
d34e99e3e6e5: Loading layer [==================================================>]  230.6MB/230.6MB
38c885e9f124: Loading layer [==================================================>]  17.41kB/17.41kB
bc7ce92d7b90: Loading layer [==================================================>]  1.536kB/1.536kB
Loaded image: mysql:5.7
inhyeok@DESKTOP-QDNEO0C:~$ docker images # 새롭게 추가된 mysql
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
httpd        latest    3a4ea134cf8e   13 days ago   145MB
debian       latest    54e726b437fb   13 days ago   124MB
mysql        5.7       be16cf2d832a   3 weeks ago   455MB
```

## docker image rm/docker rmi: 이미지 삭제

```console
# 정식명령
docker image rm [OPTIONS] IMAGE [IMAGE...]

# 압축명령
docker rmi [OPTIONS] IMAGE [IMAGE...]
```

예제.

```console
# latest가 아닌 이상 태그명을 명시해야 함
inhyeok@DESKTOP-QDNEO0C:~$ docker image rm mysql
Error: No such image: mysql
inhyeok@DESKTOP-QDNEO0C:~$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
httpd        latest    3a4ea134cf8e   13 days ago   145MB
debian       latest    54e726b437fb   13 days ago   124MB
mysql        5.7       be16cf2d832a   3 weeks ago   455MB
inhyeok@DESKTOP-QDNEO0C:~$ docker image rm mysql:5.7
Untagged: mysql:5.7
Deleted: sha256:be16cf2d832a9a54ce42144e25f5ae7cc66bccf0e003837e7b5eb1a455dc742b
...
```

또한, `-f`, `--force` 옵션을 통해 태그가 지정된 모든 이미지를 삭제할 수 있다.
이 때 이미지 ID는 전체가 아닌 일부만 써도 상관없다.

```console
inhyeok@DESKTOP-QDNEO0C:~$ docker images
REPOSITORY     TAG       IMAGE ID       CREATED       SIZE
debian-httpd   2.0       3a4ea134cf8e   13 days ago   145MB
httpd          latest    3a4ea134cf8e   13 days ago   145MB
debian         latest    54e726b437fb   13 days ago   124MB
inhyeok@DESKTOP-QDNEO0C:~$ docker image rm -f 3a4e
Untagged: debian-httpd:2.0
Untagged: httpd:latest
Untagged: httpd@sha256:db2d897cae2ad67b33435c1a5b0d6b6465137661ea7c01a5e95155f0159e1bcf
Deleted: sha256:3a4ea134cf8e081516a776ce184dedc28986f941ed214b9012dc888049480f5a
Deleted: sha256:019e5c44c73d76bc67f1618d02f9535348180094293dc4ddcfe70894209fd9ed
Deleted: sha256:eb991c200c9af34ef15003013e10c8ce8e143991de9780f2d0c5370041f3cf19
Deleted: sha256:127d0bec4c754ad1d28fcb982b114444cbc9aca95a6f5a7d74560e61a109a2fb
Deleted: sha256:c5c018d684454c7d5056c7f72a970aac612a515e70f3c858cba3978039a26248
Deleted: sha256:4695cdfb426a05673a100e69d2fe9810d9ab2b3dd88ead97c6a3627246d83815
inhyeok@DESKTOP-QDNEO0C:~$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
debian       latest    54e726b437fb   13 days ago   124MB
```

```console
# 이미지 전체 삭제
docker rmi $(docker images -q) # docker images -q: Only show image IDs

# 특정 이미지 이름이 포함된 것만 삭제
docker rmi $(docker images | grep debian)

# 특정 이미지 이름을 제외하고 삭제
docker rmi $(docker images | grep -v centos)
```

`docker image prune`은 다운로드한 이미지 중 컨테이너에 연결되지 않은 이미지를 제거하는데 사용된다.

```console
# -a 옵션은 사용 중이 아닌 모든 이미지 제거
docker image prune -a

# --filter until=<timestamp>를 통해 필터링 옵션을 줄 수 있다.
# docker image prun -a -f -filter "until=48h" # -f: Do not prompt for confirmation
```