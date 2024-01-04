---
title:  "WSL에서 딥러닝 환경 세팅하기"
toc: true
toc_sticky: true
categories:
  - Docker
tags:
  - WSL
use_math: true
last_modified_at: 2023-01-10
---

열심히 쓰긴 했는데, [Windows 10(WSL)에서 Docker를 활용하여 Tensorflow-GPU를 사용하기](https://velog.io/@inthecode/Windows-10WSL%EC%97%90%EC%84%9C-Docker%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-Tensorflow-GPU%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)를 보고 하는게 빠를 것 같다.

## 들어가며

이번에 AWS에서 WSL로 딥러닝 환경을 잠시 변경하게 되면서 WSL과 CUDA, nvidia-docker container toolkit을 세팅하게 되었다.
그러나 세팅하던 도중 글마다 설명이 달라 무엇을 따라야할지 헷갈렸고, 본래 서버세팅에도 익숙치도 않아 많이 고생하게 되었다.

MS에서 제공하는 [Get started with GPU acceleration for ML in WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)에서는 WSL과 Nvidia driver, docker로 setup이 끝나는 반면, [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl)에서는 CUDA를 설치한다.
또한 일반 블로그에서는 CUDA와 CuDNN을 모두 설치하라고 써있는데, 도대체 어느게 맞는지 확인하기가 어려웠다.

나의 경우에는 MS의 [Get started with GPU acceleration for ML in WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)을 기준으로 setup하였고, 그 결과 Windows 환경에서 Ubuntu와 GPU 그리고 docker를 사용할 수 있다.

한번 차근차근 시도해보자.

## WSL2 설치

가장 먼저 진행할 일은 WSL2를 설치하는 일이다.
WSL2를 설치하기 위해서는 **Windows 10 version 2004 이상 (Build 19041 이상)** 혹은 **Windows 11**이여야 한다.
이에 해당한다면 아래의 절차와 같이 **자동설치**를 하면 된다.
만약 그렇지 않을 경우 [수동설치](#수동설치)를 진행하면 된다.

### 자동설치

- 참고: [https://learn.microsoft.com/en-us/windows/wsl/install](https://learn.microsoft.com/en-us/windows/wsl/install)

Windows PowerShell을 관리자 권한으로 실행한 뒤 아래의 명령어를 통해 손쉽게 wsl을 설치할 수 있다.

```bash
wsl --install
```

만일 설치화면은 안나오고 WSL 관련 도움말이 나온다면 `wsl --list --online`를 통해 가능한 리눅스 배포판을 찾아보고 `wsl --install -d <DistroName>`를 통해 설치하면 된다.

이후 `wsl -l -v`를 통해 WSL 버전을 확인하면 끝.

WSL 버전을 변경하고 싶다면 `wsl --set-default-version <Version#>`을 통해 1 혹은 2를 넣어 변경하면 된다.

만약 잘못 설치했을 경우 프로그램 제거/추가를 통해 설치된 Ubuntu와 Windows Subsystem for Linux 이름을 갖는 프로그램을 전부 삭제하고, 제어판 > Windows 기능 켜기/끄기에서 **Linux용 Windows 하위 시스템**의 체크박스를 해제한 후 재부팅하면 삭제된다.

- 참고: [https://record-everything.tistory.com/entry/%EC%9C%88%EB%8F%84%EC%9A%B0-11%EC%97%90%EC%84%9C-WSL-%EC%99%84%EC%A0%84%ED%9E%88-%EC%82%AD%EC%A0%9C%ED%95%98%EA%B8%B0](https://record-everything.tistory.com/entry/%EC%9C%88%EB%8F%84%EC%9A%B0-11%EC%97%90%EC%84%9C-WSL-%EC%99%84%EC%A0%84%ED%9E%88-%EC%82%AD%EC%A0%9C%ED%95%98%EA%B8%B0)


### 수동설치

- 참고: [https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-2---check-requirements-for-running-wsl-2](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-2---check-requirements-for-running-wsl-2)

만일 Windows 버전이 **Windows 10 version 2004 이상 (Build 19041 이상)** 혹은 **Windows 11**보다 낮다면 수동설치를 진행해야 한다.

**관리자 권한**으로 PowerShell을 실행한 뒤 다음의 명령어를 통해 Linux용 Windows 하위 시스템의 옵션을 켜주고, **재부팅**한다.

```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

이 후 CPU 종류에 따라 버전이 나뉘게 된다. Windows 11은 신경쓸 필요가 없고, **Windows 10**인 경우
- x64의 경우: 1903 이상, Build 18362 이상.
- ARM64의 경우: 2004이상, Build 19041 이상.

이 필요하다.

이후 Virtual Machine 플랫폼 옵션 기능을 사용하도록 설정한다.
앞선 경우와 마찬가지로 **관리자 권한**으로 PowerShell을 실행한 뒤 다음의 명령어를 입력한다.

```bash
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

이후 재부팅.

마지막으로 Linux 커널 업데이트 패키지 다운로드를 진행한다. 
이를 위해서 [x64 머신용 최신 WSL2 Linux 커널 업데이트 패키지](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)를 다운받고 실행시켜준다. 
만일 ARM인 경우엔 [ARM64 머신용 최신 WSL2 Linux 커널 업데이트 패키지](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_arm64.msi)를 다운받는다.
사용하고 있는 CPU 종류를 모르는 경우엔 PowerShell에서 `systeminfo`를 통해 확인할 수 있다.

이후 `wsl --set-default-version 2`를 통해 새 Linux 배포를 설치할 때 WSL 2를 기본 버전으로 설정하도록 한다.

마지막으로 MS Store에서 Ubuntu를 다운받고 진행하면 완료.

혹시라도 중간에 무엇인가를 잘 못하여 삭제가 필요한 경우 PowerShell에서 다음의 명령어를 통해 삭제해준다.

```bash
# 설치된 wsl 목록 확인
PS C:\Users\admin> wslconfig.exe /l
Linux 배포용 Windows 하위 시스템:
Ubuntu-18.04(기본값)

# 설치된 wsl 목록 삭제
PS C:\Users\admin> wslconfig.exe /u Ubuntu-18.04
삭제중...

# 이후 확인하면 설치된 wsl목록이 없음
PS C:\Users\admin> wslconfig.exe /l
Windows Subsystem for Linux has no installed distributions.
Distributions can be installed by visiting the Windows Store:
https://aka.ms/wslstore

# 이후 Ubuntu 디렉토리 삭제 진행
```

이 후 앞서 활성화시켰던 WSL2의 기능을 disable로 바꿔준다.

```bash
dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart
dism.exe /online /disable-feature /featurename:VirtualMachinePlatform /norestart
```

이후 재부팅하자.

- 참고: [https://lahuman.github.io/wsl_uninstall/](https://lahuman.github.io/wsl_uninstall/)

## GPU 활성화

- 참고 (MS): [https://learn.microsoft.com/ko-kr/windows/wsl/tutorials/gpu-compute](https://learn.microsoft.com/ko-kr/windows/wsl/tutorials/gpu-compute)

WSL2에서 GPU를 사용하기 위해서는 Windows 환경이 **Windows 11 혹은 Windows 10, version 21H2** 이상이여야 한다.
우선 [NVIDIA CUDA on WSL driver](https://www.nvidia.com/Download/index.aspx)을 설치한다.

## Docker Desktop

- 참고 (Docker): [https://docs.docker.com/desktop/windows/wsl/#download](https://docs.docker.com/desktop/windows/wsl/#download)
- 참고 (MS): [https://learn.microsoft.com/ko-kr/windows/wsl/tutorials/wsl-containers#install-docker-desktop](https://learn.microsoft.com/ko-kr/windows/wsl/tutorials/wsl-containers#install-docker-desktop)

이후 [Docker Desktop](https://docs.docker.com/desktop/windows/wsl/#download)을 설치하고, **Settings > Resources > WSL Integration**을 활성화한 뒤 Apply & Restart를 누르고 Ubuntu를 재시작한다.

만일 Docker Desktop이 아닌 WSL에서 직접 Docker 엔진을 설치할 경우 다음과 같이 설치한다.

```bash
curl https://get.docker.com | sh

# Nvidia Container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

설치 후 다음의 명령어를 통해 GPU 벤치마크를 실행시키면 정상적으로 실행 되는 것을 확인할 수 있다.

```bash
docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

나의 경우에는 MS 가이드에서 제공하는 tensorflow 이미지가 제대로 동작하지 않았기 때문에 확실하게 하기 위해서 [pytorch 이미지](https://hub.docker.com/r/pytorch/pytorch)를 다운받고 컨테이너로 올렸다.

```bash
docker pull pytorch/pytorch
docker run -it --rm --gpus all pytorch/pytorch
```

이후엔 다음의 명령어를 통해 CUDA 사용이 가능한지 확인한다.
종종 CUDA는 잡히더라도 실제 학습/추론 시 CUDA 사용이 불가능한 경우도 있기 때문에 이도 간단하게 테스트한다.

```py
import torch

torch.cuda.is_available() # True

device = 'cuda:0'
net = torch.nn.Linear(3, 3)
tensor = torch.Tensor(3, 3)

net.to(device)
tensor = tensor.to(device)

net(tensor)
```

## Windows terminal 설치

윈도우 스토어를 통해 windows terminal을 설치하고, 이를 통해 깔끔하게 WSL 내 ubuntu 환경을 즐길 수 있다.

![Fig.1-windows-terminal]({{site.url}}{{site.baseurl}}/assets/posts/docker/set-wsl-for-dl-fig.1.png){: .align-center}{: width="600"}

