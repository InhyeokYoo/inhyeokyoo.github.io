---
title:  "NVDIA Triton inference server 최적화"
toc: true
toc_sticky: true
categories:
  - Deep Learning
tags:
  - triton
  - inferece server
use_math: true
last_modified_at: 2023-12-04
---

## Introduction

[github](https://github.com/triton-inference-server/server)

Triton은 다음과 같은 유용한 기능을 제공한다:
- **Concurrent model execution**
- **Dynamic batching**
- **Sequence batching** and **implicit state management** for stateful models
- Provides **Backend API** that allows adding custom backends and pre/post processing operations
- Supports writing custom backends in python, a.k.a. **Python-based backends**.
- Model pipelines using **Ensembling** or **Business Logic Scripting (BLS)**
- HTTP/REST and GRPC inference protocols based on the community developed KServe protocol
- **Metrics** indicating GPU utilization, server throughput, server latency, and more

## Triton Inference Server Backend

https://github.com/triton-inference-server/backend

> A Triton backend is the implementation that executes a model. A backend can be a wrapper around a deep-learning framework, like PyTorch, TensorFlow, TensorRT or ONNX Runtime. Or a backend can be custom C/C++ logic performing any operation (for example, image pre-processing).

This repo contains documentation on Triton backends and also source, scripts and utilities for creating Triton backends. You do not need to use anything provided in this repo to create a Triton backend but you will likely find its contents useful.

## 설치

https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker

가장 무난한 방법은 [NVIDIA GPU Cloud (NGC)](https://catalog.ngc.nvidia.com/)에 있는 docker image를 사용하는 것이다.

## 사용법

Triton을 사용하려면 model repository와 model configuration가 필요하다.




## Triton 동작 확인

```console
$ curl -v localhost:8000/v2/health/ready
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

Triton의 ready entrypoint를 이용하면 서버가 성공적으로 구동되었는지 확인할 수 있다.
HTTP request가 200을 내보낸다면, 준비가 된 것이다.



![image](https://user-images.githubusercontent.com/47516855/105610744-83d35280-5df4-11eb-9c7e-7615fc4bbf46.png){: .align-center}{: width='500'}
