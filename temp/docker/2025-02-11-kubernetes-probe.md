---
title:  "Kubernetes Probe 종류 비교"
toc: true
toc_sticky: true
categories:
  - Docker
use_math: true
last_modified_at: 2025-02-11
---

## 들어가며



## Liveness Probe

Liveness probe는 **컨테이너가 정상적으로 실행 중인지 확인**하기 위한 프로브이다.
이는 컨테이너 내부에서 실행되는 애플리케이션의 상태를 주기적으로 확인하여 문제가 발생할 경우 컨테이너를 재시작한다.
Liveness probe는 애플리케이션이 정상적인 상태를 유지하는지 확인하는 데 사용된다.

일반적으로 HTTP 요청, TCP 소켓 연결, 명령어 실행 등의 방법으로 liveness probe를 구성할 수 있다.
해당 엔드포인트로 주기적인 HTTP GET 요청을 보내 응답을 성공적으로 받는 경우 컨테이너가 정상적으로 동작 중인 것으로 판단한다.

아래는 옵션이다:
- `initialDelaySeconds`: 컨테이너가 시작된 후, liveness probe가 처음 실행되기 전에 기다리는 시간(초). 애플리케이션이 완전히 시작되기 전에 프로브가 실패하지 않도록 하기 위해 설정.
- `periodSeconds`: Liveness Probe가 주기적으로 실행되는 간격(초). `periodSeconds: 10`으로 설정하면 10초마다 liveness probe가 실행
- `timeoutSeconds`: 프로브가 응답을 기다리는 최대 시간(초). 설정된 시간 내에 응답이 없으면 프로브가 실패한 것으로 간주
- `failureThreshold`: liveness probe가 연속으로 지정된 횟수만큼 실패할 경우, Kubernetes가 컨테이너를 재시작

아래 예시를 통해 더 자세히 살펴보자.

```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    test: liveness
  name: liveness-exec
spec:
  containers:
  - name: liveness
    image: registry.k8s.io/busybox
    args:
    - /bin/sh
    - -c
    - touch /tmp/healthy; sleep 30; rm -f /tmp/healthy; sleep 600
    livenessProbe:
      exec:
        command:
        - cat
        - /tmp/healthy
      initialDelaySeconds: 5
      periodSeconds: 5
```

옵션대로라면 컨테이너가 시작한 후 5초 동안 기다리고(`initialDelaySeconds`), 이후 매 5초마다 (`periodSeconds`) liveness probe를 수행한다.
또한 위 매니페스트는 `exec.command`를 통해 health check를 하고 있는데 이는 컨테이너 시작 후 다음의 명령어를 실행한다.

```sh
/bin/sh -c "touch /tmp/healthy; sleep 30; rm -f /tmp/healthy; sleep 600"
```

이는 `/tmp/healthy` 파일을 만든 후 30초 동안은 대기하고, 이후 만들었던 `/tmp/healthy`를 제거한 뒤 600초 동안 대기한다.

따라서 첫 30초 동안에는 `/tmp/healthy`파일이 있으므로 0을 반환할 것이고, 이때동안엔 liveness probe가 성공적으로 동작한다.
이 30초 이후에는 실패한다 (0이 아닌 값을 반환).

HTTP request를 통할 때는 2xx 또는 3xx인 경우 성공으로 취급한다.

## Readiness Probe

Readiness probe는 컨테이너가 클라이언트 트래픽을 받을 준비가 되었는지 확인하기 위한 프로브이다.







아래는 triton inference server의 매니페스트이다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ template "triton-inference-server.fullname" . }}
spec:
  selector:
    matchLabels:
      app: {{ template "triton-inference-server.name" . }}
  template:
    metadata:
      labels:
        app: {{ template "triton-inference-server.name" . }}
        release: {{ .Release.Name }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.imageName }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}

          ports:
            - containerPort: 8000
              name: http
            - containerPort: 8001
              name: grpc
            - containerPort: 8002
              name: metrics
          livenessProbe:
            initialDelaySeconds: 15
            failureThreshold: 3
            periodSeconds: 10
            httpGet:
              path: /v2/health/live
              port: http
          readinessProbe:
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 3
            httpGet:
              path: /v2/health/ready
              port: http
          startupProbe:
            # allows Triton to load the models during 30*10 = 300 sec = 5 min
            # starts checking the other probes only after the success of this one
            # for details, see https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-startup-probes
            periodSeconds: 10
            failureThreshold: 30
            httpGet:
              path: /v2/health/ready
              port: http
```
