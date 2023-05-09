---
title:  "쿠버네티스 인 액션 정리"
toc: true
toc_sticky: true
categories:
  - Docker
  - kubernetes
use_math: true
last_modified_at: 2023-02-24
---

스케줄링: 파드가 특정 노드에 할당되며, 이는 즉시 실행됨

## Replication Controller

- 참고: https://velog.io/@squarebird/Kubernetes-Replica-Set%EA%B3%BC-Deployment

레플리케이션 컨트롤러 (ReplicationController):
- 파드와 파드의 복제본을 작성하고 관리
- 현재는 `deployment`를 이용하여 `ReplicaSet`을 구성하는 것이 replication을 설정하는 가장 좋은 방법임

디플로이먼트 (Deployment)
- RC의 대체
- 파드와 RC에 대해 선언적 업데이트를 제공
- 디플로이먼트를 통해 레플리카셋을 제어하고, 레플리카셋이 파드를 제어하는 구조이기 때문에 관리자는 파드를 직접 제어해서는 안됨

레플리카셋 (ReplicaSet)
- 레플리카 파드를 안정적으로 유지하는데 사용
- RC의 후속판으로 비슷한 역할을 하지만 RC는 set-based selector를 지원하지 않으므로 ReplicaSet을 사용하는게 권장됨
- 어플리케이션 업데이트에는 적용하기 어려움 --> Deployment 사용

클러스터
클러스터(cluster)는 공통의 목표를 위해 작동하는 컴퓨터 또는 애플리케이션들의 그룹이다. 클라우드 네이티브 컴퓨팅의 관점에서, 이 용어는 쿠버네티스에 가장 일반적으로 적용된다. 쿠버네티스 클러스터는 컨테이너를 통해 실행되는 서비스(또는 워크로드) 집합이며, 이는 보통 서로 다른 머신(machines)에서 구동된다. 네트워크를 통해 연결된 모든 컨테이너화 (containerized)된 서비스의 모음(collection)은 클러스터를 나타낸다.


명령적 vs 선언적
출처: [쿠버네티스에서 명령형 접근법과 선언형 접근법의 차이 이해하기](https://seongjin.me/kubernetes-imparative-vs-declarative/)


In this book, used command is `kubectl run kubia --image=luksa/kubia --port=8080 --generator=run/v1` which used to create ReplicationController back in the days when book was written however this object is currently depracated.

Now `kubectl run` command creates standalone pod without ReplicationController. So to expose it you should run:

```sh
$ # kubectl expose rc kubia --type=LoadBalancer --name kubia-http
$ kubectl expose pod kubia --type=LoadBalancer --name kubia-http

$ # In order to create a replication it is recommended to use Deployment. To create it using CLI you can simply run
$ kubectl create deployment <name_of_deployment> --image=<image_to_be_used>

$ # It will create a deployment and one pod. And then it can be exposed similarly to previous pod exposure:
$ kubectl expose deployment kubia --type=LoadBalancer --name kubia-http

```


```sh
$ kubectl create deploy kubia-deploy --image=myrepo/kubia
deployment.apps/kubia-deploy created

$ kubectl get deploy
NAME           READY   UP-TO-DATE   AVAILABLE   AGE
kubia-deploy   1/1     1            1           8s

$ kubectl scale deploy kubia-deploy --replicas=3
deployment.apps/kubia-deploy scaled

$ kubectl get po
NAME                            READY   STATUS    RESTARTS   AGE
kubia-deploy-68675b44f6-52rbb   1/1     Running   0          3m45s
kubia-deploy-68675b44f6-ddt8g   1/1     Running   0          3m45s
kubia-deploy-68675b44f6-np624   1/1     Running   0          4m42s

$ kubectl expose deploy kubia-deploy --type=LoadBalancer --name kubia-http --port=8080
service/kubia-http exposed
```

파드(pod)
- k8s는 컨테이너를 개별적으로 배포하기보다는 하나 이상의 컨테이너를 가진 파드를 배포하고 운영
  - 일반적으로 파드는 하나의 컨테이너만 포함
- Q. 파드를 구성할 때 뭘 기준으로 묶는가?
  - 컨테이너는 단일 프로세스 실행을 목적으로 설계 (자식 프로세스 생성 제외)
  - 단일 컨테이너에서 관련 없는 다른 프로세스를 실행하면 로그 관리 등 번거로움이 많음
  - 따라서 각 프로세스를 개별 컨테이너로 실행해야 하며, 여러 프로세스를 단일 컨테이너로 묶지 않기 때문에 이를 묶고 관리하는 상위구조가 바로 파드
  - 컨테이너 모음(파드)를 이용해 밀접하게 연관된 프로세스를 함께 실행하고, 단일 컨테이너 안에서 모두 함께 실행되는 것처럼 거의 동일한 환경을 제공하면서도 격리된 상태로 유지
  - 아래와 같은 질문을 할 필요가 있음:
    - 컨테이너를 함께 실행하는가? 혹은 서로 다른 호스트에서 실행할 수 있는가?
    - 여러 컨테이너가 모여 하나의 구성 요소를 나타내는가? 혹은 개별적인 구성 요소인가? 각 파드에는 밀접하게 관련 있는 구성 요소나 프로세스만 포함해야 한다
    - 컨테이너가 함께, 혹은 개별적으로 스케일링 되어야 하는가? 파드는 스케일링의 기본 단위이므로 이를 고려해서 구성해야 한다
- 파드가 여러 컨테이너를 갖고 있을 경우 모든 컨테이너는 항상 하나의 워커 노드에서 실행되지 여러 워커 노드에 걸쳐 실행되지 않음
- 파드 내부의 컨테이너는 동일한 네트워크 네임스페이스와 UTS 네임스페이스 안에서 실행되기 때문에 동일한 IP 주소와 포트 공간을 공유함
  - 따라서 파드 내부에서는 localhost를 통해 서로 통신

파드 생성:
- `kubectl run`: 제한된 속성 집합만 설정 가능
- 파드를 포함한 다른 쿠버네티스 리소스는 일반적으로 k8s REST API 엔드포인트에 JSON/YAML 메니페스트를 전송해 생성
  - YAML 파일에 모든 k8s 오브젝트를 정의하면 버전 관리 시스템에 넣는 것이 가능해져 그에 따른 모든 이점을 누릴 수 있음
  - `kubectl create -f [YAML|JSON]`
- 정의방법
  - YAML에서 사용하는 k8s API 버전과 YAML이 설명하는 리소스 유형
  - Metadata: 이름, 네임스페이스, 레이블 및 파드에 관한 기타 정보
  - Spec: 파드 컨테이너, 볼륨, 기타 데이터 등 파드 자체에 관한 실제 명세
  - Status: 파드 상태, 각 컨테이너 설명과 상태, 파드 내부 IP, 기타 기본 정보 등 현재 실행 중인 파드에 관한 현재 정보 포함

로그관련:
- 컨테이너 로그는 하루 단위로, 10MB 크기에 도달할 때마다 순환. `kubectl logs`는 마지막으로 순환된 로그 항목만 보여줌
- 파드가 삭제되면 로그도 삭제됨. 이를 방지하기 위해선 중앙집중식 로깅을 설정해야 함.

`kubectl get po kubia-manual -o yaml` 와 같이 YAML 정보 확인 가능 (json도 가능)

`kubectl logs [POD]`: 파드 로그 가져오기
`kubectl logs <POD> -c <CONTAINER>`: 파드 내 특정 컨테이너의 로그 확인

서비스:
- 하나 이상의 파다가 서비스 뒤에 존재할 때 무작위로 다른 파드를 호출
- 서비스는 다수 파드 앞에서 로드밸런서 역할을 함
- 서비스는 항상 동일한 주소를 갖음
- 5장에서 다시 살펴볼 것

포트 포워딩(port forwarding):
- `kubectl expose`외 테스트/디버깅 목적으로 연결할 때 사용
- 서비스를 거치지 않고 특정 파드와 대화 가능
- `kubectl port-forward`

레이블:
- 파드와 k8s 리소스를 조직화
- 리소스에 첨부하는 key-value로 레이블 셀렉터를 사용해 리소스를 선택할 때 활용
- 레이블 키가 리소스 내에서 고유하다면 하나 이상 원하는 만큼 레이블을 가질 수 있음
- YAML 내에서 `metadata.labels`에 추가
  - 
  ```yaml
  ...
  metadata:
    labels:
      creation_method: manual
      env: prod
      ...
  ```
- `kubectl get pods --show-labels`를 통해 label 볼 수 있음
- `kubectl get pods -L <LABLENAME>, [LABLENAME, ...]`으로 레이블을 열에 표시 가능
- `kubectl lable po kubia-manual creation_method=manual`를 통해 레이블 추가
- `kubectl label po <POD> <LABEL>=<NAME> --overwrite`를 통해 수정 가능

레이블 셀렉터
- 레이블과 함께 사용
- 특정 레이블로 태그된 파드의 부분 집합을 선택해 원하는 작업을 수행
- 특정 값과 레이블을 갖는지 여부에 따라 리소스 필터링 기준이 됨
- 리소스 선택 기준:
  - 특정 키 포함/포함X 레이블
  - 특정 키와 값을 가진 레이블
  - 특정한 키를 갖고 있지만 다른 값을 가진 레이블
- `kubectl get po -l <KEY>[=VALUE]` 특정 레이블 값 갖는 애들 뽑기
- `kubectl get po -l <KEY>`: <KEY> 레이블을 갖는 애들 뽑기. 없는 애들은 `kubectl get po -l '!<KEY>'`로 바꿔준다.
- 파드 스케줄링 제한도 가능

이전에 리소스에는 다 사용가능하다고 했으므로 노드에도 추가가 가능하다.
`kubectl label node <NODE NAME> <KEY>=<VALUE>`
특정 레이블이 달린 노드에 파드를 배포하고 싶다면 아래와 같이 `spec.nodeSelector` 메니페스트를 변경하자.

```yaml
...
spec:
  nodeSelector:
    gpu: "true"
  ...
```

어노테이션:
- k:v 쌍으로 레이블과 거의 비슷하지만 식별 정보 X
- 레이블과는 달리 오브젝트를 묶을 수 없으며, 셀렉터도 없음
- 반면 레이블보다 훨씬 더 많은 정보를 보유 가능
- 새로운 기능을 추가하거나 파드나 다른 API 오브젝트에 설명 추가하는데 사용
- `kubectl annotate pod <POD NAME> <ANNOTATION>`

파드 삭제:
- `kubectl delete po <POD NAME> [POD NAME]`
- `kubectl delete po -l creation_method=manual` 레이블을 이용하여 삭제

## 4. 레플리케이션과 그 밖의 컨트롤러: 관리되는 파드 배포

이 장이 끝나면...
- 파드의 안정적인 유지
- 동일한 파드의 여러 인스턴스 실행
- 노드 장애 시 자동으로 파드 재스케줄링
- 파드의 수평 스케줄링
- 각 클러스터 노드에서 시스템 수준의 파드 실행
- 배치 잡 실행
- 잡을 주기적/한 번만 실행하도록 스케줄링

- 파드를 직접 생성하기보단, RC/deployment 같은 리소스를 생성해 실제 파드를 생성하고 관리
- 노드 전체에 장애가 발생하면 노드 내 파드는 유실되며 RC나 이와 비슷한 컨트롤러가 이를 관리하지 않는 이상 새로운 파드로 대체되지않음
- 따라서 k8s가 컨테이너가 여전히 살아 있는지 체크하고 죽으면 다시 살리는 방법 등을 확인

### 4.1 파드를 안정적으로 유지하는 방법

- 컨테이너에 문제가 발생 시 `kubectl`이 알아서 재시작을 진행
- 그러나 무한루프, 교착 상태 등에 빠져 응답을 못하는 경우 등은 이를 통해 해결이 어려움
  - 애플리케이션을 다시 시작되도록하려면 외부에서 상태를 체크해야 함

라이브니스 프로브(liveness probe):
- 파드 호스팅 중인 노드의 `kubectl`이 컨테이너가 살아 있는지 확인할 때 사용
  - 노드 자체에 크래시가 발생한 경우 파드를 재생성하는 것은 컨트롤 플레인의 몫
  - 직접 생성한 파드는 `kubectl` 노드에서 관리되고 실행되기 때문에 노드 자체가 고장날 경우 할 수 있는게 없음
- 종류:
  - HTTP GET 프로브로 IP 주소, 포트, 경로에 HTTP GET을 수행하여 응답 코드에 오류/응답 없는 경우 프로브가 실패한 것으로 간주하여 컨테이너를 재시작한다
  - TCP 소켓 프로브: 컨테이너 포트에 TCP 연결을 시도하여 실패할 경우 컨테이너를 재기동
  - Exec 프로브: 컨테이너 내의 임의의 명령을 실행하고 명령의 종료 상태 코드를 확인(상태코드 0이면 성공).
- 파드의 스펙에 컨테이너 라이브니스 프로브를 지정할 수 있음
- 예시:
  ```yaml
  apiVersion: v1
  kind: Pod
  metadata:
    name: kubia-liveness
  spec:
    containers:
    - image: luksa/kubia-unhealthy
      name: kubia
      livenessProbe:
        httpGet:
          path: /
          port: 8080
  ```
- `kubectl describe`를 통해 관련 정보를 확인할 수 있음
  - 예시 
    ```sh
    Name:             kubia-liveness
    Namespace:        default
    Priority:         0
    Service Account:  default
    Node:             minikube/192.168.49.2
    Start Time:       Mon, 27 Feb 2023 21:34:27 -0800
    Labels:           <none>
    Annotations:      <none>
    Status:           Running
    IP:               10.244.0.26
    IPs:
      IP:  10.244.0.26
    Containers:
      kubia:
        Container ID:   docker://b1528c74ca2738c86da2145700d6c377c94b62df9dcc620dec09ed1a4b3b3f12
        Image:          luksa/kubia-unhealthy
        Image ID:       docker-pullable://luksa/kubia-unhealthy@sha256:5c746a42612be61209417d913030d97555cff0b8225092908c57634ad7c235f7
        Port:           <none>
        Host Port:      <none>
        State:          Running # 현재 컨테이너 상태
          Started:      Mon, 27 Feb 2023 21:36:30 -0800
        Last State:     Terminated # 이전 컨테이너 상태
          Reason:       Error
          Exit Code:    137
          Started:      Mon, 27 Feb 2023 21:34:44 -0800
          Finished:     Mon, 27 Feb 2023 21:36:28 -0800
        Ready:          True
        Restart Count:  1
        Liveness:       http-get http://:8080/ delay=0s timeout=1s period=10s
        ...
    ```
- timeout: 제한시간, delay: 프로브 지연시간, period: 프로브 수행 주기, failure: 연속 실패 시 다시 시작
- 설정방법: 매니페스트 `livenessProbe`에 속성 추가
- `initialDelaySeconds`를 설정하지 않으면 컨테이너가 시작하자마자 프로브를 시작하기 때문에, 약간의 텀을 줘야만 요청을 받을 준비를 할 수 있다.
- 라이브니스 프로브는 어플리케이션 내부만 체크하고 외부 요인의 영향을 받지 않도록 해야함
- 라이브니스 프로브는 너무 많은 리소스를 사용해선 안 되며, 완료하는데 오래걸려선 안된다 (1초 이내)
- 프로브에 재시도 루프를 구현하지 말 것

레플리케이션 컨트롤러 (RC, Raplication Controller)
- TODO:
  - Deployment 혹은 Replicaset에서 쓰는 방법을 확인해야 함
- k8s 리소스가 항상 실행되도록 보장: 파드의 수가 레이블 셀렉터와 일치하는지 확인
- 레플리케이션이란 파드의 여러 복제본(레플리카)을 관리하기 때문에 이러한 이름이 붙음
- 노드가 사라지거나, 노드에서 파드가 제거되는 등 어떤 이유에서든 파드가 사라지면 이를 감지하여 교체 파드를 생성
- RC로 관리되지 않는 파드는 재생성되지 않음
- RC는 특정 레이블 셀렉터와 일치하는 파드 세트에 작동한다.
- 요소:
  - 레이블 셀렉터: RC의 범위에 있는 파드 결정
  - 레플리카 수(replica count): 실행할 파드의 의도하는 수를 지정
  - 파드 템블릿(pod template): 새로운 파드 레플리카 만들 때 사용
- 오로지 레플리카 수의 변경만 기존 파드에 영향을 미침
- 레이블 셀렉터를 변경할 경우 기존 파드가 RC의 범위를 벗어나므로 관리를 중지
- 파드 생성 후에는 이미지, 환경변수 등 기타 사항에 신경을 쓰지 않으므로 템플릿은 새 파드를 생성할 때만 영향을 줌
- RC 사용시 장점:
  - 파드를 일정 수 유지
  - 노드에 장애 발생 시 노드 내 모든 파드에 관한 교체 복제본이 생성
  - 수동/자동으로 파드를 수평으로 스케일링 가능
- 파드 인스턴스는 다른 노드로 재배치되지 않고 RC가 새로운 파드 인스턴스를 생성함
- 매니페스트
- ```yaml
  apiVersion: v1
  kind: ReplicationController
  metadata:
    name: kubia
  spec:
    replicas: 3
    selector: # Pod selector: RC가 관리하는 파드 선택
      app: kubia
    template:
      metadata:
        labels: # RC의 레이블과 일치해야 함
          app: kubia
      spec:
        containers:
        - name: kubia
          image: luksa/kubia
          ports:
          - containerPort: 8080
  ```
- 파드의 레이블을 변경해 더 이상 RC의 레이블 셀렉터와 일치하지 않으면 수동으로 만든 파드와 동일한 취급, 즉, 아무도 이 파드를 관리하지 않음
- 위 매니페스트는 `app=kubia`인 파드를 관리하기 때문에 이 레이블에 변경이 일어나면 관리되지 않는다.
- ```sh
  inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get pod
  NAME          READY   STATUS    RESTARTS   AGE
  kubia-s85sh   1/1     Running   0          71m
  kubia-s9krn   1/1     Running   0          72m
  kubia-zlb2j   1/1     Running   0          72m
  inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl label pod kubia-s85sh app=foo --overwrite
  pod/kubia-s85sh labeled
  inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get pods -L app
  NAME          READY   STATUS    RESTARTS   AGE   APP
  kubia-s85sh   1/1     Running   0          72m   foo
  kubia-s9krn   1/1     Running   0          72m   kubia
  kubia-sb9jt   1/1     Running   0          18s   kubia
  kubia-zlb2j   1/1     Running   0          72m   kubia
  inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ 
  ```
- 파드 템플릿을 변경하는 경우 새롭게 생기는 파드에만 적용됨

수평 파드 스케일링
- 파드 수 변경 --> RC 리소스의 `replicas` 필드 값 변경하면 됨

삭제 시:
- RC를 삭제하면 파드도 삭제됨
- RC로 생성한 파드는 RC의 필수적인 부분이 아니라 관리를 받는 것이기 때문에 RC만 삭제하고 파드는 남겨놓을 수 있음
  - `--cascade=false` 옵션을 통해 가능
  - 다른 RC를 작성하여 관리할 수도 있음

## 레플리카셋

- RC에 비해 좀 더 풍부한 표현식을 사용함
- RC의 셀렉터는 특정 레이블을 갖는 파드만 매칭시키는 반면, 레플리카셋의 셀렉터는 특정 레이블이 없는 파드나 레이블 값과 상관없이 특정 레이블의 키를 갖는 파드와 매칭이 가능
  - e.g. `env=production`과 `env=devel`인 파드를 동시에 매칭

매니페스트:
```yaml
apiVersion: apps/v1 # 원래는 apps/v1beta2 이지만 현재는 apps/v1으로 바뀜
kind: ReplicaSet
metadata:
  name: kubia
spec:
  replicas: 3
  selector:
    matchLabels:  # RC와의 차이점: spec.selector.matchLabels에 지정해야 함
      app: kubia
  template:
    metadata:
      labels:
        app: kubia
    spec:
      containers:
      - name: kubia
        image: luksa/kubia
```

- 혹은 `MatchExporessions`를 이용하여 더 강력한 셀렉터를 만들 수 있다.

```yaml
apiVersion: apps/v1 # apps/v1beta2
kind: ReplicaSet
metadata:
  name: kubia
spec:
  replicas: 3
  selector:
    matchExpressions: # <- matchExpressions: 모든 값이 true여야 함
      - key: app
        operator: In # 레이블의 값이 지정된 값(kubia) 중 하나여야 함
        values:
         - kubia
  template:
    metadata:
      labels:
        app: kubia
    spec:
      containers:
      - name: kubia
        image: luksa/kubia
```

`kubectl delete rs <RS NAME>`을 이용하여 RS와 파드를 전부 삭제할 수 있다.

## 데몬셋을 이용하여 노드에서 정확히 한 개의 파드 실행하기

- RC/RS를 사용하면 k8s 클러스터 내 어딘가에 지정된 수만큼의 파드를 실행
- 각 노드에 하나의 파드씩만 실행하고 싶은 경우
  - e.g. 모든 노드에서 로그 수집기와 리소스 모니터링을 하는 경우, kube-proxy
- 데몬셋(DaemonSet) 오브젝트를 이용
- 복제본 수 개념 X --> 파드 셀렉터와 일치하는 파드 하나가 각 노드에서 실행 중인지 확인하는게 전부
- 노드 다운 시 다른 곳에서 파드를 생성하지 않고 새 노드가 추가되면 여기에 파드를 배포
- 파드를 삭제하는 경우엔 다시 생성함
- `node-Selector` 속성을 지정하지 않으면 모든 노드에 파드를 배포

매니페스트:
```yaml
apiVersion: apps/v1 # apps/v1beta2
kind: DaemonSet # DaemonSet으로 구성
metadata:
  name: ssd-monitor
spec:
  selector:
    matchLabels:
      app: ssd-monitor
  template:
    metadata:
      labels:
        app: ssd-monitor
    spec:
      nodeSelector: # 파드 템플릿 template.spec 내에 disk=ssd 레이블이 있는 노드를 선택하게 함
        disk: ssd
      containers:
      - name: main
        image: luksa/ssd-monito
```
매니페스트로부터 DS를 생성하더라도 파드는 배포되지 않는다.
이는 레이블을 추가하지 않았기 때문이다(`disk=ssd`).
이후 노드에 레이블을 추가하여 데몬셋이 파드를 관리하게 하자.
`kubectl label node <NODE NAME> <KEY>=<VALUE>`
아주 당연하게도 노드에서 레이블을 제거하면 파드도 제거된다.

## 4.5 완료 가능한 단일 태스크를 수행하는 파드 실행
- 이전까지는 계속 실행되야 하는 파드에 관해서만 다룸
- 작업을 완료하면 종료되는 케이스를 다뤄야 함 --> 잡

잡 리소스(Job resource)
- 다른 리소스와 유사하지만 컨테이너 내부 프로세스가 성공적으로 완료되면 컨테이너를 다시 시작하지 않는 파드를 실행
- 제대로 완료되야하는 임시 작업에 유용
- 노드 장애 발생 시 해당 노드에 있던 잡이 관리하는 파드는 RS 파드와 같은 방식으로 다른 노드로 다시 스케줄링
- 프로세스 자체에 장애 발생 시 잡에서 컨테이너를 다시 시작할지 정할 수 있음

매니페스트:
```yaml
apiVersion: batch/v1
kind: Job # Job 생성
metadata:
  name: batch-job
spec:
  template: # pod selector가 없으며, 파드 템플릿의 레이블을 기반으로 만들어짐
    metadata:
      labels:
        app: batch-job
    spec:
      restartPolicy: OnFailure # 기본 재시작 정책(always)을 사용할 수 없음
      containers:
      - name: main
        image: luksa/batch-job
```

- `kubectl get jobs`로 조회 가능
  - 조회 시 `--show-all`/`-a` 옵션을 사용하지 않으면 완료된 파드는 표시되지 않음
  - 종료되면 아래와 같이 STATUS는 Completed로 나옴
  - 파드가 삭제되지 않으므로 `kubectl logs <POD>`를 통해 로그 검사 가능

```sh
inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get po
NAME              READY   STATUS      RESTARTS   AGE
batch-job-ks4hk   0/1     Completed   0          2m14s
```

- 잡 혹은 파드를 삭제하면 파드가 삭제됨

- 두 개 이상의 파드 인스턴스를 생성하여 병렬/순차적으로 실행 가능
  - `spec`에 `completiond`와 `parallelism` 속성을 설정하여 가능

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: multi-completion-batch-job
spec:
  completions: 5 # 차례대로 다섯개의 파드를 실행: 파드를 하나 만들고, 컨테이너 완료 시 새로 생성
  template:
    metadata:
      labels:
        app: batch-job
    spec:
      restartPolicy: OnFailure
      containers:
      - name: main
        image: luksa/batch-job
```

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: multi-completion-batch-job
spec:
  completions: 5 # 다섯 개의 파드를 성공적으로 완료해야 함
  parallelism: 2 # 두 개까지 병렬로 실행 가능
  template:
    metadata:
      labels:
        app: batch-job
    spec:
      restartPolicy: OnFailure
      containers:
      - name: main
        image: luksa/batch-job
```

- 이 후 파드를 두 개 생성하고 병렬로 실행하는 것을 확인할 수 있으며, 이 중 하나가 완료되면 다섯 개의 파드가 성공적으로 완료될 때까지 파드를 실행

```sh
inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get pod
NAME                               READY   STATUS              RESTARTS   AGE
multi-completion-batch-job-hj96w   1/1     Running             0          5s
multi-completion-batch-job-mwqng   0/1     ContainerCreating   0          5s
inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get pod
NAME                               READY   STATUS              RESTARTS   AGE
multi-completion-batch-job-85xjp   1/1     Running             0          2m4s
multi-completion-batch-job-hj96w   0/1     Completed           0          4m15s
multi-completion-batch-job-l8r8g   0/1     ContainerCreating   0          0s
multi-completion-batch-job-lcjz8   0/1     Completed           0          2m8s
multi-completion-batch-job-mwqng   0/1     Completed           0          4m15s
inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get pod
NAME                               READY   STATUS      RESTARTS   AGE
multi-completion-batch-job-85xjp   0/1     Completed   0          2m53s
multi-completion-batch-job-hj96w   0/1     Completed   0          5m4s
multi-completion-batch-job-l8r8g   1/1     Running     0          49s
multi-completion-batch-job-lcjz8   0/1     Completed   0          2m57s
multi-completion-batch-job-mwqng   0/1     Completed   0          5m4s
```

잡이 실행되는 동안에도 스케일링할 수 있다.
책에서는 `kubectl scale job <JOB> --replicas=<VALUE>`를 통해 할 수 있다고 하지만, deprecated 되었기 때문에 `kubectl edit`를 통해 진행한다.
```sh
inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl edit job <JOB>
inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get pod
NAME                               READY   STATUS              RESTARTS   AGE
multi-completion-batch-job-6dvh9   1/1     Running             0          72s
multi-completion-batch-job-jgwjm   0/1     ContainerCreating   0          4s
multi-completion-batch-job-jhkc7   0/1     Completed           0          3m22s
multi-completion-batch-job-qt9b5   0/1     Completed           0          3m22s
multi-completion-batch-job-sjtlt   1/1     Running             0          75s
inhyeok@ubuntu:~/study/kubernetes-in-action/Chapter04$ kubectl get pod
NAME                               READY   STATUS      RESTARTS   AGE
multi-completion-batch-job-6dvh9   1/1     Running     0          77s
multi-completion-batch-job-jgwjm   1/1     Running     0          9s
multi-completion-batch-job-jhkc7   0/1     Completed   0          3m27s
multi-completion-batch-job-qt9b5   0/1     Completed   0          3m27s
multi-completion-batch-job-sjtlt   1/1     Running     0          80s
```

- edit를 통해 `spec.parallelism`을 2에서 3으로 변경하면 즉시 새로운 파드가 생성되어 `ContainerCreating`상태가 되는 것을 확인할 수 있다.

- 파드가 특정 상태에 빠져서 완료되지 않는 경우 `spec.activeDeadlineSeconds` 속성을 통해 파드의 실행 시간을 제어할 수 있음
  - 이보다 오래걸리면 시스템이 종료를 시도하고 잡을 실패한 것으로 간주
  - `spec.backoffLimit` 필드를 통해 잡을 재시도하는 횟수를 지정. 기본값은 6

- 주기적으로 진행하려면 크론잡(CronJob) 리소스를 만들어 구성
  - 지정 시간/지정 간격으로 반복하는 작업을 리눅스/유닉스 OS에서는 크론작업이라 부름
    - 아래와 같이 설정
      ```
      *　　　　　　*　　　　　　*　　　　　　*　　　　　　*
      분(0-59)　　시간(0-23)　　일(1-31)　　월(1-12)　　　요일(0-7, 일월화수목금토일)
      ```

메니페스트:
```yaml
apiVersion: batch/v1 # batch/v1beta1에서 변경
kind: CronJob
metadata:
  name: batch-job-every-fifteen-minutes
spec:
  schedule: "0,15,30,45 * * * *" # 리눅스의 크론과 동일하게 사용: 매시간 0, 15, 30, 45분에 실행
  startingDeadlineSenconds: 15
  jobTemplate:
    spec:
      template:
        metadata:
          labels:
            app: periodic-batch-job
        spec:
          restartPolicy: OnFailure
          containers:
          - name: main
            image: luksa/batch-jo
```

- 이전 잡에는 보이지 않던 `spec.jobTemplate`가 보인다.
  - 이는 크론잡이 생성하는 잡에 대한 템플릿으로, 잡과 완전 동일한 구조를 갖고 있지만 중첩되어 있으며 `apiVersion`, `kind`와 같은 것이 없다.
- 잡이나 파드가 상대적으로 늦게 생성되거나 실행될 수 있는데, 이는 `spec.startingDeadlineSeconds` 필드를 지정하여 데드라인을 설정할 수 있음
  - 위의 예시에선 10:30:15까지 시작하지 않으면 잡이 실행되지 않고 실패로 표시
  - 일반적으론 설정 시간에 항상 하나의 잡만 생성하지만 여러 개의 잡이 동시에 생성되거나 전혀 생성되지 않을 수 있음
    - 여러 잡이 생성되는 경우 멱등성(한 번 시행이 아니라 여러 번 시행해도 원치 않는 결과가 초래되지 않음)을 가져야 함
    - 잡이 생성되지 않는 경우 다음 번 잡 실행이 이전에 누락된 실행에서 완료했어야 하는 작업을 수행하는지 확인

