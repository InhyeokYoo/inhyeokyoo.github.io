---
title:  "쿠버네티스 인 액션 정리: Ch05 서비스"
toc: true
toc_sticky: true
categories:
  - Docker
  - kubernetes
use_math: true
last_modified_at: 2023-03-21
---

다루는 내용:
- 단일 주소로 파드를 노출하는 서비스 리소스 만들기
- 클러스터 안에서 서비스 검색
- 외부 클라이언트에 서비스 노출
- 클러스터 내에서 외부 서비스 노출
- 파드가 서비스할 준비가 됐는지 제어
- 서비스 문제 해결

- 지금까지는 파드를 실행하도록 보장하고 파드를 배포함
- 이젠 애플리케이션을 통해 외부 요청에 응답하도록 함
- 파드가 다른 파드를 찾도록 할 것임
  - 파드는 일시적임. 즉, 노드에서 제거되거나 수가 줄거나 다른 노드로 이동이 빈번
  - 노드에 파드를 스케줄링한 후 파드가 시작되기 바로 전에 파드의 IP 주소를 할당하므로 클라이언트가 파드의 IP 주소를 미리 알 수 없음
  - 수평 스케일링을 통해 여러 파드가 생성됨. 각 파드마다 IP 주소가 다른데, 클라이언트는 서비스를 지원하는 파드의 수와 IP에 종속되지 않아야 하며, 모든 파드는 단일 IP 주소로 액세스 가능해야 함

포트관련: 전체 서비스 흐름으로 보면 NodePort --> Port --> targetPort 
- NodePort: 외부에서 접속하기 위해 사용하는 포트
- port: Cluster 내부에서 사용할 Service 객체의 포트
- targetPort: Service 객체로 전달된 요청을 Pod(deployment)로 전달할때 사용하는 포트

## 서비스란?

참고자료:
- https://blog.frec.kr/cloud/service_1
- https://seongjin.me/kubernetes-service-types/
- https://1week.tistory.com/60

- 서비스는 동일한 서비스를 제공하는 파드 그룹에 지속적인 단일 접점을 만들려고 할 때 생성하는 리소스
- 각 서비스는 서비스가 존재하는 동안 절대 바뀌지 않는 IP주소와 포트를 갖음
  - 클라이언트는 해당 IP와 포트로 접속하여 서비스를 지원하는 파드 중 하나로 연결됨 --> 위 문제 해결!

예시:
- FE 웹 서버와 BE DB 서버가 있고, FE 역할을 하는 파드는 여러 개가 있을 수 있지만 DB 파드는 하나만 있음
  - 웹 서버 개수에 상관없이 외부 클라이언트는 FE 파드에 연결해야 함
  - FE 파드는 DB 파드에 연결해야 하며, IP 주소가 변경될 수 있음
- FE 파드에 대한 서비스를 만들고 이를 클러스터 외부에서 엑세스 하도록 구성 --> 외부 클라이언트가 파드에 연결할 수 있는 하나의 고정 IP 주소 노출
- 똑같이 백엔드 파드에 관한 서비스를 생성해 안정적인 주소를 만듬
- 파드의 IP 주소가 변경되더라도 서비스 주소는 그대로 유지

## 서비스 생성
- 서비스에 파드를 연결하려면 레이블 셀렉터를 사용
- 이전에는 `kubectl expose`를 사용했는데, 이번에는 선언적인 방법으로 진행

매니페스트:
```yaml
apiVersion: v1
kind: Service # Service로 지정
metadata:
  name: kubia
spec:
  ports:
  - port: 80 # 서비스가 사용할 포트
    targetPort: 8080 # 서비스가 포워드 할 컨테이너 포트
  selector:
    app: kubia # 레이블에 해당하는 모든 파드가 이 서비스에 포함
```

- 서비스를 조회하면 아래와 같이 *ClusterIP*를 확인할 수 있다.
  - ClusterIP이므로 이는 클러스터 내부에서만 접근이 가능하다.

```sh
kubectl get svc
NAME         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1      <none>        443/TCP   3d2h
kubia        ClusterIP   10.100.56.46   <none>        80/TCP    43s
```

> ClusterIP는 파드들이 클러스터 내부의 다른 리소스들과 통신할 수 있도록 해주는 가상의 클러스터 전용 IP다. 이 유형의 서비스는 <ClusterIP>로 들어온 클러스터 내부 트래픽을 해당 파드의 <파드IP>:<targetPort>로 넘겨주도록 동작하므로, 오직 클러스터 내부에서만 접근 가능하게 된다. 쿠버네티스가 지원하는 기본적인 형태의 서비스다.

- 클러스터 내에서 서비스를 테스트하는 방법은 다음과 같음
  1. 서비스 ClusterIP로 요청을 보내고 응답으로 로그로 남기는 파드를 생성하고, 로그를 검사하여 서비스 응답이 무엇인지 확인
  2. 노드로 ssh접속하고 curl 명령
  3. `kubectl exec`로 기존 파드에서 curl 명령 실행
- 3번으로 진행해보자

- `kubectl exec`를 통해 기존 파드 컨테이너 내에서 원격으로 임의의 명령어 실행이 가능
  - 이를 통해 컨테이너 내용, 상태, 환경 등을 검사하는데 사용
  - `-c` 옵션을 통해 파드 내 컨테이너 선택 가능

```sh
kubectl get pod
NAME          READY   STATUS    RESTARTS       AGE
kubia-84544   1/1     Running   1 (3m2s ago)   15h
kubia-8pvzb   1/1     Running   1 (3m2s ago)   15h
kubia-ntfdg   1/1     Running   1 (3m2s ago)   15h
kubectl get svc
NAME         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1      <none>        443/TCP   3d18h
kubia        ClusterIP   10.100.56.46   <none>        80/TCP    15h
kubectl exec kubia-84544 -- curl -s http://10.100.56.46
You've hit kubia-8pvzb
```

- 명령어의 더블 대시(--)는 kubectl 명령줄 옵션의 끝을 의미
  - 더블 대시 이하는 파드 내에서 실행
  - 뒤 `curl` 명령어에 대시명령어(`-s`)가 있으므로, 이와 구분하기 위함

- 위 명령어의 동작 방식 이해:
  1. k8s에 `kubectl exec`를 지시하여 kubia-84544 파드에 curl 명령어를 입력
  2. kubia-84544 파드에서 HTTP 요청을 서비스 IP로 전송
  3. 서비스에는 3개의 파드가 연결되어 있으므로 k8s 서비스 프록시가 이 중 임의의 파드에 요청 전달
  4. 요청 받은 파드 내에서 실행 중인 컨테이너에서 요청을 처리 (해당 파드의 이름을 포함하는 HTTP 응답 반환)
      - 이때 요청 받은 파드는 임의의 파드이므로 매 실행마다 달라질 수 있음
  5. curl이 표준 출력으로 응답을 출력하고 이를 kubectl이 있는 로컬 시스템 표준 출력에 다시 표시

[세션 어피니티 (session affinity)](https://kubernetes.io/docs/reference/networking/virtual-ips/#session-affinity):
- 앞선 예시와 같이 **임의의 파드가 아닌 모든 요청을 같은 파드로 리디렉션**하려면 세션 어피니티 속성(`spec.sessionAffinity`)을 기본값 None 대신 ClinetIP로 설정
- None과 ClientIP 두 가지만 설정 가능

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia
spec:
  sessionAffinity: ClientIP # sessionAffinity 추가
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

- 동일한 서비스에서 여러 개의 포트 노출을 지원할 수 있음
- 따라서 포트만을 위해 여러 개의 서비스를 만들 필요는 없음
- 여러 포트를 만들 때는 각 포트의 이름을 지정해야 함
- 레이블 셀렉터는 서비스 전체에 적용되며, 각 포트를 개별적으로 구성할 수 없다
  - 이 경우 서비스를 두 개 만드는 것이 필요하다

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia
spec:
  ports:
  - name: http # 포트 80 -> 8080
    port: 80
    targetPort: 8080
  - name: https # 포트 443 -> 8443
    port: 443
    targetPort: 8443
  selector:
    app: kubia
```

- 대상 포트를 번호가 아닌 이름으로 참조할 수 있음
- 이를 통해 서비스 스펙을 좀 더 명확하게 할 수 있음
  - 포트가 변경되는 경우, 서비스 스펙을 변경하지 않고 파드 스펙에서 포트 번호를 변경하기만 하면 됨
- 파드 메니페스트:
  - 
    ```yaml
    kind: pod
    spec:
      containers:
      - name: kubia
        ports:
        - name: http # 컨테이너 포트 8080 이름을 http로 정의
          containerPort: 8080
        - name: https # 컨테이너 포트 8443 이름을 https로 정의
          containerPort: 8443
    ```
- 서비스 메니페스트:
  - 
    ```yaml
    apiVersion: v1
    kind: Service
    spec:
      ports:
      - name: http
        port: 80
        targetPort: http # 이전에는 포트 번호를 썼는데, 이제는 이름으로 접근
      - name: https
        port: 443
        targetPort: https  # 이전에는 포트 번호를 썼는데, 이제는 이름으로 접근
    ```

### 서비스 검색

- 서비스를 만들면 파드에 접속할 수 있도록 안정적인 IP 주소와 포트를 생성한다.
- 근데 클라이언트 파드의 경우는 어떻게 하는가?
  - 이 경우 서비스를 검색하여 서비스의 IP/포트를 검색하도록 한다

환경변수를 통한 서비스 검색
- 파드가 시작할경우 k8s에서 각 서비스를 가리키는 환경변수 셋을 초기화
  - 따라서 클라이언트 파드보다 서비스를 먼저 생성하면 파드의 프로세스로 하여금 환경변수를 검사하여 서비스의 IP/포트를 알아낼 수 있음
- 환경변수는 `kubectl exec`를 통해 확인할 수 있음

```sh
kubectl exec kubia-mx5ww -- env
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOSTNAME=kubia-mx5ww
KUBIA_PORT=tcp://10.110.194.90:80
KUBIA_SERVICE_HOST=10.110.194.90 # 서비스의 클러스터 IP
KUBIA_PORT_80_TCP_PROTO=tcp
KUBIA_PORT_80_TCP_PORT=80
KUBIA_PORT_80_TCP_ADDR=10.110.194.90
KUBERNETES_SERVICE_PORT=443
KUBIA_SERVICE_PORT=80  # 서비스가 제공되는 포트
KUBERNETES_SERVICE_HOST=10.96.0.1
KUBERNETES_SERVICE_PORT_HTTPS=443
KUBERNETES_PORT_443_TCP=tcp://10.96.0.1:443
KUBERNETES_PORT_443_TCP_PROTO=tcp
KUBERNETES_PORT_443_TCP_ADDR=10.96.0.1
KUBIA_PORT_80_TCP=tcp://10.110.194.90:80
KUBERNETES_PORT=tcp://10.96.0.1:443
KUBERNETES_PORT_443_TCP_PORT=443
NPM_CONFIG_LOGLEVEL=info
NODE_VERSION=7.9.0
YARN_VERSION=0.22.0
HOME=/root
```

- 클러스터 내에는 kubernetes 서비스와 kubia 서비스 두 개가 존재
  - `kubia get svc`로도 확인했었음
- 이전의 FE/BE 예시에서 생각해보자:
  - FE 파드는 BE의 DB 서버 파드에 접속한다
  - 이때 이 DB를 `backend-database`라는 서비스로 BE 파드를 노출한다고 하자
  - 그러면 FE 파드에서 환경변수 `BACKEND_DATABASE_SERVICE_HOST`와 `BACKEND_DATABASE_SERVICE_PORT`를 통해 DB 서비스의 IP주소와 포트를 찾을 수 있다
- 이름을 보면 알겠지만 환경변수의 네이밍은 대문자로 이루어져있으며, dash(-)는 underbar(_)로 변환되며, 서비스 이름이 환경변수의 접두어가 된다

DNS를 통한 서비스 검색
- 환경변수를 쓰는 대신 DNS 도메인을 사용하는 것이 일반적일 것이다
- 따라서 k8s에 서버를 포함하고 DNS를 통해 서비스 IP를 찾게하는 방법을 고려해볼 수 있다
  - DNS: 도메인 이름과 IP 주소를 서로 변환하는 역할
- `kube-system` 네임스페이스 안에 `kube-dns`라는 파드가 DNS 서버를 실행하며, 클러스터에 실행 중인 모든 파드가 이를 사용하도록 구성
  - k8s는 각 컨테이너의 `/etc/resolv.conf` 파일을 수정하여 이를 수행
- 파드 내 실행 중인 프로세스에서 수행된 모든 DNS 쿼리는 시스템에서 실행 중인 모든 서비스를 알고 있는 k8s의 자체 DNS 서버로 처리
  - 파드가 내부 DNS 서버를 사용할지 여부는 파드 스펙의 `dnsPolicy`를 통해 구성 가능
- 각 서비스는 내부 DNS 서버에서 DNS 항목을 가져오고, 서비스 이름을 알고 있는 클라이언트 파드는 환경변수 대신 FQDN으로 엑세스가 가능

FQDN을 통한 서비스 연결
- FQDN (Fully Qualified Domain Name): 명확한 도메인 표기법을 칭한다. 예로 소프트웨어 설치 중 도메인명을 요구하면, YAHOO.COM. 을 입력할지, WWW.YAHOO.COM. 을 입력할지 모호하다. 그래서 이러한 모호성을 피하기 위해 FQDN이란 단어를 사용하며, 이는 Namespace 계층상에서 최종 호스트명을 포함하는 도메인명을 뜻한다. www(호스트명), yahoo.com.(도메인명), www.yahoo.com.(FQDN)
- 앞선 예제에서 FE 파드는 다음 FQDN으로 BE의 DB 서비스에 접속할 수 있다
  - `backend-database.default.svc.cluster.local`
  - `backend-database`: 서비스 이름
  - `default`: 서비스가 정의된 네임스페이스
  - `svc.cluster.local`: 모든 클러스터의 로컬 서비스 이름에 사용되는 클러스터의 도메인 접속사
- 그러나 이 경우에도 클라이언트는 여전히 서비스의 포트 번호가 필요
  - 표준 포트(HTTP의 80, Postgres의 5432)를 사용하는 경우는 괜찮
  - 그 외의 경우엔 클라이언트가 환경변수에서 포트 번호를 얻어야 함
- FE 파드가 DB 파드와 동일 네임스페이스에 있는 경우 `svc.cluster.local` 접미사와 네임스페이스를 생략하여 `backend-database`라 할 수 있음

- 그럼 이제 FQDN으로 kubia 서비스에 연결해보자
- 이를 위해서는 `kubectl exec`를 통해 bash 셀을 실행하여 컨테이너에 접근해야 한다

```sh
 kubectl exec -it kubia-mx5ww -- bash

root@kubia-mx5ww:/# curl http://kubia.default.svc.cluster.local
You've hit kubia-mx5ww

root@kubia-mx5ww:/# curl http://kubia.default                  
You've hit kubia-mx5ww

root@kubia-mx5ww:/# curl http://kubia        
You've hit kubia-mx5ww
```

- 앞서 설명했던 것 처럼 `svc.cluster.local` 접미사와 네임스페이스를 생략하여 접속하는 것을 확인할 수 있다
- 컨테이너에서 `/etc/resolv.conf`를 확인해보면 이해할 수 있다 

```sh
root@kubia-mx5ww:/# cat /etc/resolv.conf
nameserver 10.96.0.10
search default.svc.cluster.local svc.cluster.local cluster.local localdomain
options ndots:5
```

- 만약 서비스에 엑세스 할 수 없다면?
  - 일반적으로는 위와 같이 기존 파드를 입력하고 서비스에 엑세스 하여 문제를 파악할 것이다
  - 그 후 curl을 통해 액세스할 수 없다면 핑을 날려 동작 여부를 확인할 것이다
- 그러나 핑을 날릴 경우 아래와 같이 응답이 없다
  - 서비스의 ClusterIP가 가상 IP 이므로 서비스 포트와 결합된 경우에만 의미가 있기 때문

```sh
root@kubia-mx5ww:/# ping kubia
PING kubia.default.svc.cluster.local (10.110.194.90): 56 data bytes
^C--- kubia.default.svc.cluster.local ping statistics ---
162 packets transmitted, 0 packets received, 100% packet loss
```

## 클러스터 외부에 있는 서비스 연결

- 지금까지는 클러스터 내부에서 실행 중인 파드와의 통신을 지원하는 서비스만 살펴봄
- 이제부턴 외부 서비스를 노출하는 경우를 살펴봄
  - 즉, 이전처럼 서비스가 클러스터 내 파드로 연결을 전달하는 대신 외부 IP와 포트로 연결을 전달
- 이 경우 서비스 로드밸런싱과 검색 모두 활용이 가능

### 서비스 엔드포인트(Endpoint)

> In the Kubernetes API, an Endpoints (the resource kind is plural) defines a list of network endpoints, typically referenced by a Service to define which Pods the traffic can be sent to.
> 
> The EndpointSlice API is the recommended replacement for Endpoints.

> 엔드포인트(endpoint): 서비스를 사용가능하도록 하는 서비스에서 제공하는 커뮤니케이션 채널의 한쪽 끝, 즉 요청을 받아 응답을 제공하는 서비스를 사용할 수 있는 지점을 의미

- 사실 서비스는 파드에 직접 연결되는 대신 엔드포인트 리소스를 통해 파드에 연결됨
  - 엔드포인트: 서비스가 트래픽을 전달하고자 하는 파드의 집합
  - 서비스는 엔드포인트에 매핑된 파드의 IP정보를 가지고 파드에게 트래픽을 전달
  - 새로 추가된 파드가 서비스의 라벨을 달고있다면 실제로는 엔드포인트에 해당 파드의 IP가 추가됨으로써 이러한 동작을 하게함
- `kubectl describe`을 통해 이를 확인 가능

```sh
kubectl describe svc kubia
Name:              kubia
Namespace:         default
Labels:            <none>
Annotations:       <none>
Selector:          app=kubia # 서비스의 파드 셀렉터는 엔드포인트 목록을 만드는 데 사용
Type:              ClusterIP
IP Family Policy:  SingleStack
IP Families:       IPv4
IP:                10.110.194.90
IPs:               10.110.194.90
Port:              <unset>  80/TCP
TargetPort:        8080/TCP
Endpoints:         10.244.0.29:8080,10.244.0.30:8080,10.244.0.31:8080 # 서비스의 엔드포인트를 나타내는 파드 IP 및 포트의 목록
Session Affinity:  ClientIP
Events:            <none>
```

- 엔드포인트 리소스는 서비스로 노출되는 파드의 IP 주소와 포트 목록을 의미
- 이는 다른 리소스와 비슷하므로 마찬가지로 `kubectl get endpoints <SERVICE>`을 통해 불러올 수 있음

```sh
kubectl get endpoints kubia
NAME    ENDPOINTS                                            AGE
kubia   10.244.0.29:8080,10.244.0.30:8080,10.244.0.31:8080   13d
```

- 파드 셀렉터는 서비스 스펙에 정의되어 있으나 들어오는 연결을 전달할 때 직접 사용하지 않음
  - 대신 IP와 포트 목록을 작성하며, 엔드포인트 리소스에 저장됨
- 클라이언트가 서비스에 연결하면 서비스 프록시는 이 중 하나의 IP와 포트 쌍을 선택하고, 들어온 연결을 대상 파드의 수신 대기 서버로 전달

### 서비스 엔드포인트 수동으로 구성하기

- 서비스의 엔드포인트를 서비스와 분리하면 이를 수동으로 구성하고 업데이트가 가능
- 파드 셀렉터 없이 서비스를 만들면 k8s는 엔드포인트 리소스를 만들지 못함
  - 파드 셀렉터가 없기 때문에 어느 파드가 서비스에 포함되는지 알 수가 없음
- 수동으로 관리되는 엔드포인트를 사용해 서비스를 만들려면 서비스와 엔드포인트 리소스를 모두 만들어야 함

셀렉터 없이 서비스 생성:
- 간단하게 포트 80으로 들어오는 `external-service`라는 서비스를 정의해봄
```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-service  # 서비스 이름은 엔드포인트 오브젝트 이름과 일치해야 함
spec: # 파드 셀렉터가 없음
  ports:
  - port: 80
```

셀렉터가 없는 서비스에 관한 엔드포인트 리소스 생성
- 엔드포인트는 서비스 속성이 아니라 별도의 리소스
- 이전에 셀렉터가 없는 서비스를 만들었으므로 엔드포인트 리소스가 자동으로 생성되지 않음
- 다음의 메니페스트를 통해 엔드포인트 리소스를 생성해보자

```yaml
apiVersion: v1
kind: Endpoints
metadata:
  name: external-service # 엔드포인트의 이름은 서비스 이름과 같아야함 (위)
subsets:  
  - addresses:  # 서비스가 연결을 전달할 엔드포인트의 IP
    - ip: 11.11.11.11
    - ip: 22.22.22.22
    ports:
    - port: 80 # 엔드포인트의 대상 포트
```

- 위 처럼 엔드포인트는
  1. 서비스와 이름이 같아야 하고
  2. 서비스를 제공하는 대상 IP 주소와 포트 목록을 가져야 함
- 서비스와 엔드포인트 리소스가 모두 서버에 올라가면 파드 셀렉터가 있는 일반 서비스처럼 사용할 수 있음
- 서비스가 만들어진 후 만들어진 컨테이너 안에는 서비스의 환경변수가 포함
- (IP, 포트) 쌍에 대한 모든 연결은 서비스 엔드포인트 간에 로드밸런싱함

### 외부 서비스를 위한 별칭 생성

- 엔드포인트를 수동으로 구성하여 외부 서비스를 노출하는 대신 FQDN을 통해 외부 서비스 참조가 가능

ExternalName 서비스 생성
- 외부 서비스의 별칭으로 사용되는 서비스를 만드려면 type 필드를 `ExternalName`으로 설정하여 서비스 리소스를 만듬

예시: `external-service-externalname.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-service
spec:
  type: ExternalName  # 서비스 유형이 ExternalName으로 설정
  externalName: api.somecompany.com # FQDN
  ports:
  - port: 80
```

- 파드는 서비스의 FQDN을 사용하는 대신 `external-service.default.svc.cluster.local` 도메인으로 외부 서비스에 연결이 가능
  - 서비스를 사용하는 파드에서 실제 서비스 이름과 위치가 숨겨져있음
  - 추후 `externalName` 속성을 변경하거나 유형을 다시 ClusterIP로 변경하고 서비스 스펙을 만들어 서비스 스펙을 수정하면 추후 다른 서비스를 가리킬 수 있음
- DNS 레벨에서만 구현되며 서비스에 관한 간단한 CNAME DNS 레코드가 생성됨
- 따라서 서비스에 연결하는 클라이언트는 서비스 프록시를 무시하고 외부 서비스에 직접 연결할 수 있음
- 이러한 이유로 `ExternalName` 유형의 서비스는 ClusterIP를 얻지 못함

## 외부 클라이언트에 서비스 노출

- 특정 서비스를 외부에 노출해 외부 클라이언트가 엑세스하려면?
  1. 노드포트(NodePort)로 서비스 유형 설정: 노드포트 서비스의 경우 각 클러스터 노드는 노드 자체에서 포트를 열고 해당 포트로 수신된 트래픽을 서비스로 전달함. 이 서비스는 내부 클러스터 IP와 포트로 액세스할 수 있을 뿐만 아니라 모든 노드의 전용 포트로도 액세스할 수 있음
  2. 서비스 유형을 노드포트 유형의 확장인 로드밸런서로 설정: k8s가 실행 중인 클라우드 인프라에서 프로비저닝된 전용 로드밸런서로 서비스에 엑세스가 가능. 로드밸런서는 트래픽을 모든 노드의 노드포트로 전달함. 클라이언트는 로드밸런서의 IP로 서비스에 엑세스가 가능
  3. 단일 IP 주소로 여러 서비스를 노출하는 인그레스 리소스 만들기: HTTP 레벨(네트워크 7계층)에서 작동하므로 4계층 서비스보다 더 많은 기능 제공이 가능.

### 1. 노드포트(NodePort)로 서비스 유형 설정

> NodePort는 외부에서 노드 IP의 특정 포트(`<NodeIP>:<NodePort>`)로 들어오는 요청을 감지하여, 해당 포트와 연결된 파드로 트래픽을 전달하는 유형의 서비스다. 이때 클러스터 내부로 들어온 트래픽을 특정 파드로 연결하기 위한 ClusterIP 역시 자동으로 생성된다.

> A NodePort service is the most primitive way to get external traffic directly to your service. NodePort, as the name implies, opens a specific port on all the Nodes (the VMs),  and any traffic that is sent to this port is forwarded to the service.

- 노드포트 서비스를 만들면 k8s는 모든 노드에 특정 포트를 할당하고(즉, 모든 노드에서 동일한 포트 번호가 사용됨) 서비스를 구성하는 파드로 들어오는 연결을 전달
- 일반 서비스와 유사하지만 서비스의 내부 ClusterIP뿐만 아니라 모든 노드의 IP와 할당된 노드포트로 서비스에 액세스가 가능

생성: `kubia-svc-nodeport.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia-nodeport
spec:
  type: NodePort # 서비스 유형이 NodePort가 됨
  ports:
  - port: 80  # 서비스 내부 ClusterIP의 포트
    targetPort: 8080  # 서비스 대상 파드의 포트
    nodePort: 30123 # 각 클러스터 노드의 포트 30123으로 서비스에 엑세스할 수 있음
  selector:
    app: kubia
```

- 유형을 노드포트로 설정하고 서비스가 모든 클러스터 노드에 바인딩돼야 하는 노드 포트를 지정함
  - 노드포트를 반드시 지정할 필요는 없음. 생략시 k8s가 알아서 포트를 선택함

```sh
kubectl get svc kubia-nodeport
NAME             TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
kubia-nodeport   NodePort   10.111.131.47   <none>        80:30123/TCP   2s
```

- EXTERNAL-IP열에는 <nodes>라 표시되어있고, 클러스터 노드의 IP 주소로 서비스에 액세스할 수 있다는 뜻이다
  - 책에는 <nodes>라고 표시되어 있다고 하는데, 실제로는 <none>으로 나온다
- PORT(S)에는 클러스터 IP의 내부포트(80)과 노드포트(30123)이 모두 표시된다.
- 본 서비스를 액세스할 수 있는 주소로는 다음과 같다
  - 10.111.131.47:80
  - <첫 번째 노드의 IP>:30123
  - <두 번째 노드의 IP>:30123 등
- 노드포트로 서비스에 접속하려면 이에 대한 외부 연결을 허용하도록 GCP의 방화벽을 구성해야 함
  - minikube의 경우 `minikube start`를 실행하여 브라우저로 노드포트에 액세스하면 됨

- 어떤 노드든 30123번 포트로 파드에 엑세스가 가능
- 그러나 클라이언트가 첫 번째 노드에만 요청하면 해당 노드가 장애가 날 경우 더 이상 서비스에 액세스가 불가능
- 따라서 노드 앞에 로드밸런서를 배치하여 모든 노드에 요청을 분산시키고 해당 시점에 오프라인 상태인 노드로 요청을 보내지 않도록 하는 것이 좋음

### 2. 서비스 유형을 노드포트 유형의 확장인 로드밸런서로 설정

- 클라우드를 통해 실행되는 k8s 클러스터는 일반적으로 클라우드 인프라에서 로드밸런서를 자동으로 *프로비저닝*하는 기능을 제공
  - 프로비저닝(provisioning): 사용자의 요구에 맞게 시스템 자원을 할당, 배치, 배포해 두었다가 필요 시 시스템을 즉시 사용할 수 있는 상태로 미리 준비해 두는 것
- 노드포트 대신 서비스 유형을 로드밸런서로 설정하면 끝
- 로드밸런서는 공개적으로 액세스 가능한 고유한 IP주소를 가지며, 모든 연결을 서비스로 전달하므로 로드밸런서의 IP 주소를 통해 서비스에 액세스가 가능
- 만일 로드밸런서를 지원하지 않는 환경이라면 로드밸런서가 프로비저닝 되진 않지만 노드포트 서비스처럼 작동함

서비스생성:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia-loadbalancer
spec:
  type: LoadBalancer 
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

- minikube를 쓰는 경우 로드밸런서 서비스를 지원하지 않음
  - 새로운 터미널에서 `minikube tunnel`을 실행시키면 가능
  - 아니면 그냥 노트포트로 서비스에 액세스할 수 있음 (이 경우 로드밸런서 프로비저닝 X)

```sh
kubectl get svc # 로드밸런서가 pending
NAME                 TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
kubernetes           ClusterIP      10.96.0.1       <none>        443/TCP        18d
kubia                ClusterIP      10.110.194.90   <none>        80/TCP         14d
kubia-loadbalancer   LoadBalancer   10.104.74.139   <pending>     80:31084/TCP   3s
kubia-nodeport       NodePort       10.111.131.47   <none>        80:30123/TCP   4h54m
```

```sh
 minikube tunnel
Status:
        machine: minikube
        pid: 1584567
        route: 10.96.0.0/12 -> 192.168.49.2
        minikube: Running
        services: [kubia-loadbalancer]
    errors: 
                minikube: no errors
                router: no errors
                loadbalancer emulator: no errors
```

```sh
kubectl get svc # IP 할당이 성공적으로 됨
NAME                 TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)        AGE
kubernetes           ClusterIP      10.96.0.1       <none>          443/TCP        18d
kubia                ClusterIP      10.110.194.90   <none>          80/TCP         14d
kubia-loadbalancer   LoadBalancer   10.104.74.139   10.104.74.139   80:31084/TCP   65s
kubia-nodeport       NodePort       10.111.131.47   <none>
curl 10.104.74.139 # curl을 통해 확인하는 모습
You've hit kubia-p4wkd
```

- 로드밸런서를 이용하는 경우 방화벽을 설정할 필요가 없음
- 현재 서비스가 노출되어 있으므로 웹 브라우저를 통해 서비스에 액세스가 가능함
  - 브라우저를 통해 액세스할 경우 매번 같은 파드만 노출되지만 curl을 쓸 경우 파드가 변함
  - 이는 앞서 살펴본 세션 어피니티와 관련된 내용
  - 그러나 `kubectl describe`를 살펴봐도 세션 어피니티는 설정되지 않음
    ```sh
    kubectl describe svc kubia-loadbalancer
    Name:                     kubia-loadbalancer
    Namespace:                default
    Labels:                   <none>
    Annotations:              <none>
    Selector:                 app=kubia
    Type:                     LoadBalancer
    IP Family Policy:         SingleStack
    IP Families:              IPv4
    IP:                       10.104.74.139
    IPs:                      10.104.74.139
    LoadBalancer Ingress:     10.104.74.139
    Port:                     <unset>  80/TCP
    TargetPort:               8080/TCP
    NodePort:                 <unset>  31084/TCP
    Endpoints:                10.244.0.29:8080,10.244.0.30:8080,10.244.0.31:8080
    Session Affinity:         None # 세션 어피니티가 비어있음
    External Traffic Policy:  Cluster 
    Events:                   <none>
    ```
- 브라우저는 keep-alive 연결을 사용하고 같은 연결로 모든 요청을 보내는 반면 curl은 매번 새롭게 연결
- 서비스는 연결 수준에서 작동하고 처음 서비스 연결하면 임의의 파드가 선택된 이후 해당 연결에 속하는 모든 패킷을 같은 파드로 전송하기 때문

- 로드밸런서 type service는 추가 인프라 제공 로드밸런서가 있는 노드포트 서비스
  - kubectl describe를 통해 이를 잘 확인할 수 있음
  - 이전 노드포트 예제와 같이 노드포트 서비스에 대한 방화벽을 열어 노드 IP로도 접근이 가능


```sh
kubectl describe svc kubia-loadbalancer
Name:                     kubia-loadbalancer
Namespace:                default
Labels:                   <none>
Annotations:              <none>
Selector:                 app=kubia
Type:                     LoadBalancer
IP Family Policy:         SingleStack
IP Families:              IPv4
IP:                       10.104.74.139
IPs:                      10.104.74.139
Port:                     <unset>  80/TCP
TargetPort:               8080/TCP
NodePort:                 <unset>  31084/TCP # 노드포트가 설
Endpoints:                10.244.0.29:8080,10.244.0.30:8080,10.244.0.31:8080
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
```

### 외부 연결의 특성 이해

불필요한 네트워크 *홉*의 이해와 예방
- 홉(hop): 컴퓨터 네트워크에서 출발지와 목적지 사이에 위치한 경로의 한 부분. 데이터 패킷은 브리지, 라우터, 게이트웨이를 거치면서 출발지에서 목적지로 경유. 패킷이 다음 네트워크 장비로 이동할 때마다 홉이 하나 발생.
- 외부 클라이언트가 노드포트로 서비스에 접속하는 경우 (로드밸런서 유무에 상관없음) 파드가 연결을 수신한 동일한 노드에서 실행 중일 수 있음
- 따라서 파드에 도달하려면 추가적인 네트워크 홉이 필요할 수도 있음
  - 이것이 항상 바람직하진 않음
- 이를 방지하려면 외부 연결을 수신한 노드 내에서 실행 중인 파드로만 외부 트래픽을 전달하도록 함
  - 서비스의 스펙 섹션의 `externalTraficPolicy` 필드를 설정
- 서비스 정의에 이를 설정 후 노드포트로 외부 연결이 열리면 서비스 프록시는 로컬에 실행 중인 파드를 선택
  - 로컬 파드가 없으면 연결이 중단
- 따라서 로드밸런서는 이러한 파드가 하나 이상 있는 노드에만 연결을 전달하도록 해야 함
- `externalTraficPolicy`를 사용할 경우 연결이 모든 파드에 균등하게 분산되지 않는다는 단점도 있음
- 노드 A/B가 있고, A 노드에 파드가 1개, B 노드에 파드가 2개 있다고 가정하자
  - 그러면 로드밸런서가 연결의 50% 씩 각 노드로 전달
  - A 노드에 있는 하나의 파드에는 50%의 연결이 수신
  - B 노드의 있는 두개의 파드에 50%의 연결이 수신되어 각 파드 당 25%의 연결을 수신하게 됨

클라이언트 IP가 보존되지 않음 인식
- 클러스터 내 클라이언트가 서비스로 연결할 때 파드는 클라이언트의 IP 주소를 얻을 수 있음
- 그러나 노드포트로 연결을 수신하면 패킷에서 소스 네트워크 주소변환(SNAT)이 일어나 패킷의 소스 IP가 변경
- 따라서 파드가 실제 클라이언트 IP를 볼 수 없음
  - 웹 서버처럼 엑세스 로그에 브라우저 IP를 표시하는 기능 등에 문제가 발생
- 앞서 살펴본 Local External Traffic Policy는 연결을 수신하는 노드와 대상 파드를 호스팅하는 노드 사이에 추가 홉이 없음
  - 따라서 클라이언트 IP보존에 영향을 미침 (SNAT가 수행되지 않음)

## 인그레스 리소스로 서비스 외부 노출

- 외부로 노출하는 방법 중 세번째 방법
- 인그레스(ingress): 인그레스는 클러스터 외부에서 클러스터 내부 서비스로 HTTP와 HTTPS 경로를 노출한다. 트래픽 라우팅은 인그레스 리소스에 정의된 규칙에 의해 컨트롤된다.
- Unlike all the above examples, Ingress is actually NOT a type of service. Instead, it sits in front of multiple services and act as a “smart router” or entrypoint into your cluster. There are many types of Ingress controllers, from the Google Cloud Load Balancer, Nginx, Contour, Istio, and more.

왜 인그레스가 필요한가?
- 로드밸런서 서비스는 자신의 공용 IP 주소를 가진 로드밸런서가 필요한 반면, 인그레스는 한 IP 주소로 수십 개의 서비스에 접근이 가능하도록함
- 클라이언트가 HTTP 요청을 인그레스에 보낼 때, 요청한 호스트와 경로에 따라 요청을 전달할 서비스가 결정
- 인그레스는 네트워크 스택의 애플리케이션 계층(HTTP)에서 작동
- 서비스가 할 수 없는 쿠키 기반 세션 어피니티와 같은 기능도 제공

인그레스 컨트롤러가 필요한 경우
- 인그레스를 작동시키려면 클러스터에 인그레스 컨트롤러를 실행해야 함
- minikube의 경우 에드온을 통하여 따로 활성화 시켜줘야 함

```sh
 minikube addons list # addon 확인
|-----------------------------|----------|--------------|--------------------------------|
|         ADDON NAME          | PROFILE  |    STATUS    |           MAINTAINER           |
|-----------------------------|----------|--------------|--------------------------------|
| ingress                     | minikube | disabled     | Kubernetes                     |
| ingress-dns                 | minikube | disabled     | Google                         |
|-----------------------------|----------|--------------|--------------------------------|
 minikube addons enable ingress # ingress addon 활성화
💡  ingress is an addon maintained by Kubernetes. For any concerns contact minikube on GitHub.
You can view the list of minikube maintainers at: https://github.com/kubernetes/minikube/blob/master/OWNERS
    ▪ Using image registry.k8s.io/ingress-nginx/kube-webhook-certgen:v20220916-gd32f8c343
    ▪ Using image registry.k8s.io/ingress-nginx/kube-webhook-certgen:v20220916-gd32f8c343
    ▪ Using image registry.k8s.io/ingress-nginx/controller:v1.5.1
🔎  Verifying ingress addon...
🌟  The 'ingress' addon is enabled
 kubectl get po --all-namespaces # 인그레스 컨트롤러가 파드로 기동되므로 조회해보자
NAMESPACE       NAME                                       READY   STATUS      RESTARTS        AGE
default         kubia-mx5ww                                1/1     Running     1 (2m34s ago)   42h
default         kubia-p4wkd                                1/1     Running     1 (2m34s ago)   42h
default         kubia-x54zp                                1/1     Running     1 (2m34s ago)   42h
ingress-nginx   ingress-nginx-admission-create-s4ffl       0/1     Completed   0               86s
ingress-nginx   ingress-nginx-admission-patch-f968j        0/1     Completed   0               86s
ingress-nginx   ingress-nginx-controller-77669ff58-dldf5   1/1     Running     0               86s
kube-system     coredns-787d4945fb-d8bct                   1/1     Running     6 (10h ago)     18d
kube-system     etcd-minikube                              1/1     Running     6 (10h ago)     18d
kube-system     kube-apiserver-minikube                    1/1     Running     6 (10h ago)     18d
kube-system     kube-controller-manager-minikube           1/1     Running     6 (10h ago)     18d
kube-system     kube-proxy-85kfh                           1/1     Running     6 (10h ago)     18d
kube-system     kube-scheduler-minikube                    1/1     Running     6 (10h ago)     18d
kube-system     storage-provisioner                        1/1     Running     10 (103s ago)   18d
```

### 인그레스 리소스 생성하기

- 책에서 나오는 예제는 실행되지 않으므로 수정하자

```yaml
# 책에 나온 버전은 오래되어 아래와 같이 수정해야 함
# apiVersion: networking.k8s.io/v1
# kind: Ingress
# metadata:
#   name: kubia
# spec:
#   rules:
#   - host: kubia.example.com # 서비스에 맵핑할 도메인 이름
#     http:
#       paths:  # 모든 요청은 kubia-nodeport 서비스의 포트 80으로 전달
#       - path: /
#         backend:
#           serviceName: kubia-nodeport
#           servicePort: 80
apiVersion: networking.k8s.io/v1 # extensions/v1beta1 에서 변경
kind: Ingress
metadata:
  name: kubia
spec:
  rules:
  - host: kubia.example.com # 서비스에 맵핑할 도메인 이름
    http:
      paths:  # 모든 요청은 kubia-nodeport 서비스의 포트 80으로 전달
      - path: /
        pathType: Prefix  # 새롭게 추가
        backend:
          service:   # ServiceName, ServicePort를 하나로 묶음
            name: kubia-nodeport
            port:
              number: 80
```

- 위 인그레스는 Host kubia.example.com으로 요청되는 인그레스 커트롤러에 수신되는 모든 HTTP 요청을 80번 포트의 kubia-nodeport 서비스로 전송한다
- GKE같은 클라우드 공급자의 인그레스 컨트롤러는 인그레스가 노드포트 서비스를 가리킬 것을 요구하지만 k8s 자체의 요구 사항은 아님

### 인그레스 서비스 액세스하기

- kubia.example.com에 접속하려면 도메인 이름이 인그레스 컨트롤러의 IP와 맵핑되야 함

```sh
kubectl get ingress
NAME    CLASS   HOSTS               ADDRESS        PORTS   AGE
kubia   nginx   kubia.example.com   192.168.49.2   80      80s
```

- ADDRESS에 표시된게 바로 IP이다
- 이후 해당 IP로 확인하도록 DNS 서버를 구성하거나 `/etc/hosts`에 다음을 추가한다
  - 192.168.49.2  kubia.example.com
  - TODO: DNS 서버 구성은 어떻게..?
- 이후 curl을 날려보면 잘 작동하는 것을 확인

```sh
curl http://kubia.example.com
You've hit kubia-mx5ww
```

- 인그레스의 동작 방식은 아래와 같음
  1. 클라이언트가 kubia.example.com를 찾음  (DNS 조회)
  2. DNS 서버(로컬 운영체제)가 IP를 반환
  3. 클라이언트가 인그레스 컨트롤러에 헤더 Host:kubia.example.com과 함께 HTTP GET 요청을 보냄
  4. 인그레스 컨트롤러가 해당 헤더에서 클라이언트가 액세스하려는 서비스를 결정
  5. 이후 서비스와 연관된 엔드포인트로 파드 IP를 조회한 다음 클라이언트 요청을 파드에 전달
- 인그레스 컨트롤러가 서비스로 요청을 전달하는게 아니라 파드를 선택하는데만 사용함
  - 대부분의 컨트롤러도 이와 비슷하게 동작함

### 인그레스 하나로 여러 서비스 노출하기

- 동일한 호스트의 다른 경로에 대한 처리
- `.spec.ruls.host.http.paths` 필드에서 다르게 지정해줌

```yaml
...
spec:
  rules:
  - host: kubia.example.com
    http:
      paths: 
      - path: /kubia  # kubia.example.com/kubia에 대한 처리
        pathType: Prefix
        backend:
          service:
            name: kubia
            port:
              number: 80
      - path: /bar  # kubia.example.com/bar에 대한 처리
        pathType: Prefix
        backend:
          service:
            name: bar
            port:
              number: 80
```

- 다른 호스트에 대한 서비스 매핑
  - 경로 대신 호스트를 기반으로 다른 서비스를 맵핑 가능
  - 컨트롤러가 받은 요청은 호스트 헤더에 따라 foo 또는 bar로 전달
  - DNS는 foo.example.com과 bar.example.com 도메인 이름을 모두 인그레스 컨트롤러의 IP 주소로 지정해야 함

```yaml
spec:
  rules:
  - host: foo.example.com # foo.example.com에 대한 요청을 서비스 foo로 연결
    http:
      paths: 
      - path: /
        pathType: Prefix
        backend:
          service:
            name: foo
            port:
              number: 80
  - host: bar.example.com # bar.example.com에 대한 요청을 서비스 bar로 연결
    http:
      paths: 
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bar
            port:
              number: 80
```

### TLS 트래픽을 처리하는 인그레스

- HTTPS를 지원하도록 하자

TLS 인증서 생성
- 클라이언트가 인그레스 컨트롤러에 대한 TLS 연결을 하면 인그레스 컨트롤러가 TLS 연결을 종료 (TLS Termination)
- 클라이언트와 컨트롤러 간 통신은 암호화되지만, 컨트롤러와 백엔드 파드 간의 통신은 암호화가 되지 않음
- 파드에서 실행 중인 앱은 TLS를 지원할 필요가 없음
- 예를 들어 파드가 웹 서버를 실행할 때 HTTP 트래픽만 허용하고 인그레스 컨트롤러가 TLS 관련된 것을 처리하도록 만들 수 있음
  - 이럴러면 인그레스 컨트롤러가 TLS와 관련된 모든 것을 처리하도록 인증서와 개인키를 첨부해야함
  - 인증서와 개인키는 k8s 리소스의 시크릿(secret)에 저장

개인키와 인증서 생성
```sh
openssl genrsa -out tls.key 2048
openssl req -new -x509 -key tls.key -out tls.cert -days 360 -subj /CN=kubia.example.com
kubectl create secret tls tls-secret --cert=tls.cert --key=tls.key
secret/tls-secret created # 시크릿 생성
```

- 이후 다음과 같이 메니페스트를 업데이트함

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: kubia
spec:
  tls: # tls 지정
  - hosts: 
    - kubia.example.com
    secretName: tls-secret
  rules:
  - host: kubia.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kubia-nodeport
            port:
              number: 80
```

- 이전에 kubia ingress를 만들었으므로 이를 삭제하고 다시만듬
  - 번거로우니까 그냥 `kubectl apply -f kubia-ingress-tls.yaml`를 통해 업데이트할 수 있음

```sh
kubectl apply -f kubia-ingress-tls.yaml

Warning: resource ingresses/kubia is missing the kubectl.kubernetes.io/last-applied-configuration annotation which is required by kubectl apply. kubectl apply should only be used on resources created declaratively by either kubectl create --save-config or kubectl apply. The missing annotation will be patched automatically.
ingress.networking.k8s.io/kubia configured

curl -k -v https://kubia.example.com/kubia
...
You've hit kubia-x54zp
```

- 제대로 동작하는 것을 확인
- 출력에는 앱의 응답과 더불어 인그레스에 구성한 서버 인증서도 표시됨
- 인그레스에 대한 지원은 인그레스 컨트롤러 구현마다 서로 다르므로 확인하고 진행할 것

## 파드가 연결을 수락할 준비가 됐을 때 신호 보내기

- 앞서 파드의 레이블이 서비스의 파드 셀렉터와 일치할 경우 파드가 서비스의 엔드포인트로 포함됨을 확인함
- 서비스의 파드 셀렉터와 일치하는 파드는 생성되자마자 서비스의 일부로 포함되어 바로 요청이 전달된다
  - 이 경우 만일 파드가 준비되지 않으면?
- 이 경우 파드가 준비되기 전까지 요청을 전달하지 않아야 함

### 레디니스 프로브 (Readiness probe)

- 앞서 살펴본 라이브니스 프로브와 비슷하게 특정 파드가 클라이언트 요청을 수신하는지 결정
  - 레디니스 프로브가 성공을 반환하면 파드가 준비된거
- 여기서 준비가 됐다는 말은 컨테이너마다 다를 수 있음
  - GET 요청에 응답하는지, 특정 URL을 호출할 수 있는지 등
  - 이를 작성하는건 애플리케이션 개발자의 몫

유형
- exec: 프로세스를 실행하며, 컨테이너 상태를 프로세스의 종료 상태 코드로 결정
- HTTP GET: HTTP GET 요청을 보내고 응답을 받아 컨테이너의 상태를 파악
- TCP 소켓: 지정된 포트로 TCP 연결을 열고, 소켓이 연결되면 준비된 것으로 간주

작동
- 레디니스 점검 수행에 앞서 미리 설정한 시간을 기다리도록 구성할 수 있음(`initialDelaySeconds`)
- 이후 죽지적으로 프로브를 호출하여 그 결과에 따라 작동
- 파드가 준비되지 않았으면 서비스에서 제거
- 준비되면 다시 서비스(엔드포인트)에 추가
- 라이브니스와는 다르게 실패하더라도 컨테이너가 종료 혹은 재시작되지는 않음
- 레디니스 프로브는 요청을 처리할 준비가 된 파드의 컨테이너만 요청을 수신하도록 함
- 파드의 레디니스 프로브가 실패하면 엔드포인트 오브젝트에서 파드가 제거됨

중요성
- 파드 그룹(e.g. 앱 서버 실행 파드)이 다른 파드(BE의 DB서버)에서 제공하는 서비스에 의존한다고 가정
- FE 파드 중 하나에 문제가 발생하여 DB에 연결할 수 없는 경우, 파드가 해당 요청을 처리할 수 없음을 알려야함
- 다른 파드가 멀쩡한 경우 그쪽으로 연결하기 때문에 정상적으로 요청을 처리하게 됨
- 레디니스 프로브를 사용하면 클라이언트가 정상인 파드하고만 통신하니까 시스템에 문제가 있다는 것을 알기 힘들 것임

### 파드에 레디니스 프로브 추가

- RC의 파드 템플릿을 수정하여 기존 파드에 레디니스 프로브를 추가해보자
  - `kubecl edit rc kubia`
- yaml에 `spec.template.spec.containers` 아래의 첫 번째 컨테이너에 다음을 추가

```yaml
apiVersion: v1
kind: ReplicationController
...
spec:
  ...
  template:
    ...
    spec:
      containers:
      - image: luksa/kubia
        imagePullPolicy: Always
        name: kubia
        ports:
        - containerPort: 8080
          protocol: TCP
        readinessProbe: # 다음을 추가
          exec:
            command:
            - ls
            - /var/ready
```

- 레디니스 프로브는 컨테이너 내부에서 `ls /var/ready`를 주기적으로 수행
  - 파일이 존재하면 종료 코드 0을 반환
  - 파일이 있으면(1) 레디니스 프로브 성공
  - 좀 이상한 레디니스 프로브처럼 보이지만, 이는 문제의 파일을 생성하거나 제거해 그 결과를 바로 전환할 수 있기 때문임
  - 파일이 없으면 준비가 안된거
- 아직 파일이 없으니까 준비가 안된 것 같지만 그렇진 않음
  - RC의 경우에도 파드 템플릿 변경이 기존 파드에는 영향을 미치진 않음
  - 즉, 기존 파드엔 레디니스 프로브가 정의되지 않은 상태
- 파드를 지우면 RC에 의해 새로운 파드가 생성

```sh
kubectl get pods
NAME          READY   STATUS    RESTARTS      AGE
kubia-6x52m   0/1     Running   0             34s
kubia-bg8s5   0/1     Running   0             34s
kubia-wmswc   0/1     Running   0             34s
```

- 새로 생성된 파드는 READY상태가 되질 않음
  - 앞서 설정한 `ls` 때문
  - `/var/ready` 파일을 만들어서 성공을 반환하도록 해보자

```sh
kubectl exec kubia-6x52m -- touch /var/ready
kubectl get pods
NAME          READY   STATUS    RESTARTS      AGE
kubia-6x52m   1/1     Running   0             34s
kubia-bg8s5   0/1     Running   0             34s
kubia-wmswc   0/1     Running   0             34s
```

- 아마 바로 준비가 되진 않을텐데, 이는 10초마다 주기적으로 프로브를 실행하기 때문
  - 따라서 10초안에는 레디니스 프로브가 이를 실행하고 파드가 준비상태로 바뀔 것임
- 준비상태가 된 후 해당 IP는 서비스의 유일한 엔드포인트가 되야 함

```sh
kubectl get endpoints kubia-loadbalancer
NAME                 ENDPOINTS          AGE
kubia-loadbalancer   10.244.0.39:8080   2d1h
```

- 아까 살펴본 loadbalancer의 ExternalIP로 curl을 날려서 제대로 되는지 보자
  - 아주 당연하게도 하나의 파드만 있으므로 매번 동일한 파드에 접속해야 함

```sh
curl http://10.104.74.139
You've hit kubia-6x52m
curl http://10.104.74.139
You've hit kubia-6x52m
curl http://10.104.74.139
```

- `/var/ready` 파일을 삭제하면 서비스에서 파드가 다시 제거됨

### 실제 상황에서 레디니스 프로브의 역할

- 위 예제는 레디니스 프로브의 기능을 보여주는 예시일뿐 실용적이진 않음
- 실제로는 애플리케이션이 요청을 수신할 수 있는지 여부에 따라 성공/실패를 해야할 것임
- 서비스에서 파드를 수동으로 제거하기 위해서는 프로브의 스위치를 수동으로 전환하는 대신 파드를 삭제하거나 파드 레이블을 변경
  - 팁: 파드와 서비스 레이블 셀렉터에 `enabled=true` 레이블을 추가/제거하는 식으로 할 수 있음

레디니스 프로브를 항상 정의하라
- 레디니스 프로브를 끝내기 전에 다음의 두 가지를 명심하자
  1. 파드에 레디니스 프로브를 추가하지 않으면 파드가 시작하는 즉시 서비스 엔드포인트가 됨
  2. 파드의 종료 코드는 포함하지 말자

## 헤드리스 서비스로 개별 파드 찾기

https://interp.blog/k8s-headless-service-why/

> 헤드리스(Headless) 서비스
>
>로드 밸런싱 (Load-balancing) 이나 단일 서비스 IP 가 필요하지 않은 경우엔, ‘헤드리스’ 서비스라는 것을 만들 수 있다. `.spec.clusterIP: None` 을 명시적으로 지정하면 된다.
>
> 이 헤드리스 서비스를 통해, 쿠버네티스의 구현에 의존하지 않고도 다른 서비스 디스커버리 메커니즘과 인터페이스할 수 있다.
>
> 헤드리스 서비스의 경우, 클러스터 IP가 할당되지 않고, kube-proxy가 이러한 서비스를 처리하지 않으며, 플랫폼에 의해 로드 밸런싱 또는 프록시를 하지 않는다. DNS가 자동으로 구성되는 방법은 서비스에 `selector`가 정의되어 있는지에 달려있다.

- 지금까지는 서비스가 안정적인 IP 주소를 제공하여 파드나 엔드포인트에 클라이언트가 연결
  - 서비스의 연결은 임의의 파드로 전달
- 만약 클라이언트가 모든 파드에 연결해야 된다면?
- 파드가 다른 파드에 연결해야 한다면?
- 서비스로 연결하는건 확실한 방법이 아님
- 클라이언트가 모든 파드에 연결하려면 각 파드의 IP를 알아야 하는데, 이를 위해선 k8s API 서버를 통해 파드와 IP 주소를 가져오는 것
  - 그러나 애플리케이션은 k8s와 무관하게(K8s-agnostic) 유지하는 것이 좋음
- 클라이언트가 DNS 조회로 파드 IP를 찾도록 할 수 있음
  - 서비스에 대한 DNS 조회가 이루어지면 하나의 IP(ClusterIP)를 반환
  - 서비스에 ClusterIP가 필요하지 않으면(`.spec.clusterIP: None`) DNS 서버가 하나의 서비스 IP 대신 파드 IP 목록을 반환
- DNS 서버는 하나의 *DNN A 레코드*를 반환하는 대신 서비스에 대한 여러 개의 A 레코드를 반환
  - 각 레코드는 해당 시점에서 서비스를 지원하는 개별 파드의 IP
- 그러므로 클라이언트는 간단한 DNS A 레코드 조회를 수행하고, 서비스에 포함된 모든 파드의 IP를 얻을 수 있음
- 클라이언트는 해당 정보를 사용해 파드에 연결

### 헤드리스 서비스 생성
- 앞서 살펴보았듯 `.spec.clusterIP: None`를 통해 생성
  - k8s는 클라이언트가 서비스의 파드에 연결할 수 있는 ClusterIP를 할당하지 않음 (=헤드리스)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kubia-headless
spec:
  clusterIP: None # Headless
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: kubia
```

```sh
kubectl describe svc kubia-headless
Name:              kubia-headless
Namespace:         default
Labels:            <none>
Annotations:       <none>
Selector:          app=kubia
Type:              ClusterIP
IP Family Policy:  SingleStack
IP Families:       IPv4
IP:                None
IPs:               None
Port:              <unset>  80/TCP
TargetPort:        8080/TCP
Endpoints:         10.244.0.39:8080
Session Affinity:  None
Events:            <none>
```

- ClusterIP가 없고, 엔드포인트에 파드 셀렉터와 일치하는 파드가 포함되어있음을 볼 수 있음
  - 앞서 레디니스 프로브로 인해 파드가 하나 밖에 준비되지 않음. 따라서 엔드포인트가 하나임
- 계속하기에 앞서 다른 파드도 준비되도록 하자

### DNS로 파드 찾기

- 파드가 준비되면 DNS 조회로 실제 파드 IP를 얻을 수 있는지 확인이 가능
  - 파드 내부에서 진행
- kubia 컨테이너 이미지에는 nslookup(or dig) 바이너리가 포함되지 않아 DNS 조회가 안됨
- 따라서 아래와 같이 새로운 컨테이너가 필요

```sh
kubectl run dnsutils --image=tutum/dnsutils --command -- sleep infinity
pod/dnsutils created
```

- 새 파드로 DNS 조회 수행

```yaml
kubectl exec dnsutils -- nslookup kubia-headless
Server:         10.96.0.10
Address:        10.96.0.10#53

Name:   kubia-headless.default.svc.cluster.local
Address: 10.244.0.40
Name:   kubia-headless.default.svc.cluster.local
Address: 10.244.0.39
Name:   kubia-headless.default.svc.cluster.local
Address: 10.244.0.41
```

- kubia-headless.default.svc.cluster.local에 대해 3개의 FQDN을 반환
  - 파드임
- 여기서 반환된 IP는 일반적인 서비스를 DNS와 반환하는 것과 다름
  - 일반적인 서비스(e.g. kubia)의 DNS가 반환하는 것은 ClusterIP임

```sh
kubectl exec dnsutils -- nslookup kubia
Server:         10.96.0.10
Address:        10.96.0.10#53

Name:   kubia.default.svc.cluster.local
Address: 10.110.194.90
```

- 일반 서비스와 마찬가지로 서비스의 DNS 이름에 연결하여 파드에 연결할 수 있음
  - 일반 서비스는 서비스 프록시에 연결
  - 헤드리스의 경우엔 파드의 IP를 통해 직접 파드에 연결
    - 파드 간의 로드밸런싱은 제공하지만, 서비스 프록시 대신 DNS 라운드 로빈을 통해 연결

### 모든 파드 검색 - 준비되지 않은 파드도 포함

- 준비된 파드만 엔드포인트가 됨
- 준비되지 않은 것도 찾고 싶다면?
- DNS 조회 메커니즘을 사용해 준비되지 않은 파드를 찾으면 됨
- k8s로 하여금 레디니스 상태에 상관없이 모든 파드를 서비스에 추가되게 하려면 `spec.publishNotReadyAddresses`에 true를 설정

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    app: akka-bootstrap-demo
  annotations:
    service.alpha.kubernetes.io/tolerate-unready-endpoints: "true"
  name: "integration-test-kubernetes-dns-internal"
spec:
  ports:
  - name: management
    port: 8558
    protocol: TCP
    targetPort: 8558
  - name: remoting
    port: 2552
    protocol: TCP
    targetPort: 2552
  selector:
    app: akka-bootstrap-demo
  clusterIP: None
  publishNotReadyAddresses: true  # 확인
```

## 서비스 문제 해결

서비스로 파드에 액세스할 수 없는 경우 다음과 같은 내용을 확인한 후 다시 시작
1. 외부가 아닌 클러스터 내에서 서비스의 ClusterIP에 연결되는지 확인
2. 서비스에 액세스할 수 있는지 확인하려고 핑을 날리지 말 것
3. 레디니스 프로브를 정의했다면 성공했는지 먼저 확인. 실패했다면 서비스에 파드가 추가되지 않음
4. 파드가 서비스의 일부인지 확인하려면 `kubectl get endpoints`를 이용
5. FQDN (e.g. kubia.default.svc.cluster.local) 혹은 이의 일부(kubia.default, kubia)로 액세스할 때 동작하지 않으면 FQDN 대신 ClusterIP로 접속되는지 확인
6. 대상 포트가 아닌 서비스로 노출된 포트에 연결하는 중인지 확인
7. 파드 IP에 직접 연결하여 파드가 올바른 포트에 연결되었는지 체크
8. 파드 IP로 애플리케이션에 접속할 수 없으면 애플리케이션이 localhost에만 바인딩하는지 확인