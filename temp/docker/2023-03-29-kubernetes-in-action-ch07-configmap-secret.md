---
title:  "쿠버네티스 인 액션 정리: Ch06 컨피그맵과 시크릿"
toc: true
toc_sticky: true
categories:
  - Docker
  - kubernetes
use_math: true
last_modified_at: 2023-04-07
---

## 개요

다루는 내용:
- 컨테이너 주 프로세스 변경
- 애플리케이션에 명령줄 옵션 전달
- 애플리케이션에 노출되는 환경변수 설정
- 컨피그맵으로 애플리케이션 설정
- 시크릿을 통한 민감한 정보 전달

## 1. 컨테이너화된 애플리케이션 설정

- 일반적으로는 명령줄 인수로 애플리케이션에 필요한 설정을 넘겨줌
- 혹은 환경변수를 이용
  - 설정 파일을 이미지에 포함하거나 볼륨을 마운트해야 되기 때문에 컨테이너 내부 설정 파일을 사용하는 것이 까다롭기 때문
    - 이미지 포함할 경우 이미지에 접근이 가능하다면 누구나 볼 수 있기 때문에 소스 코드에 설정을 넣고 하드코딩하는거나 마찬
- 가장 좋은 방법은 k8s에서 설정 데이터를 저장하는 리소스인 컨피그맵(ConfigMap)을 사용하는 것
- 컨피그맵을 사용하면 다음 방법을 통해 애플리케이션을 구성할 수 있음
  - 컨테이너에 명령줄 인수 전달
  - 각 컨테이너를 위한 사용자 정의 환경변수 지정
  - 특수한 유형의 볼륨을 통해 설정 파일을 컨테이너에 마운트
- 이 중 민감한 정보(자격증명, 암호화키 등)은 시크릿(Secert) 오브젝트를 통해 설정하면 됨

## 2. 컨테이너에 명령줄 인자 전달

### 2.1 도커에서 명령어와 인자 정의

- 컨테이너에서 실행하는 명령은 명령어와 인자로 나뉨
- `ENTRYPOINT`: 컨테이너가 시작될 때 호출된 명령어를 정의
- `CMD`: `ENTRYPOINT`에 전달되는 인자를 정의
- shell과 exec 형식을 지원
  - shell: `ENTRYPOINT node app.js`
  - exec: `ENTRYPOINT ["node", "app.js"]`

> exec는 주어진 명령어를 실행하는데 새로운 프로세스를 생성하지 않고, 쉘 프로세스를 대체.
> 예를 들어 bash 쉘에서 자바 프로그램을 실행하면 자바 프로그램의 ppid가 bash 쉘이 되고, 자바 프로그램이 bash 쉘의 하위 프로세스로 실행. exec 커맨드로 실행하면 bash쉘의 프로세스가 자바 프로그램이 됨 (ppid 없음). 이후 자바프로그램이 종료되면 프로세스가 종료되며 bash 쉘로 돌아오지 않음.

### 2.2 쿠버네티스에서 명령과 인자 재정의

- k8s에서 컨테이너를 정의할 때, ENTRYPOINT와 CMD를 재정의할 수 있음
  - 컨테이너 정의 안(`spec.containers.command`, `spec.containers.args`)에 command와 args 속성을 지정
    - 즉, `ENTRYPOINT`에 해당하는게 `command`이고, `CMD`에 해당하는게 `args`
  - 단, 파드 생성 이후 업데이트할 수 없음
  - 여러개의 args를 갖는 경우:
  - 
    ```yaml
    args:
    - foo
    - bar
    - "15"
    ```
  - 숫자는 따옴표로 묶어야하며, 문자열은 그냥 넣으면 됨

## 3. 컨테이너의 환경변수 설정

- 환경변수도 파드 생성 후에는 업데이트 불가능함을 주의

### 3.1 컨테이너 정의에 환경변수 지정

- 환경변수는 파드 레벨이 아닌 컨테이너 정의 안에 설정
  - `spec.containers.env`

### 3.2 변수값에서 환경변수 참조

- `$var` 구문을 이용해 정의된 환경변수를 참조할 수 있음

```yaml
    env:
    - name: FIRST_VAR
      value: "foo"
    - name: SECOND_VAR
      value: "$(FIRST_VAR)bar"
```

- 이 경우 `SECOND_VAR`의 값은 `"foobar"`가 됨
  - 앞서 2절에서 배운 `command`와 `args` 모두 이와 같은 방식을 사용할 수 있음

## 4. 컨피그맵으로 설정 분리

- 파드 정의에서 하드코딩된 값을 가져오는 것은 효율적이나 프로덕션과 개발을 위해서는 분리된 파드 정의가 필요할 것임
- 여러 환경에서 동일한 파드 정의를 사용하려면 설정을 분리하는 것이 권장
- 애플리케이션 구성의 요점은 환경에 따라 다르거나, 자주 변경되는 설정을 소스 코드와 별도로 유지하는 것

### 4.1 컨피그맵

- 컨피그맵(configmap)을 통해 설정 옵션을 별도 오브젝트로 분리할 수 있음
- 컨피그맵은 짧은 문자열에서 전체 설정 파일에 이르는 값을 갖는 key-value 쌍임
- 애플리케이션에선 컨피그맵을 직접 읽거나 존재 자체를 몰라도 됨
  - 맵의 내용을 컨테이너의 환경변수나 볼륨 파일로 전달
- 필요한 경우 애플리케이션에서 k8s REST API 엔드포인트를 통해 컨피그맵을 읽을 수는 있으나 반드시 필요한 경우가 아니라면 무관하도록 유지
- 컨피그맵을 통해 개발, 테스트, QA, 프로덕션 등 다양한 환경에 동일한 이름으로 여러 매니페스트를 유지할 수 있음

### 4.2 컨피그맵 생성

- `kubectl create configmap <CONFIGMAP>`을 통해 생성
  - e.g. `kubectl create configmap <CONFIGMAP> --from-literal=<KEY>=<VALUE>`
  - 컨피그맵 키는 유효한 DNS 서브도메인이여야 함(영숫자, 대시, 밑줄, 점으로 구성)
- **전체 설정 파일**같은 데이터를 통째로 저장할 수도 있음
  - e.g. `kubectl create configmap <CONFIGMAP> --from-file=<FILE>`
  - `config-file.conf` 파일 내용을 `my-config` 컨피그맵의 키 값으로 지정
- 키 이름을 직접 지정하는 것도 가능
  - e.g. `kubectl create configmap <CONFIGMAP> --from-file=<KEY>=<FILE>`
  - 파일 내용을 특정 키 값으로 저장
- 디렉터리 내 파일로도 생성이 가능
  - e.g. `kubectl create configmap <CONFIGMAP> --from-file=/path/to/dir`
  - 각 파일을 개별 항목으로 작성하며, 파일 이름이 **컨피그맵 키로 사용하기에 유효한 파일만** 추가

- 컨피그맵의 조회는 다음을 통해 수행할 수 있다
  - `kubectl get configmap`
  - `kubectl describe configmap <CONFIGMAP>`
  - `kubectl get configmap <CONFIGMAP> -o yaml`



### 4.3 컨피그맵 항목을 환경변수로 컨테이너에 전달

- 환경변수를 설정하여 파드 내 컨테이너에 전달
  - `spec.containers.env.valueFrom.configMapKeyRef`에 `name`과 `key`를 지정

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-env-from-configmap
spec:
  containers:
  - image: luksa/fortune:env
    env:
    - name: INTERVAL  # INTERVAL 환경변수 설정
      valueFrom:  # 고정 값을 설정하는 대신 컨피그맵 키에서 값을 가져와 초기화
        configMapKeyRef:
          name: fortune-config  # 참조하는 컨피그맵의 이름
          key: sleep-interval # 컨피그맵에서 이 키로 지정된 값을 변수로 설정
...
```

- 여러 개의 환경변수를 넣어야 되면 피곤해짐
- 존재하지 않는 컨피그맵을 참조하는 경우 컨테이너가 시작할 수 없음
  - 누락된 컨피그맵을 생성하면 파드를 다시 만들지 않아도 컨테이너가 시작됨
  - `configMapKeyRef.optional: true`로 지정하면 없어도 시작할 수 있음

### 4.4 컨피그맵의 모든 항목을 한 번에 환경변수로 전달

- `spec.containers.env`속성 대신 `spec.containers.envFrom`을 통해 컨피그맵의 모든 항목을 환경변수로 만들 수 있음

```yaml
spec:
  containers:
    - image: some-image
      envFrom:  # envFrom 사용
      - prefix: CONFIG_ # 환경변수 접두사 (optional). 없으면 키와 동일함
        configMapRef:
          name: my-config-map # my-config-map 이름의 컨피그맵 참조
```

- 만일 대시(-)와 같이 환경변수로 적합하지 않은 이름을 갖는 경우 키로 반환하지 않음

### 4.5 컨피그맵 항목을 명령줄 인자로 전달

- 컨피그맵 값을 컨테이너 내 프로세스의 인자로 전달하는 방법
- 컨피그맵 항목을 환경변수로 초기화한 후 이를 인자로 참조하면 됨

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-args-from-configmap
spec:
  containers:
  - image: luksa/fortune:args # 이 이미지에선 첫 번째 인자에서 interval이란 변수를 가져옴
    env:  # 환경변수 정의 시작
    - name: INTERVAL
      valueFrom: 
        configMapKeyRef:
          name: fortune-config
          key: sleep-interval
    args: ["$(INTERVAL)"] # 인자에 앞서 정의한 환경변수 지정
...
```

### 4.6 컨피그맵 볼륨을 사용하여 컨피그맵 항목을 파일로 노출하기

- 파일을 컨테이너에 노출하면 챕터 6에서 보았던 컨피그맵 볼륨이 필요
- 컨피그맵 볼륨은 파일로 컨피그맵의 각 항목을 노출하며, 프로세스는 이를 읽어 값을 얻음

컨피그맵 생성
- `kubectl delete configmap fortune-config` 명령으로 이전에 만들었던 fortune-config를 삭제
  - 이후 아래의 `configmap-files/my-nginx-config.conf` 파일 생성
  - ```
    server {
        listen              80;
        server_name         www.kubia-example.com;

        # 일반 텍스트와 XML 파일에 대해 gzip 압축 활성화
        gzip on;
        gzip_types text/plain application/xml;

        location / {
            root   /usr/share/nginx/html;
            index  index.html index.htm;
        }

    }
    ```
- 컨피그맵에 sleep-interval 항목도 포함시키려면, 동일한 디렉터리에 sleep-interval이라는 텍스트 파일을 생성하고, 25를 저장
- `kubectl create configmap fortune-config --from-file=configmap-files`로 컨피그맵 생성
- 이를 파드에서 사용하려면 컨피그맵 항목에서 생성된 파일로 볼륨을 초기화
- 예제구성:
  - 현재 예제는 nginx를 사용
  - nginx는 `/etc/nginx/nginx.conf`파일의 설정을 읽음
  - 그러나 기본 설정 옵션을 가진 파일을 이미 포함하여, 이를 무시하고 싶지 않음
  - nginx의 기본 설정 파일은 `/etc/nginx/conf.d` 디렉터리의 모든 .conf 파일
  - 따라서 이에 원하는 설정을 추가하는 식으로 구성

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-configmap-volume
spec:
  containers:
  ...
  - image: nginx:alpine
    name: web-server
    volumeMounts:
    ...
    - name: config
      mountPath: /etc/nginx/conf.d  # 컨피그맵 볼륨을 마운트하는 위치
      readOnly: true
    - name: config
      mountPath: /tmp/whole-fortune-config-volume
      readOnly: true
    ...
  volumes:
  - name: html
    emptyDir: {}
  - name: config
    configMap:  # 볼륨이 fortune-config 컨피그맵을 참조
      name: fortune-config
```

- 현재 구성대로라면 `my-nginx-config.conf`와 `sleep-interval` 컨피그맵이 파일로 추가
- sleep-interval은 fortuneloop 컨테이너에서 사용되지만, 같이 포함되어 있음
- 여러 컨피그맵을 동일한 파드의 컨테이너를 구성하는 데 사용하는건 조금 이상함
- 동일한 파드에 있는 컨테이너들은 컨테이너가 서로 밀접한 관게를 갖으므로 하나의 유닛으로 설정되어야 함
- 이를 위해서는 볼륨을 컨피그맵의 일부만으로 채울수 있음 (이 경우 `my-nginx-config.conf`)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-configmap-volume-with-items
spec:
  ...
  volumes:
  - name: html
    emptyDir: {}
  - name: config
    configMap:
      name: fortune-config  # 컨피그맵 이름
      items:  # 볼륨에 포함할 항목을 조회하여 선택함
      - key: my-nginx-config.conf # 해당 키 아래 항목들을 가져옴
        path: gzip.conf # 가져온 값들을 여기에 저장
```

- 이를 통해 `/etc/nginx/conf.d` 디렉터리에는 `gzip.conf`파일만 남게되어 깔끔하게 유지할 수 있다

디렉터리를 마운트할 때 주의할 점
- 컨테이너 이미지 자체에 있던 `/etc/nginx/conf.d`에 볼륨을 마운트하면 기존의 파일은 숨겨진다
  - 원래 있던 파일은 해당 파일시스템이 마운트되어 있는 동안 접근이 불가능
  - 만일 중요한 파일을 포함하는 디렉토리에 볼륨을 마운트하면 문제가 발생할 것임
- 기존 디렉터리 내 파일을 숨기지 않고 볼륨을 마운트하려면, 전체 볼륨을 마운트하는 대신 volumneMount에 **subPath** 속성으로 파일이나 디렉터리 하나를 볼륨에 마운트할 수 있다
- e.g. 컨피그맵 볼륨의 여러 파일 중 `myconfig.conf` 파일을 `/etc/someconfig.conf` 파일로 추가

```yaml
spec:
  containers:
  - image: some/image
    volumeMounts:
    - name: myvolume
      mountPath: /etc/someconfig.conf # 디렉터리가 아닌 파일을 마운트함
      subPath: myconfig.conf  # 전체 볼륨을 마운트하는 대신 myconfig.conf 항목만 마운트함
```

- subpath는 모든 종류의 볼륨을 마운트할 때 사용할 수 있음

컨피그맵 볼륨 안에 있는 파일의 권한(permission) 설정
- 기본적으로 컨피그맵 볼륨의 모든 파일 권한은 *644(-rw-r-r--)*로 설정
  - 644란 owner는 read(4)/write(2)가 가능, group과 other은 read(4)만 가능
- 이를 변경하려면 볼륨 정의 안의 defaultMode 속성을 설정해 변경할 수 있음

> 유닉스/리눅스 시스템에서의 파일모드는 네자리 숫자로 되어 있다.
> 755(-rwxr-xr-x), 644(-rw-r--r--)와 같은 세 자리는 앞에 0이 생략된 것. 
> 접근 권한은 8진수 또는 r(read, 4), w(write, 2), x(execute, 1)로 표현이 가능.
> 8진수로 표현할 때는 권한의 합으로 표시하며, 소유자, 그룹 소유자, 기타 사용자를 위한 파일 모드를 설정함
> 맨 앞은 특수권한으로 SetUID(4), SetGID(2), Sticky bit(1)로 표현된다
> 
> ![image](https://user-images.githubusercontent.com/47516855/230543982-efc06be3-847e-417f-9d53-4eb4d69eb7ee.png){: .align-center}{: width="600"}
>
> 출처: [[Unix/Linux] 특수 권한(setuid, setgid, sticky bit)[오늘도 난, 하하하]. (2023.04.07). URL: https://eunguru.tistory.com/115](https://eunguru.tistory.com/115)


```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-configmap-volume
spec:
  ...
  volumes:
  - name: html
    emptyDir: {}
  - name: config
    configMap:
      name: fortune-config
      defaultMode: 0660 # 모든 파일 권한을 ---rw-rw---로 설정
```

- 컨피그맵은 중요하지 않은 설정 데이터에만 사용해야 하지만, defaultMode를 통해 소유자와 그룹만 파일을 읽도록 만들 수 있음
- TODO: 책에는 "6600"으로 표현되어있으며, 이 경우 -rw-rw-----가 됨

### 4.7 애플리케이션의 재시작 없이 설정을 업데이트하는 방법
- 컨피그맵을 사용하면 파드를 다시 만들거나 컨테이너를 재기동하지 않고도 설정의 변경이 가능하다
  - 단 오래걸릴 수 있다 (최대 1분)

컨피그맵 편집
- 컨피그맵을 변경하고, 파드 안의 프로세스가 이를 다시 로드하는 방법을 살펴보자
- 이전에 했던 Nginx 설정 파일을 편집하여 gzip 압축을 해제하자
- `kubectl edit configmap fortune-config` 명령으로 컨피그맵을 편집할 수 있다
  - 여기서 `gzip on`을 `gzip off`로 변경
- 이후 `kubectl exec fortune-configmap-volume -c web-server -- cat /etc/nginx/conf.d/my-nginx-config.conf`를 통해 파일 내용을 출력해보자
- 설정이 변경된 것을 확인할 수는 있으나 nginx에는 변화가 없다
  - 이는 파일의 변경을 감시하지 않으며, 다시 로드하는 기능이 없기 때문

Nginx에 신호를 전달하여 설정을 다시 로드
- `kubectl exec fortune-configmap-volume -c web-server -- nginx -s reload`
  - 리로드 전까진 응답을 압축하여 보내다가 리로드를 수행하면 응답이 압축되지 않는 것을 확인할 수 있음

파일이 한꺼번에 업데이트되는 방법 이해하기
- 컨피그맵 볼륨에 있는 모든 파일은 한 번에 동시에 업데이트 됨
- 이는 심볼릭 링크를 이용하여 수행됨
  - `kubectl exec -it fortune-configmap-volume -c web-server -- ls -lA /etc/nginx/conf.d`로 확인
  - 마운트된 컨피그맵 볼륨 안의 파일은 `..data`의 파일을 가리키는 심볼릭 링크
  - `..data` 또한 심볼릭링크
- 컨피그맵이 업데이트되면 새 디렉터리를 생성하여 모든 파일을 쓴 다음 `..data` 심볼릭 링크가 새 디렉터리를 가리키도록 하여 한번에 가져옴

이미 존재하는 디렉터리에 파일만 마운트하는 경우 업데이트가 되지 않음
- 전체 볼륨 대신 단일 파일을 컨테이너에 마운트하는 경우 파일이 업데이트되지 않음 (2023-04-07 기준 유효)
- 개별 파일을 추가한 후 원본 컨피그맵을 통해 이를 업데이트하려면 전체 볼륨을 다른 디렉터리에 마운트한 후 해당 파일을 가리키는 심볼릭 링크를 생성하여 해결
  - 컨테이너 이미지에서 심볼릭 링크를 만들거나 컨테이너를 시작할 때 만들 수 있음

컨피그맵 업데이트의 결과 이해하기
- 컨테이너의 가장 중요한 기능은 불변성
  - 즉, 같은 이미지에서 생성된 컨테이너는 차이가 없음
  - 실행 중인 컨테이너가 컨피그맵을 이용하여 변경하는 경우 이러한 불변성을 우회하게 되는 것이니 잘못된 것일까?
- 애플리케이션이 설정을 다시 읽는 기능을 지원하지 않는 경우에는 문제가 발생
  - 컨피그맵이 변경된 이후 생성된 파드와 변경되기 전 실행된 파드의 설정이 서로 다름
  - 따라서 애플리케이션이 설정을 자동으로 읽지않으면 파드가 사용 중인, 이미 존재하는 컨피그맵을 수정하는 것은 권장되지 않음
- Reload를 지원한다면 별 문제가 되지는 않음
  - 그러나 컨피그맵 볼륨의 파일이 실행 중인 모든 인스턴스에 거쳐 동기적으로 업데이트되지 않음
  - 따라서 개별 파드의 파일이 최대 1분 동안은 동기화되지 않을수도 있다는 점을 명심해야 함

## 5. 시크릿으로 민감한 데이터를 컨테이너에 전달
- 민감한 정보는 컨피그맵이 아닌 시크릿을 이용하여 컨테이너에 전달함

### 5.1 시크릿

- 시크릿은 컨피그맵과 유사하게 key-value 쌍으로 이루어짐
- 시크릿은 다음과 같은 상황에서 사용
  - 환경변수로 시크릿 항목을 컨테이너에 전달할 때
  - 시크릿 항목을 볼륨 파일로 노출할 때
- k8s는 시크릿에 접근해야 하는 파드를 실행하는 노드에만 개별 시크릿을 배포하여 보안을 유지
- 노드 자체적으로 시크릿을 항상 메모리에만 저장하며, 물리적 장치에는 저장하지 않음
- 마스터 노드(etcd)에는 시크릿을 암호화하지 않는 형식으로 저장하므로 마스터 노드를 보호하는 것이 필요함
  - 이에는 etcd 저장소를 안전하게 하는 것뿐만 아니라 권한이 없는 유저가 API 서버를 이용하지 못하게 하는 것도 포함
  - 왜냐하면 파드를 만들 수 있는 누구나 시크릿을 파드에 마운트하고 민감한 데이터에 접근하는 것이 가능하기 때문
- 시크릿과 컨피그맵 중 어느 것을 사용할지 선택하는 기준은 다음과 같다
  - 민감하지 않은 일반 설정 데이터 → 컨피그맵
  - 민감한 데이터 → 시크릿


## 5.2 기본 토큰 시크릿 소개
- `kubectl describe` 명령을 사용할 때 항상 시크릿을 보게 됨
  - 모든 파드에는 secret 볼륨이 자동으로 연결돼어 있음
  - 이는 default-token 시크릿임
- 시크릿도 k8s의 리소스이므로 조회가 가능
  - `kubectl get secrets`를 통해 디폴트 시크릿을 한번 확인해보자
- `kubectl describe secrets`를 통해 시크릿을 자세히 살펴보자
  - 시크릿이 갖는 세가지 항목 `ca.crt`, `namespace`, `token`을 확인할 수 있음
  - 이는 파드 안에서 k8s API 서버와 통신할 때 필요한 모든 것을 나타냄
- 이상적으로는 애플리케이션이 k8s를 인지할 필요가 없지만, k8s와 직접 대화하는 것 외에 대안이 없으면 secret 볼륨을 통해 제공된 파일을 사용
- `kubectl describe pod`를 통해 실제로 secret 볼륨이 마운트 된 것을 확인할 수 있음
  - 시크릿은 컨피그맵과 비슷하므로, secret 볼륨이 마운트된 디렉터리에서 이 파일을 확인할 수 있을 것임
  - `kubectl exec <POD> -- ls /var/run/secrets/kubernetes.io/serviceaccount/`를 통해 확인해보자

## 5.3 시크릿의 생성
- 시크릿을 사용하여 fortune-serving Nginx 컨테이너가 HTTPS 트래픽을 제공하도록 해보자
  - 이를 위해서는 인증서와 개인 키가 필요 → 민감한 데이터이므로 시크릿에 포함
- 인증서와 개인키는 아래 방법으로 만들 수 있음
  - `openssl genrsa -out https.key 2048`
  - `oepnssl req -new -x509 -key https.key -out https.cert -days 3650 -subj/CN=www.kubia-example.com`
  - 혹은 코드 아카이브에 있는 파일을 사용해도 무방 (`fortune-https`)
- 추가적인 설명을 위해 다음과 같은 파일을 생성
  - `echo bar > foo`
- 이후 다음의 명령어를 통해 시크릿을 만들 수 있음
  - `kubectl create generic fortune-https --from-file=https.key --from-file=https.cert --from-file=foo`
  - `kubectl create generic fortune-https --from-file=fortune-https`

### 5.4 컨피그맵과 시크릿 비교
- `kubectl get secret fortune-https -o yaml`을 통해 이전의 컨피그맵과 시크릿을 비교해보자
- 시크릿은 Base64로 인코딩되어 있음
  - 이는 일반 텍스트 외 바이너리값도 담을 수 있기 때문
- 모든 민감한 데이터가 바이너리 형태는 아니기 때문에 문자를 `stringData` 필드로 설정할 수도 있음
  - 반대로 Base64로 인코딩했다면 `data` 필드에 넣어도 무방
  - `stringData` 필드의 모든 키-값 쌍은 data 필드로 합쳐짐
  - 만약 키가 `data` 와 `stringData` 필드 모두에 정의되어 있으면, `stringData` 필드에 지정된 값이 **우선적으로 사용**
  - stringData 필드는 쓰기 전용으로, 값을 설정할 때만 사용할 수 있다
  - `kubectl get -o yaml` 명령으로 시크릿의 yaml 정의를 가져올 때 stringData는 표시되지 않음
  - 그러나 이로 지정한 모든 항목은 data 항목 아래에 다른 모든 항목처럼 Base64로 인코딩되어 표시
- secret 볼륨을 통해 컨테이너에 노출하면 값의 타입에 관계없이 실제 형식으로 디코딩되어 파일에 기록됨
- 환경변수로 시크릿 항목을 노출할 때도 마찬가지
- 두 경우 모두 애플리케이션에서 디코딩할 필요없이 파일 내용을 읽거나 환경변숫값을 찾아 직접 사용할 수 있음

### 5.5 파드에서 시크릿 사용
- fortun-https 시크릿을 Nginx에서 사용할 수 있도록 설정해보자

HTTPS를 활성화하도록 fortune-config 컨피그맵 수정
- 컨피그맵을 수정하자 (`kubectl edit configmap fortune-config`)
- 설정에서 서버가 인증서와 키 파일을 `/etc/nginx/certs`에서 읽기 때문에 secret 볼륨을 해당 위치로 마운트해야함

fortune-https 시크릿을 파드에 마운트
- 새로운 fortune-https 파드를 만들고 다음 yaml 파일을 참고해 인증서와 키가 담긴 secret 볼륨을 web-server 컨테이너 안에 마운트

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: fortune-https
spec:
  containers:
  - image: luksa/fortune:env
    name: html-generator
    env:
    - name: INTERVAL
      valueFrom: 
        configMapKeyRef:
          name: fortune-config
          key: sleep-interval
    volumeMounts:
    - name: html
      mountPath: /var/htdocs
  - image: nginx:alpine
    name: web-server
    volumeMounts:
    - name: html
      mountPath: /usr/share/nginx/html
      readOnly: true
    - name: config
      mountPath: /etc/nginx/conf.d
      readOnly: true
    - name: certs
      mountPath: /etc/nginx/certs/  # Nginx 서버의 인증서와 키 위치에 시크릿 볼륨을 마운트
      readOnly: true
    ports:
    - containerPort: 80
    - containerPort: 443
  volumes:
  - name: html
    emptyDir: {}
  - name: config
    configMap:
      name: fortune-config
      items:
      - key: my-nginx-config.conf
        path: https.conf
  - name: certs # fortune-https 시크릿을 참조하도록 secret 볼륨을 정의
    secret:
      secretName: fortune-https
```

- 컨피그맵과 마찬가지로 secret 볼륨 또한 defaultMode 속성을 통해 볼륨에 노출된 파일 권한을 지정할 수 있는 기능을 지원함

Nginx가 시크릿의 인증서와 키를 사용하는지 테스트
- 파드가 실행되면 포트 포워드 터널링으로 파드의 44ㄷ번 포트를 열고 curl 명령으로 요청을 보내어 https 트래픽을 제공하는지 확인하자
```sh
kubectl port-forward fortune-https 8443:443 & 
curl https://localhost:8443 -k -v
```

시크릿을 볼륨을 메모리에 저장하는 이유
- secret 볼륨은 시크릿 파일을 저장하는 데 인메모리 파일시스템(*tmpfs*)을 사용
  - 이는 민감한 데이터를 디스크에 저장하여 노출시킬 위험을 피하기 위함
- `kubectl exec fortune-https -c web-server -- mount | grep cert`를 통해 조회할 수 있음

환경변수로 시크릿 노출
- 컨피그맵과 마찬가지로 시크릿의 개별 항목을 환경변수로 노출할 수 있음
- 예제:
  - 시크릿에서 foo 키를 환경변수 FOO_SECRET으로 노출하려면 다음을 컨테이너 정의(`spec.containers.env`)에 추가

```yaml
    env:
    - name: FOO_SECRET
      valueFrom:
        secretKeyRef: # 컨피그맵의 configMapKeyRef와 동일
          name: fortune-https # 키를 갖고 있는 시크릿의 이름
          key: foo  # 노출할 시크릿의 키 이름
```

- 애플리케이션이 로그에 환경변수를 남길 수도 있기 때문에 조심히 사용해야 함
- 자식 프로세스는 상위 프로세스의 모든 환경변수를 상속받기 때문에 써드파티 바이너리를 실행할 경우 어떻게 활용되는지 알기가 힘듬
- 따라서 환경변수로 시크릿을 컨테이너에 넘기는 것은 의도치 않은 노출을 야기하기 때문에 안전을 위해서라면 항상 secret 볼륨을 사용해야 함

### 5.6 이미지를 가져올 때 사용하는 시크릿 이해
- k8s에서 자격증명을 전달하는 경우 (e.g. 프라이빗 컨테이너 이미지 레지스트리 사용)

도커 허브에서 프라이빗 이미지 사용하는 방법
1. 도커 레지스트리 자격증명을 가진 시크릿 생성
  - ```sh
    kubectl create secret docker-registry mydockerhubsecret \
    --docker-username=myusername --docker-password=mypassword \ 
    --docker-email=my.email@provider.com
    ```
  - generic 시크릿과는 다르게 docker-registry 형식을 가진 시크릿을 생성
  - 이 안에 도커 허브에서 사용할 이름, 패스워드, 이메일을 지정
2. 파드 매니페스트 안에 imagePullSecrets 필드에 해당 시크릿 참조
  - ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: private-pod
    spec:
      imagePullSecrets: # 프라이빗 이미지 레지스트리에서 이미지를 가져올 수 있도록 설정
      - name: mydockerhubsecret
      containers:
      - image: username/private:tag
        name: main
    ```

모든

