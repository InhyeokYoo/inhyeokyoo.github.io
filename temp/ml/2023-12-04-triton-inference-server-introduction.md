---
title:  "NVDIA Triton inference server 개념 소개"
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

## Architectur

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#scheduling-and-batching

Triton supports batch inferencing by allowing individual inference requests to specify a batch of inputs.
The inferencing for a batch of inputs is performed at the same time which is especially important for GPUs since it can greatly increase inferencing throughput.
In many use cases the individual inference requests are not batched, therefore, they do not benefit from the throughput benefits of batching.

The inference server contains multiple scheduling and batching algorithms that support many different model types and use-cases. More information about model types and schedulers can be found in [Models And Schedulers](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#models-and-schedulers).

### Stateless Models

With respect to Triton's schedulers, a stateless model does not maintain state between inference requests. Each inference performed on a stateless model is independent of all other inferences using that model.

Examples of stateless models are CNNs such as image classification and object detection. The default scheduler or dynamic batcher can be used as the scheduler for these stateless models.

RNNs and similar models which do have internal memory can be stateless as long as the state they maintain does not span inference requests.
For example, an RNN that iterates over all elements in a batch is considered stateless by Triton if the internal state is not carried between batches of inference requests.
The default scheduler can be used for these stateless models.
The dynamic batcher cannot be used since the model is typically not expecting the batch to represent multiple inference requests.


## 사용법

Triton을 사용하려면 model repository와 model configuration가 필요하다.

## Model repository

Model repository는 Triton이 모델과 메타데이터를 읽는 방법으로, 로컬이나 네트워크 파일 시스템 혹은 AWS S3와 같은 cloud object store를 이용할 수 있다.
자세한 내용은 [공식 문서](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-repository-locations)를 확인해보자.

Model repository는 다음과 같은 구조로 되어 있다.

```
# Example repository structure
<model-repository>/
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  ...
```

- `model-name`: 모델의 이름.
- `config.pbtxt`: 각 모델에 대한 configuration (model configuration)으로, backend, name, shape, 모델 인풋과 아웃풋에 대한 datatype을 정의해야한다.
- `version`: 같은 모델에 대한 여러개의 versioning을 지원. TODO: (https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#version-policy)

## Model configuration

위를 통해 model repository를 정의했으면 그 다음은 `config.pbtxt` model configuration을 살펴보자.
형태는 아래와 같다.

```
name: "text_detection"
backend: "onnxruntime"
max_batch_size : 256
input [
  {
    name: "input_images:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 3 ]
  }
]
output [
  {
    name: "feature_fusion/Conv_7/Sigmoid:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }
]
output [
  {
    name: "feature_fusion/concat_3:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 5 ]
  }
]
```

- `name` (*Optional*): model directory의 이름과 정확히 매칭되야 한다.
- `backend`: model이 실행되는데 필요한 backend를 나타낸다. TensorFlow, PyTorch, ONNX, Python 등 다양한 종류가 있다. 가능한 backend의 종류를 살펴보려면 [공식문서](https://github.com/triton-inference-server/backend#backends)를 참고하자.
- `max_batch_size`: 말 그대로 최대 batch size를 의미한다. 
- `input`/`output`: 인풋/아웃풋의 이름과 데이터타입 등을 정의한다.
- `model_transaction_policy`: model의 transactions TODO: 잘 모르겠음..

대부분의 경우에는 [Auto-Generated Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#auto-generated-model-configuration)으로 인해 자동으로 인풋과 아웃풋을 채워주므로 크게 신경쓸 필요는 없다.




실행방법은 다음과 같다.

```console
$ tritonserver --model-repository=<model-repository-path>
```

TODO: 실행 옵션


### Decoupled Backends and Models

https://github.com/triton-inference-server/python_backend/blob/main/README.md#decoupled-mode
https://github.com/triton-inference-server/python_backend/tree/main/examples/decoupled
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/decoupled_models.html#deploying-decoupled-models


### Scheduling And Batching

Triton supports batch inferencing by allowing individual inference requests to specify a batch of inputs.
The inferencing for a batch of inputs is performed at the same time which is especially important for GPUs since it can greatly increase inferencing throughput.
In many use cases the individual inference requests are not batched, therefore, they do not benefit from the throughput benefits of batching.

### Default Scheduler
The default scheduler is used for a model if none of the scheduling_choice properties are specified in the model configuration. The default scheduler simply distributes inference requests to all model instances configured for the model.

### Dynamic Batcher

Dynamic batching is a feature of Triton that allows inference requests to be combined by the server, so that a batch is created dynamically. Creating a batch of requests typically results in increased throughput. The dynamic batcher should be used for stateless models. The dynamically created batches are distributed to all model instances configured for the model.

Dynamic batching is enabled and configured independently for each model using the ModelDynamicBatching property in the model configuration. These settings control the preferred size(s) of the dynamically created batches, the maximum time that requests can be delayed in the scheduler to allow other requests to join the dynamic batch, and queue properties such a queue size, priorities, and time-outs. Refer to this guide for a more detailed example of dynamic batching.

Dynamic batching can be enabled and configured on per model basis by specifying selections in the model's config.pbtxt. Dynamic Batching can be enabled with its default settings by adding the following to the config.pbtxt file:

```
dynamic_batching { }
```

While Triton batches these incoming requests without any delay, users can choose to allocate a limited delay for the scheduler to collect more inference requests to be used by the dynamic batcher.

```
dynamic_batching {
    max_queue_delay_microseconds: 100
}
```

delay를 사용하는 경우, delay만큼 batch를 채우고, 그 이후 request를 진행함.

Recommended Configuration Process:
- Decide on a maximum batch size for the model
- Add the following to the model configuration to enable the dynamic batcher with all default settings. By default the dynamic batcher will create batches as large as possible up to the maximum batch size and will not delay when forming batches.
  ```
  dynamic_batching { }
  ```
- Use the Performance Analyzer to determine the latency and throughput provided by the default dynamic batcher configuration.
- If the default configuration results in latency values that are within your latency budget, try one or both of the following to trade off increased latency for increased throughput:
  - Increase maximum batch size.
  - Set batch delay to a non-zero value. Try increasing delay values until the latency budget is exceeded to see the impact on throughput.
- Preferred batch sizes should not be used for most models. A preferred batch size(s) should only be configured if that batch size results in significantly higher performance than other batch sizes.

#### Preferred Batch Sizes

the batch sizes that the dynamic batcher should attempt to create.
For most models, *preferred_batch_size* should not be specified, as described in Recommended Configuration Process.
**An exception is TensorRT** models that specify multiple optimization profiles for different batch sizes.
In this case, because some optimization profiles may give significant performance improvement compared to others, it may make sense to use *preferred_batch_size* for the batch sizes supported by those higher-performance optimization profiles.

The following example shows the configuration that enables dynamic batching with preferred batch sizes of 4 and 8.

```
  dynamic_batching {
    preferred_batch_size: [ 4, 8 ]
  }
```

When a model instance becomes available for inferencing, the dynamic batcher will attempt to create batches from the requests that are available in the scheduler.
Requests are added to the batch in the order the requests were received.
If the dynamic batcher can form a batch of a preferred size(s) it will create a batch of the largest possible preferred size and send it for inferencing. If the dynamic batcher cannot form a batch of a preferred size (or if the dynamic batcher is not configured with any preferred batch sizes), it will send a batch of the largest size possible that is less than the maximum batch size allowed by the model (but see the following section for the delay option that changes this behavior).

The size of generated batches can be examined in aggregate using [count metrics](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md#inference-request-metrics).

#### Delayed Batching

The dynamic batcher can be configured to allow requests to be delayed for a limited time in the scheduler to allow other requests to join the dynamic batch.
For example, the following configuration sets the maximum delay time of 100 microseconds for a request.

```
  dynamic_batching {
    max_queue_delay_microseconds: 100
  }
```

The max_queue_delay_microseconds property setting changes the dynamic batcher behavior when a maximum size (or preferred size) batch cannot be created.
When a batch of a maximum or preferred size cannot be created from the available requests, the dynamic batcher will delay sending the batch as long as no request is delayed longer than the configured max_queue_delay_microseconds value.
If a new request arrives during this delay and allows the dynamic batcher to form a batch of a maximum or preferred batch size, then that batch is sent immediately for inferencing.
If the delay expires the dynamic batcher sends the batch as is, even though it is not a maximum or preferred size.

#### Preserve Ordering

The preserve_ordering property is used to force all responses to be returned in the same order as requests were received. See the protobuf documentation for details.

#### Priority Levels

By default the dynamic batcher maintains a single queue that holds all inference requests for a model.
The requests are processed and batched in order.
The priority_levels property can be used to create multiple priority levels within the dynamic batcher so that requests with higher priority are allowed to bypass requests with lower priority.
Requests at the same priority level are processed in order. Inference requests that do not set a priority are scheduled using the default_priority_level property.

#### Queue Policy

The dynamic batcher provides several settings that control how requests are queued for batching.

When **priority_levels** is not defined, the **ModelQueuePolicy** for the single queue can be set with **default_queue_policy**.
When priority_levels is defined, each priority level can have a different **ModelQueuePolicy** as specified by **default_queue_policy** and **priority_queue_policy**.

The **ModelQueuePolicy** property allows a maximum queue size to be set using the **max_queue_size**.
The **timeout_action**, **default_timeout_microseconds** and **allow_timeout_override** settings allow the queue to be configured so that individual requests are rejected or deferred if their time in the queue exceeds a specified timeout.


#### Custom Batching

You can set custom batching rules that work in addition to the specified behavior of the dynamic batcher.
To do so, you would implement five functions in tritonbackend.h and create a shared library.
These functions are described below.

### Sequence Batcher

Like the dynamic batcher, the sequence batcher combines non-batched inference requests, so that a batch is created dynamically. Unlike the dynamic batcher, the sequence batcher should be used for stateful models where a sequence of inference requests must be routed to the same model instance. The dynamically created batches are distributed to all model instances configured for the model.

Sequence batching is enabled and configured independently for each model using the ModelSequenceBatching property in the model configuration. These settings control the sequence timeout as well as configuring how Triton will send control signals to the model indicating sequence start, end, ready and correlation ID. See Stateful Models for more information and examples.


https://github.com/triton-inference-server/server/blob/main/docs/user_guide/ragged_batching.md#ragged-batching







### Instance group

https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization#concurrent-model-execution

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups

특정 GPU만 세팅하는 방법
TODO: 아래의 경우엔 gpu1, 2 각각에 하나씩 들어가는건지?
```
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 1, 2 ]
    }
  ]
```

### Rate Limiter Configuration

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#rate-limiter-configuration

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/rate_limiter.md#rate-limiter

> Rate Limiting이란 특정 시간 내에 할 수 있는 API 호출 수를 의미합니다. 사용자의 API 호출 수가 Rate Limit값을 초과하면 API 호출이 제한되며 요청이 > 실패하게 됩니다. 즉, 단어 그대로 API의 속도를 제한합니다. Rate Limiter를 적용하면 고가용성과 안정성을 보장할 수 있습니다. 무차별적으로 인입되는 > 요청을 받는 서버에서의 불안정성을 생각해보면 RateLimiter의 이 특징을 금방 납득할 수 있습니다.
> 출처: https://gngsn.tistory.com/224 [ENFJ.dev:티스토리]

Instance group optionally specifies rate limiter configuration which **controls how the rate limiter operates on the instances in the group**.

The rate limiter configuration is ignored if rate limiting is off (`--rate-limit=off`).
Triton schedules execution of a request (or set of requests when using dynamic batching) as soon as a model instance is available.
This behavior is typically best suited for performance.
However, there can be cases where running all the models simultaneously places excessive load on the server.
For instance, model execution on some frameworks dynamically allocate memory.
Running all such models simultaneously may lead to system going out-of-memory.

Rate limiter allows to postpone the inference execution on some model instances such that not all of them runs simultaneously. The model priorities are used to decide which model instance to schedule next.

If rate limiting is on and if an instance_group does not provide this configuration, then the execution on the model instances belonging to this group will not be limited in any way by the rate limiter. 
The configuration includes the following specifications:

#### Resources

The set of resources required to execute a model instance. 
The "name" field identifies the resource and "count" field refers to the number of copies of the resource that the model instance in the group requires to run. 
The "global" field specifies whether the resource is per-device or shared globally across the system.
Loaded models can not specify a resource with the same name both as global and non-global.
If no resources are provided then triton assumes the execution of model instance does not require any resources and will start executing as soon as model instance is available.

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/rate_limiter.md#resources

Resources are identified by a unique name and a count indicating the number of copies of the resource. By default, model instance uses no rate-limiter resources. By listing a resource/count the model instance indicates that it requires that many resources to be available on the model instance device before it can be allowed to execute. When under execution the specified many resources are allocated to the model instance only to be released when the execution is over. The available number of resource copies are, by default, the max across all model instances that list that resource. For example, assume three loaded model instances A, B and C each specifying the following resource requirements for a single device:

```
A: [R1: 4, R2: 4]
B: [R2: 5, R3: 10, R4: 5]
C: [R1: 1, R3: 7, R4: 2]
```



#### Priority

https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#priority

Priority serves as a weighting value to be used for prioritizing across all the instances of all the models. An instance with priority 2 will be given 1/2 the number of scheduling chances as an instance with priority 1.

The following example specifies the instances in the group requires four "R1" and two "R2" resources for execution. Resource "R2" is a global resource. Additionally, the rate-limiter priority of the instance_group is 2.

```
  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0, 1, 2 ]
      rate_limiter {
        resources [
          {
            name: "R1"
            count: 4
          },
          {
            name: "R2"
            global: True
            count: 2
          }
        ]
        priority: 2
      }
    }
  ]
```

The above configuration creates 3 model instances, one on each device (0, 1 and 2). The three instances will not contend for "R1" among themselves as "R1" is local for their own device, however, they will contend for "R2" because it is specified as a global resource which means "R2" is shared across the system. Though these instances don't contend for "R1" among themsleves, but they will contend for "R1" with other model instances which includes "R1" in their resource requirements and run on the same device as them.

In a resource constrained system, there will be a contention for the resources among model instances to execute their inference requests. Priority setting helps determining which model instance to select for next execution. See priority for more information.

## Triton Performance Analyzer

https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md

특정 input을 넣으려면 `--input-data` 옵션을 사용 (https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/input_data.md)

사용법:

```console
# NOTE: "my_model" represents a model currently being served by Triton
$ perf_analyzer -m my_model
...

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 482.8 infer/sec, latency 12613 usec
```

기타 cli 옵션은 [문서](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/cli.md#--concurrency-rangestartendstep)를 참고.

- This gives us a sanity test that we are able to successfully form input requests and receive output responses to communicate with the model backend via Triton APIs.
- The definition of “performing well” is subject to change for each use case. Some common metrics are throughput, latency, and GPU utilization. There are many variables that can be tweaked just within your model configuration (config.pbtxt) to obtain different results.
- As your model, config, or use case evolves, Perf Analyzer is a great tool to quickly verify model functionality and performance.

### Inference Load Modes

https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/inference_load_modes.md#concurrency-mode

Perf Analyzer has several modes for generating inference request load for a model.

#### Concurrency Mode

Perf Analyzer attempts to send inference requests to the server such that N requests are always outstanding during profiling.

> An outstanding request is one which has not been served yet.
> For instance, an application could make 30 concurrent requests to different web servers. 10 of them may come back with a response, while the other 20 have not been serviced. Therefore, those 20 are outstanding since they are waiting for a response.

For example, when using `--concurrency-range=4`, Perf Analyzer will to attempt to have 4 outgoing inference requests at all times during profiling.

#### Periodic Concurrency Mode

periodically launch a new set of inference requests until the total number of inference requests that has been launched since the beginning reaches N requests.

For example, when using `--periodic-concurrency-range 10:100:30`, Perf Analyzer will start with 10 concurrent requests and for every step, it will launch 30 new inference requests until the total number of requests launched since the beginning reaches 100. Additionally, the user can also specify when to launch the new requests by specifying `--request-period M`. This will set Perf Analyzer to launch a new set of requests whenever **all** of the latest set of launched concurrent requests received M number of responses back from the server.

10의 request로 시작하여 매 스텝마다 30개의 새로운 request를 생성한다. 이는 시작할 때부터의 request 총 합이 100이 될때가지 지속된다.

The periodic concurrency mode is currently supported only by gRPC protocol and with decoupled models. Additionally, the user must also specify a file where Perf Analyzer could dump all the profiled data using `--profile-export-file``.

#### Request Rate Mode

send N inference requests per second to the server during profiling. For example, when using `--request-rate-range=20`, Perf Analyzer will attempt to send 20 requests per second during profiling.

#### Custom Interval Mode

send inference requests according to intervals (between requests, looping if necessary) provided by the user in the form of a text file with one time interval (in microseconds) per line. For example, when using `--request-intervals=my_intervals.txt`, where `my_intervals.txt` contains:

```
100000
200000
500000
```

Perf Analyzer will attempt to send requests at the following times: 0.1s, 0.3s, 0.8s, 0.9s, 1.1s, 1.6s, and so on, during profiling.

### Performance Measurement Modes

https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/measurements_metrics.md#time-windows

#### Time Windows

When using time windows measurement mode (`--measurement-mode=time_windows`), Perf Analyzer will count how many requests have completed during a window of duration `X` (in milliseconds, via `--measurement-interval=X`, default is 5000). This is the default measurement mode.

특정 시간(window) `X` (`--measurement-interval=X`로 넣을 수 있으며, 디폴트 값은 5000 milliseconds)동안 얼마나 많은 request를 처리할 수 있는지 측정.

#### Count Windows

When using count windows measurement mode (`--measurement-mode=count_windows`), Perf Analyzer will start the window duration at 1 second and potentially dynamically increase it until `X` requests have completed (via `--measurement-request-count=X`, default is 50).

1초 동안 window duration을 시작하며, `X` request를 처리할 때까지 window를 늘림 (디폴트 값 50)

#### Metric

- Throughput (초): 측정 시간동안의 request 처리 개수.
- Latency: request-response까지의 시간. 일반적으로 HTTP가 gRPC보다 정확함. 구성 요소로는 queue, compute, overhead가 있음.
  - queue: request에 의한 inference schedule queue가 모델이 작동 가능할 때까지 기다리는 평균 시간
  - compute: 실제 inference를 수행하는데 걸리는 평균 시간 (GPU 오버헤드 등 포함).
  - overhead: gRPC 및 HTTP가 통신할 때 endpoint에서 제대로 잡히지 않는데 걸리는 평균 시간. 









## Triton Model Analyzer

https://github.com/triton-inference-server/model_analyzer

Triton Model Analyzer는 Triton Inference Server로 동작하는 모델의 최적화 툴이다.


### Quick Search

https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md#quick-search-mode

- Search type: Heuristic sweep using a hill-climbing algorithm to find an optimal configuration
- Default for:
    - Single ensemble models
    - Single BLS models
    - Multiple models being profiled concurrently
- Command: `--run-config-search-mode quick`

sparsely search the Max Batch Size, Dynamic Batching, and Instance Group spaces by utilizing a heuristic hill-climbing algorithm to help you quickly find a more optimal configuration.

Default search mode when profiling ensemble models, BLS models, or multiple models concurrently

limitations: If model config parameters are specified, they can contain only one possible combination of parameters

An example model analyzer YAML config that performs a Quick Search:

```yaml
model_repository: /path/to/model/repository/

run_config_search_mode: quick
run_config_search_max_instance_count: 4
run_config_search_max_concurrency: 16
run_config_search_min_model_batch_size: 8

profile_models:
  - model_A
```

Using the `--run-config-search-<min/max>...` config options you have the ability to clamp the algorithm's upper or lower bounds for the model's batch size and instance group count, as well as the client's request concurrency.

Note: By default, quick search runs unbounded and ignores any default values for these settings


### Automatic Brute Search

exhaustively search the Max Batch Size, Dynamic Batching, and Instance Group parameters of your model configuration

- Search type: Brute-force sweep of the cross product of all possible configurations
- Default for:
  - Single models, which are not ensemble or BLS
  - Multiple models being profiled sequentially
  - Command: `--run-config-search-mode brute`

It has two modes:
- Automatic:
  - No model config parameters are specified (default)
- Manual:
  - Any model config parameters are specified
  - `--run-config-search-disable option is specified`

The parameters that are automatically searched are model maximum batch size, model instance groups, and request concurrencies. Additionally, dynamic_batching will be enabled if it is legal to do so.


### Ensemble Model Search




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
k8s의 경우 이를 통해 startup probe를 사용할 수 있다.



![image](https://user-images.githubusercontent.com/47516855/105610744-83d35280-5df4-11eb-9c7e-7615fc4bbf46.png){: .align-center}{: width='500'}


## k8s

Kubernetes Deploy w/ Helm: https://github.com/okdimok/server/tree/62a981aa485b0310b5c25229fa9f9698c7d0763b/deploy/k8s-onprem

Metrics: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md#inference-request-metrics

참고자료:
https://velog.io/@hbjs97/triton-inference-server-%EB%AA%A8%EB%8D%B8%EA%B4%80%EB%A6%AC