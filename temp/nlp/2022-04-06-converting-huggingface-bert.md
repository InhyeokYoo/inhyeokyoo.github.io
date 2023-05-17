---
title:  "BERT 모델을 huggingface로 변환하기"
toc: true
toc_sticky: true
categories:
  - NLP
tags:
  - huggingface
use_math: true
last_modified_at: 2022-04-06
---

## 들어가며

흔하진 않지만 특별한 이유로 BERT와 같은 language model을 원작자가 제공하는 깃헙 소스나 기타 다른 소스를 통해 학습하는 경우가 있다. 그러나 대부분의 자연어처리 모델은 허깅페이스에 많이 의존하고 있으므로, 다른 소스를 활용하여 LM을 학습한다는 것은 큰 제약사항이 아닐수 없다. 대부분의 모델은 LM에 특정한 구조 (e.g. poly-encoder)를 얹어 학습하는 경우가 많기 때문이다. 따라서 이의 호환성을 일일이 수정하기란 여간 쉽지 않다.

나 또한 Nivida에서 제공하는 코드를 활용하여 학습한 BERT 모델을 사용하고 있었는데, huggingface BERT를 기반으로 하는 모델에 일일이 맞춰주기가 어려웠다. 이번 시간에는 일반 BERT 모델을 huggingface에서 사용하는 `BertModel`로 변환하는 것을 살펴보도록 하겠다.

## 준비물

huggingface에서 제공하는 [`configuration_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/configuration_bert.py)와 [`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) 등을 살펴보자. `Fairseq`와 같은 좀 더 범용적인 라이브러리에선 직접적으로 이를 변환하는 [코드](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py)가 있으므로 이를 살펴보아도 좋다. 다만 저 링크는 BART에 관련된 것이므로 직접적으로 참고하긴 어렵다.

또한 내가 학습한 모델의 원본도 살펴봐야 할 것이다. 본인의 코드가 있다면 이를 활용하면 된다. 나의 경우엔 nvidia에서 제공하는 [코드](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py)였으므로, 이 또한 살펴본다.

## 변환과정

BERT를 변환하는 과정은 크게 세 가지로 보면 된다.
1. vocab 처리
2. config 처리
3. model 내 submodule의 name처리 (ordered dict 형태로)
4. 변환 전 결과물과 변환 후 결과물의 비교

### 1. vocab 처리

내 경우엔 `transformers.BertTokenizerFast`를 사용하고 있었으므로 호환에 딱히 문제가 되지 않았다. 
다른 vocab을 사용하는 경우 `##`과 같은 prefix를 잘 수정해주도록 하자.


### 2. config 처리

config 또한 호환이 잘 가게끔 만들어놓으면 된다. 
Huggingface의 `transformers`내에서 필요로 하는 [`config.json`](https://huggingface.co/bert-base-cased/blob/main/config.json)를 살펴보고, 이와 동일하게 적어주면 될듯 하다. 
자세한 사항은 config 링크를 살펴보자.


### 3. model 내 submodule의 name처리

제일 중요하고 귀찮은 파트이다. 
직접 만든 BERT와 huggingface에서 제공하는 `BertModel`은 레이어의 이름이 다를뿐 같은 모델이라고 봐도 된다. 
따라서 내가 만든 BERT의 레이어 이름과 `BertModel`의 레이어 이름을 **서로 연결해주고 변환**시켜줘야 한다.

#### 1). transformers 뜯어보기

시작에 앞서 살펴볼 것이 몇 가지 있다.

우선 내 모델은 pre training 용으로 사용한 모델이다.
따라서 동일한 역할을 하는 `transformers.BertForMaskedLM`이 변환하기에 적합한 모델일 것이다.
그러나 실제로 원하는 것은 PLM이기 때문에 `transformers.BertModel`만 필요하다.

구체적으로 어느 모델을 사용할지 확인하기 위해 huggingface에서 만든 두 모델을 살펴보자.


```python
from transformers import BertPreTrainedModel, BertConfig, BertModel, BertTokenizerFast, BertForMaskedLM
from pathlib import Path
import torch
import torch.nn as nn

# BertModel/BertForMaskedLM 객체 생성

bert_config = BertConfig.from_json_file(nv_bert_path / Path("config.json"))
bert_lm = BertForMaskedLM(bert_config)
bert = BertModel(bert_config)

# state_dict 생성하여 비교하자
bert_state_dict = bert.state_dict()
bert_lm_state_dict = bert_lm.state_dict()

for item in bert_lm.named_children():
    print(item[0])

# 비교를 위한 method
def show_weights(bert):
    # check dimension
    state_dict = bert.state_dict()
    print(f"# of layers: {len(state_dict)}")
    print("Layers' name and their dimensions are:")
    for name, layer in state_dict.items():
        print(f"  {name} : {layer.shape}")

show_weights(bert_lm)
# show_weights(bert) # 둘 다 비교
```

결과

```pycon
# of layers: 205
Layers' name and their dimensions are:
  bert.embeddings.position_ids : torch.Size([1, 512])
  bert.embeddings.word_embeddings.weight : torch.Size([35000, 768])
  bert.embeddings.position_embeddings.weight : torch.Size([512, 768])
  bert.embeddings.token_type_embeddings.weight : torch.Size([2, 768])
  bert.embeddings.LayerNorm.weight : torch.Size([768])
  ...

# of layers: 200
Layers' name and their dimensions are:
  embeddings.position_ids : torch.Size([1, 512])
  embeddings.word_embeddings.weight : torch.Size([35000, 768])
  embeddings.position_embeddings.weight : torch.Size([512, 768])
  embeddings.token_type_embeddings.weight : torch.Size([2, 768])
  embeddings.LayerNorm.weight : torch.Size([768])
  embeddings.LayerNorm.bias : torch.Size([768])
  ...
```

몇 가지 차이가 눈에 띈다.

1. 첫 번째로는 레이어의 수가 차이가 난다. (`BertForMaskedLM`이 5개 레이어 더 많음)
2. 두 번째로는 레이어 앞에 `bert`라는 이름이 붙었다.
3. 세 번째로는 `pooler`가 없다.

1번 항목부터 우선 살펴보자.
레이어의 수가 200개가 넘어가다 보니 이를 일일이 살펴보기 어렵다.
`named_children()`를 이용하여 자세히 살펴보자.

```py
# BertForMaskedLM
print("BertForMaskedLM의 submodule:")
for model_name1, layer in bert_lm.named_children():
    print(model_name1)
    for model_name2, layer in layer.named_children():
        print("  "+model_name2)

# BertModel
print("BertModel의 submodule")
for model_name1, layer in bert_lm.named_children():
    print(model_name1)
    for model_name2, layer in layer.named_children():
        print("  "+model_name2)
```

```pycon
BertForMaskedLM의 submodule:
bert
  embeddings
  encoder
cls
  predictions

BertModel의 submodule:
embeddings
  word_embeddings
  position_embeddings
  token_type_embeddings
  LayerNorm
  dropout
encoder
  layer
pooler
  dense
  activation
```

보아하니 `BertForMaskedLM` 내에서 `BertModel` 객체를 받는 것으로 보인다.
실제로도 그러한지 허깅페이스의 `modeling_bert.py`를 살펴보자

```py
class BertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        ...
```

앞서 본 것과 마찬가지로 bert와 cls로 이루어진 것으로 보인다.
그리고 그 내부 bert는 `BertModel`이며, 그 안에는 위에서 본 것과 같이 인코더 등으로 이루어진다.

딱 보기에는 cls 서브모듈과 pooler 서브 모듈이 없는 것 같은데 실제로도 그러한지 살펴보자.
이를 위해서 다음과 같은 함수를 통해 substring을 통해 layer 이름에 접근, 비교한다.

```py
# BertForMaskedLM에는 layer 앞에 bert가 붙어서 substring match를 통해 비교

same_layers = {} # 동일한 layer들 모음
differences = [] # 서로 다른 레이어들. 키밸류는 bert lm layer:bert layer로 한다.

for layer_name, layer in set(bert_state_dict.items()):
    # substring matching: BertModel 레이어에는 있지만 BertForMaskedLM에는 없는 레이어 찾기
    matching = [bert_lm_layer for bert_lm_layer in set(bert_lm_state_dict.keys()) if layer_name in bert_lm_layer]
    if matching:
        # BertForMaskedLM에 있으면 smae_layers에 키밸류로 저장
        matched_layer = matching[0]
        if bert_lm_state_dict[matched_layer].shape == layer.shape:
            same_layers[matched_layer] = layer_name
        else:
            print(f"LM layer {matched_layer} has different dim with {layer_name}")
    else:
        # BertForMaskedLM에 없으면 differences에 추가
        differences.append(layer_name)

# 각 레이어와 같은 레이어, 다른 레이어 갯수를 확인
print(len(bert_lm_state_dict), len(bert_state_dict), len(same_layers), len(differences))
```

같은 레이어는 살펴볼 필요가 없으므로 `BertModel`에만 있는 레이어를 살펴보자 (`differences` 확인).

```pycon
['pooler.dense.weight', 'pooler.dense.bias']
```

예상대로 pooler가 나왔다.
Pooler는 BertModel을 제외한 모든 모델에서 사용하지 않음에 주의하자.

그렇다면 `BertForMaskedLM`에만 있는 레이어를 살펴보자.

```py
set(bert_lm_state_dict.keys()) - set(same_layers.keys())
```

```pycon
{'cls.predictions.bias',
 'cls.predictions.decoder.bias',
 'cls.predictions.decoder.weight',
 'cls.predictions.transform.LayerNorm.bias',
 'cls.predictions.transform.LayerNorm.weight',
 'cls.predictions.transform.dense.bias',
 'cls.predictions.transform.dense.weight'}
```

예상대로 classification에 쓰는 레이어들이다.

### 2). 커스텀 BERT 뜯어보기

변환할 모델을 정하기 위해 내 모델을 살펴보도록 하자.
모델은 [여기](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py)서 확인할 수 있다.

```python
from models.nvidia_bert.modeling import BertForMaskedLM as nv_BertForMaskedLM, BertConfig as nv_BertConfig

# load nvida bert
nv_bert_path = Path("/home/jovyan/models/assets/based-model/")
config = nv_BertConfig.from_json_file(nv_bert_path / Path("config.json"))
nv_bert = nv_BertForMaskedLM(config)
nv_bert.load_state_dict(torch.load(nv_bert_path / Path('BERT-base-pretrained.pt'), map_location="cpu")["model"], strict=False)
```

불러왔으니 이제 뜯어보자.
앞서 정의한 `show_weights` 함수를 사용하자.

```py
show_weights(nv_bert)
```

```pycon
# of layers: 205
Layers' name and their dimensions are:
  bert.embeddings.word_embeddings.weight : torch.Size([35000, 768])
  bert.embeddings.position_embeddings.weight : torch.Size([512, 768])
  bert.embeddings.token_type_embeddings.weight : torch.Size([2, 768])
  bert.embeddings.LayerNorm.weight : torch.Size([768])
  bert.embeddings.LayerNorm.bias : torch.Size([768])
  ...
  bert.pooler.dense_act.bias : torch.Size([768])
  cls.predictions.bias : torch.Size([35000])
  cls.predictions.transform.dense_act.weight : torch.Size([768, 768])
  cls.predictions.transform.dense_act.bias : torch.Size([768])
  cls.predictions.transform.LayerNorm.weight : torch.Size([768])
  cls.predictions.transform.LayerNorm.bias : torch.Size([768])
  cls.predictions.decoder.weight : torch.Size([35000, 768])
```

앞서 살펴본 _`BertForMaskedLM`과 레이어 수가 같다니, 개꿀이네?_ 라고 생각하면 오산이다.
자세히 살펴보면 이전엔 없던 pooler가 있으며, `bert.embeddings.position_ids`가 없다.
따라서 비록 MLM용 weight가 있을지라도 **pooler를 불러오기 위해서는** `BertModel`을 사용해야 한다. 

레이어 수도 그렇고 모든게 일치할 것이라 생각했는데 실제로는 달랐던 것을 확인할 수 있다.
따라서 반드시 직접 확인해보자.











우선 토크나이저만 봤을 때는 큰 이상이 없으므로, 이제 모델의 state를 살펴보자.

```py
# check nv_bert's state dict
nv_state_dict = nv_bert.bert.state_dict()
print(len(nv_state_dict))
for key in nv_state_dict.keys():
    print(key)

# check transformers' bert's state dict
state_dict = bert.state_dict()
print(len(state_dict))
for key in state_dict.keys():
    print(key)
```

결과

```
199
embeddings.word_embeddings.weight
embeddings.position_embeddings.weight
embeddings.token_type_embeddings.weight
embeddings.LayerNorm.weight
embeddings.LayerNorm.bias
...

200
embeddings.position_ids : embeddings.position_ids
embeddings.word_embeddings.weight : embeddings.word_embeddings.weight
embeddings.position_embeddings.weight : embeddings.position_embeddings.weight
embeddings.token_type_embeddings.weight : embeddings.token_type_embeddings.weight
embeddings.LayerNorm.weight : embeddings.LayerNorm.weight
embeddings.LayerNorm.bias : embeddings.LayerNorm.bias
...
```

크게 두 가지 차이점이 보인다. 첫번째는 이름이 약간씩 다르다는 것이다. 따라서 이를 바탕으로 두 모델의 submodule을 맵핑해주면 된다. 두번째 차이점은 레이어의 수가 huggingface에서 제공하는 bert가 1개 더 많다는 것. 그리고 이의 이름은 `position_ids`로 보인다. 

이를 자세히 살펴보기 위해 `transformers`의 [modeling_bert.BertEmbeddings](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L180)을 보도록 하자.

```py
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
```

보아하니 position embedding 부분에 대해 forward에서 인풋으로 받지 않으면 이를 자동적으로 생성해주는 것으로 보인다. 그렇다면 우리의 모델은 어떻게 되어있을까? [nvidia bert의 `BertEmbeddings`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py#L285)를 보도록 하자.

```py
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    distillation : Final[bool]
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.distillation = getattr(config, 'distillation', False)
        if self.distillation:
            self.distill_state_dict = OrderedDict()
            self.distill_config = config.distillation_config
        else :
            self.distill_config = {'use_embedding_states' : False }

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
```

여기서는 `position_ids`에 대한 레이어가 없고, 이를 직접 생성한 뒤 position embedding을 만드는 것으로 보인다.

개인적으로는 nvidia에서 제공하는 것이 더 효율적으로 보이나, 우리의 목표는 어쨋든 이를 변환하는 것이기 때문에 변환하도록 한다.

`position_ids`는 trainable parameter가 아니라 문장 내 토큰의 인덱스를 생성하는 레이어이기 때문에 큰 노력없이 이를 만들 수 있다.

따라서 우리의 nvidia bert에다가 다음과 같이 레이어를 추가하도록 한다.

```py
# add position_ids
nv_bert.bert.embeddings.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
nv_state_dict = nv_bert.bert.state_dict()
print(len(nv_state_dict)) # 199 -> 200
```

이렇게 하면 레이어가 추가된 모습을 볼 수 있다.

자 이제는 맵핑만이 남았다. Nvidia에서 제공하는 BERT또한 huggingface에 많이 의존을 하기 때문에 레이어의 이름이 거의 비슷하다. 따라서 쉽게 맵핑할 수 있었다. 이런 경우가 아니라면 일일이 dict을 만든 후 직접 만든 BERT의 레이어 이름과 huggingface 레어이 이름을 맵핑해야 할 것이다.

```py
nv_state_dict = nv_bert.bert.state_dict()

for key in nv_state_dict.copy():
    if 'intermediate.dense_act.' in key:
        val = nv_state_dict.pop(key)
        new_key = key.replace('intermediate.dense_act.', 'intermediate.dense.')
        nv_state_dict[new_key] = val
    elif 'pooler.dense_act.' in key:
        val = nv_state_dict.pop(key)
        nv_state_dict[key.replace('pooler.dense_act.', 'pooler.dense.')] = val
```

나의 경우엔 `intermediate.dense_act.`를 `intermediate.dense.`로, `pooler.dense_act.`를 `pooler.dense.`로 변환해주기만 하면 되기 때문에 큰 무리없이 변환을 완료하였다.

잘 되었는지 다음의 코드를 통해 확인해보자.

```py
# Show the differences
print(len(nv_state_dict), len(state_dict)) # 200, 200
print(set(nv_state_dict.keys()) - set(state_dict.keys()), set(state_dict.keys()) - set(nv_state_dict.keys())) # empty sets
```

이상이 없는 것을 확인할 수 있다.

마지막으로 저장만이 남았다. 새롭게 huggingface Bert를 만든 후, 여기에 우리의 원본 bert의 파라미터를 집어넣고, 그 후 저장할 것이다.

```py
bert_config = 'config.json path'
bert_config = BertConfig.from_json_file(bert_config)

model = BertModel(bert_config).eval()
model.load_state_dict(nv_state_dict) # create new bert and then load the parameters

model.save_pretrained("./new_model") # save model
```

자 이제 변환작업이 완료되었다. 결과물을 확인해보자.

### 4. 변환 전 결과물과 변환 후 결과물의 비교

성공적으로 변환이 완료되었는지 확인하기 위해, 변환 전의 결과와 변환 후의 결과를 비교한다. 이 둘이 같으면 변환이 성공적이라 말할 수 있다.

```py
# Load huggingface BertModel
tokenizer = BertTokenizerFast.from_pretrained('bert directory')
bert = BertModel.from_pretrained('bert directory')

# bert output
input_ids = torch.LongTensor(bert_tokens['input_ids']).view(1, -1)
attention_mask  = torch.LongTensor(bert_tokens['attention_mask']).view(1, -1)
token_type_ids  = torch.LongTensor(bert_tokens['token_type_ids']).view(1, -1)
bert_output = bert(input_ids, attention_mask, token_type_ids)[0]
bert_output, bert_output.shape

# nvidia output
input_ids = torch.LongTensor(bert_tokens['input_ids']).view(1, -1)
attention_mask  = torch.LongTensor(bert_tokens['attention_mask']).view(1, -1)
token_type_ids  = torch.LongTensor(bert_tokens['token_type_ids']).view(1, -1)
nvidia_outputs = nv_bert(input_ids, token_type_ids, attention_mask)[0]
nvidia_outputs, nvidia_outputs.shape
```

결과

```
# bert result
(tensor([[[ 0.3003,  1.0346, -0.5846,  ...,  2.0145,  0.4735, -1.1030],
          [ 0.1724,  1.4453, -0.0366,  ...,  0.5263,  0.7330, -0.6834],
          [ 0.0622,  1.3580, -0.1182,  ...,  0.4719,  0.7171, -0.5882],
          ...,
          [ 0.0336,  1.4403, -0.0711,  ...,  0.3191,  0.6646, -0.7537],
          [ 0.2466,  1.5484,  0.0144,  ..., -0.0041,  0.6202, -0.7716],
          [ 0.4875,  1.0405, -0.5882,  ...,  1.8235,  0.5350, -0.9061]]],
        grad_fn=<NativeLayerNormBackward>),
 torch.Size([1, 9, 768]))

# nvidia result
(tensor([[[-1.8637, -1.2017,  0.7798,  ..., -1.0716,  0.4124, -1.2521],
          [-0.4457, -1.1522, -0.0286,  ..., -0.9415, -0.8271, -1.1542],
          [-1.3970,  0.5442,  0.0069,  ..., -1.4798, -1.1996, -1.4399],
          ...,
          [-0.5910,  0.0696, -0.0943,  ..., -1.2924, -0.4478, -1.3587],
          [ 0.2626,  0.5570, -1.0558,  ..., -0.6793, -0.7356, -1.8436],
          [-0.1291, -0.1396, -0.0704,  ..., -0.8915, -0.7698, -0.8588]]],
        grad_fn=<AddBackward0>),
 torch.Size([1, 9, 35000])) 
```

아뿔싸... nvidia bert에는 LM head가 달려있어서 동일한 아웃풋을 내지 않는다. 따라서 새롭게 함수를 짜서 final layer만 뽑아내도록 해야한다.

나의 경우는 이게 귀찮아서 레이어 몇개만 뽑아서 직접 비교해보았다. 레이어의 이름과 파라미터는 `state_dict()` 함수를 통해 뽑아낼 수 있다. 직접 비교하였더니 동일한 것을 확인하였고, 이를 통해 포팅이 성공적으로 끝난 것을 알 수 있었다.
