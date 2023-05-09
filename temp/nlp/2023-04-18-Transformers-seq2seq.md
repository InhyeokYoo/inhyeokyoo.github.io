---
title:  "transformers: seq2seq"
toc: true
toc_sticky: true
permalink: /project/nlp/transformers/seq2seq
categories:
  - NLP
  - transformers
tags:
  - seq2seq
use_math: true
last_modified_at: 2023-03-26
---

## 들어가며

- 패딩토큰을 -100으로 설정해 loss function이 이를 무시하도록 함
  - `transformers.DataCollatorForSeq2Seq`에서 알아서해줌
- Baseline으로 기사의 맨 처음 세 문장을 선택하기도 함
- GPT-2의 경우 TL:DR을 통해 요약을 진행하기도 함
- T5: "summarizae: <ARTICLE>"

## 평가지표

### BLEU

- 블루라 읽음
- 단어/n-gram을 체크
- 정밀도를 근간으로 함
- 두 텍스트를 비교할 때 정답 텍스트에 있는 단어가 생성된 텍스트에 얼마나 자주 등장하는지 카운트한 뒤 생성된 텍스트 길이로 나눔
- 반복적인 생성에 보상을 주지 않도록 분자의 count를 clip
  - 생성된 문장에서 ngram 등장 횟수는 참조 문장에 나타난 횟수보다 작게 됨
- 재현율을 고려하지 않기 때문에 짧지만 정밀하게 생성된 시퀀스가 긴 문장보다 유리
- 마지막 항은 1에서 N까지 n-gram에서 수정 정밀도의 기하 평균
- BLUE-4 점수가 실제로 많이 사용

단점
- 동의어 고려 X
- 토큰화된 텍스를 기대하기 때문에 다른 토크나이징을 사용하면 결과가 달라짐


### SacreBLEU

- BLEU에 토큰화 단계를 내재화해 문제 해결

### ROUGE

- 루주([ro͞oZH])라 발음


## CNN/DailyMail

- 300K의 (뉴스기사, 요약) pair로 구성
- 요약은 기사에 첨부한 글머리 목록의 내용인데, 본문에서 추출된 것이 아니라 새로운 문장으로 구성



{: .align-center}{: width="300"}
