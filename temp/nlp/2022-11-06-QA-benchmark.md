---
title:  "QA task와 데이터셋을 알아보자"
toc: true
toc_sticky: true
permalink: /project/nlp/QA-benchmark/
categories:
  - NLP
tags:
  - benchmark
  - TODO
use_math: true
last_modified_at: 2021-07-15
---

## 들어가며

NLP관련 논문들을 읽다보면 다양한 태스크와 데이터셋이 등장한다. 
이에 대해 모르면 모델이 어떤 태스크에 장점이 있는지 알기 힘들뿐더러, 데이터셋을 특정 태스크에 맞게 조작하거나, input을 변형해야 할 때 도움을 받기 힘들다. 
따라서 직접 모델을 구축하고 input을 넣을 때 inductive bias를 준다거나, implementation detail을 만드는게 까다로워진다. 
아직까진 한국어로 정리된 자료가 없는 것도 같으니 논문을 읽으면서 한번쯤은 봤던 데이터셋과 태스크를 정리해보도록 하자.

아래는 본 내용에 대한 summary이다.


관련정보:
- QA 데이터셋 정리: http://nlpprogress.com/english/question_answering.html
- 클라우데라 패스트 포워드: https://qa.fastforwardlabs.com/

### QA와 MRC의 차이

MRC vs. QA
https://arxiv.org/pdf/2006.11880.pdf

본 포스트를 작성하는 과정에서 QA와 MRC가 비슷하게 쓰이는 것을 확인할 수 있었다.
이 둘에 대한 차이는 다음과 같다.

QA vs. MRC
Question answering is just the broad category of which MRC is one specific part. MRC requires some context along with the request or question, while QA does not. If you ask "Who is the US president?", it is QA since there is no context to read and comprehend. If you add a paragraph of context that somehow contains the answer, then it can be considered an MRC task.

MRC tasks are a group of tasks which to solve them you need the ability to read some text
QA tasks are a group of tasks which to solve them you have to answer a question

QA tasks can be solved in different ways, as we saw before (Retriever-Generator, Retriever-Extractor, Generator, and other). Sometimes a QA task is solved with a MRC technique, this is the reason for the convergence.

Retriever-Extractor and Retriever-Generator, for example, are solutions that fall in this case because they required the ability to read a text. Instead, the solution that is made up only of the Generator solve a QA problem without requiring this skill (so without using a MRC solution).

Just to be clear: of course, not all the MRC solutions are made up to solve QA problems. MRC is the ability to comprehend a text by reading it. This ability can be tested with different tasks, which can also be funded in other NLP problems. If while I am solving a QA task I find a MRC problem, I of course use the MRC solutions that fit well to solve it.

Another interesting thing that I found interesting in that article is the interesting classification of MRC tasks showed below.

https://alessandro-lombardini.medium.com/summary-of-question-and-answering-task-889d5cf70017



## dd

Reading comprehension

Most current question answering datasets frame the task as reading comprehension where the question is about a paragraph or document and the answer often is a span in the document. The Machine Reading group at UCL also provides an overview of reading comprehension tasks.

http://nlpprogress.com/english/question_answering.html#reading-comprehension

## Machine Reading Comprehension

Reading Comprehension은 말 그대로 독해로, 기계가 주어진 문맥을 이해하는 것이다. 대부분의 QA 태스크가 MRC에 해당한다. 이는 문서나 문장에 대한 이해를 하지 않고서는 질문에 답변을 할 수 없기 때문이다.

## Question Answering

Question asnwering은 문단/문서와 관련된 질문을 주고, 이에 해당하는 연속적인 단어 (span)을 찾도록 하는 것이다. 

이에는 SQuAD, HotPotQA, bAbI, TriviaQA, WikiQA등이 있다.

Question answering can be segmented into domain-specific tasks like community question answering and knowledge-base question answering. Popular benchmark datasets for evaluation question answering systems include SQuAD, HotPotQA, bAbI, TriviaQA, WikiQA, and many others. Models for question answering are typically evaluated on metrics like EM and F1. Some recent top performing models are T5 and XLNet.

> The simplest type of question answering systems are dealing with factoid questions (Jurafsky and Martin, 2008). The answer of this type of questions are **simply one or more words which gives the precise answer of the question**. For example questions like “What is a female rabbit called?” or “Who discovered electricity?” are factoid questions. Sometimes the question asks for a body of information instead of a fact. For example questions like “What is gymnophobia?” or “Why did the world enter a global depression in 1929?” are of these type.To answer these questions typically a summary of one or more documents should be given to the user. (Loni, 2014)

> A non-factoid question answering (QA) is an umbrella term that covers all question-answering topics beyond factoid question answering. As a quick reminder: a factoid QA is about **providing concise facts**. For example, "who is the headmaster of Hogwarts?", "What is the population of Mars", and so on, so forth. [Quora: Natural Language Processing: What is "Non-factoid question answering"?](https://www.quora.com/Natural-Language-Processing-What-is-Non-factoid-question-answering)

> We find it encouraging that the model can remember facts, understand  contexts, perform  common  sense  reasoning without the complexity in traditional pipelines.  What surprises  us  is  that  the  model  does  so  without  any  explicit knowledge representation component except for the parameters in the word vectors. Perhaps  most  practically  significant  is  the  fact  that  the model can  generalize to  new  questions.   In  other  words, it does not simply look up for an answer by matching the question with  the existing database.   In fact,  most of the questions presented above, except for the first conversation,do not appear in the training (Vinyals & Le, 2015).


### SQuAD v1.1

The Stanford Question Answering Dataset consists of 107,785 question-answer pairs on 536 articles. The text passages are taken from Wikipedia across a wide range of topics, and the question-answer pairs themselves are human annotated via crowdsourcing.

SQuAD is notable in that while the answers are contained verbatim within the corresponding text passage, they need not be entities and sets of candidates answers are not provided. This makes SQuAD the first large scale QA dataset where answers are spans of text, which must be identified without additional clues. 

> Conventionally, a computer consists of at least one processing element, typically a central processing unit (CPU), and some form of memory. The processing element carries out arithmetic and logic operations, and a sequencing and control unit can change the order of operations in response to stored information. Peripheral devices allow information to be retrieved from an external source, and the result of operations saved and retrieved.
> 
> In computer terms, what does CPU stand for?
> 
> What are the devices called that are from an external source?
> 
> What are two things that a computer always has? 

### SQuAD v2.0

### RACE (ReAding Comprehension dataset from Examinations)

RACE는 지문과 질문, 그리고 네개의 보기를 주고 정답을 맞추게끔 되어있다. 
이를 위해 각각의 보기를 지문, 질문과 concat하고, `[CLS]` token을 이용하여 정답인지 아닌지를 예측하도록 하였다.
질문, 정답의 경우 128이하로 맞춰주어 지문과 합쳤을 때 512이하의 길이를 갖게끔 하였다.

### Natural Questions (NQ)

구글에서 공개한 데이터 셋으로 위키피디아 article 하나와 이에 대한 질문으로 이루어져있다.
목표는 아티클 내 몇 단어로 이루어진 **짧은 답변** 과 문단(paragraph)전체로 이루어진 **긴 답변** 을 찾는 것이다.
답은 존재하지 않을 수도 있다.
성능은 사람이 만든 정답셋과 모델 결과에 대해 F1 score를 계산하여 평가한다.

## HotpotQA

여러개의 context에서 evidence를 잘 조합하여 질문에 대한 답을 하는 데이터셋이다.
실험에서는 HotpotQA의 distractor setting을 사용했는데, 이에는 10개의 문단이 주어지고 이 중 2개만 유용한 정보를 포함하며 나머지는 distractor로 이루어져있다.
태스크는 질문에 대한 답변은 물론 문장단위(sentence granularity)의 evidence도 검증한다.

## WikiHop

HotpotQA와 비슷한 형태로, 복수개의 context가 위키피디아 복수개의 article 일부에 해당한다.
목표는 특정 객체(entity)에 대한 성질을 찾는 것인데, article에는 이에 대한 설명이 주어지지 않는다.
데이터는 질의, 정답 후보군(multi-hop), 정답 후보에 대한 context로 이루어져있다.

## SubjQA

- 여섯 분야의 제품과 서비스에서 10K개의 영어 고객 리뷰로 구성
  - 트립어디바이저, 음식점, 영화, 책, 전자 제품, 식료품


Tokenizer의 결과 중 `token_type_ids`는 context(1)와 질문(0)을 구분해준다.

pipeline에 `handle_impossible_answer`파라미터를 통해 정답이 없는 문제를 해결할 수 있음.

로짓에 argmax를 적용할 경우 context 대신 질문에 속한 토큰을 선택해 범위를 벗어날 수 있음.
따라서 context 범위 내에 있는지, 시작 인덱스 < 종료 인덱스인지 등의 제약조건을 걸어줘야 함.

F1-Score에만 의존하면 결과가 왜곡.
EM과 F1 사이에서 균형을 잡아야 함


{: .text-center}
