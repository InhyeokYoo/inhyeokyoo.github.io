---
title:  "Text-to-SQL 정ㅣ"
toc: true
toc_sticky: true
categories:
  - LLM
  - Prompt Engineering
  - Large Language Model
tags:
  - text-to-sql
  - text2sql
use_math: true
last_modified_at: 2025-01-13
---

## 들어가며


## 물어보새 (배민)

출처:
- [AI 데이터 분석가 ‘물어보새’ 등장 – 1부. RAG와 Text-To-SQL 활용](https://techblog.woowahan.com/18144/)
- [AI 데이터 분석가 ‘물어보새’ 등장 – 2부. Data Discovery](https://techblog.woowahan.com/18362/)

문제정의부터 먼저 진행:
- 설문조사 결과 응답자의 약 95%가 데이터를 활용하여 업무를 수행
- 절반 이상의 응답자는 SQL을 업무에 활용하고 싶어도 학습 시간이 부족하거나, SQL 구문에 비즈니스 로직과 다양한 추출 조건을 반영하여 작성하는 데 어려움을 느낌
- 또한, 데이터를 적절히 추출했는지 신뢰도에 대한 고민도 있음


해결방안:
- 프로덕트 목적: **구성원의 데이터 리터러시 상향 평준화**
  - 데이터 리터러시: '데이터를 통해 유의미한 정보를 추출하고 해석하며, 신뢰성을 검증하고, 데이터 탐색과 분석을 통해 통찰을 도출하고 합리적인 의사결정을 내릴 수 있는 소통 능력'으로 정의
  - 핵심요소:
      1. 체계화: 데이터서비스실에서 제공하는 데이터 카탈로그의 테이블 메타 정보, 데이터 거버넌스의 비즈니스 용어, 그리고 검증된 데이터 마트를 활용해 데이터 정보의 일관된 체계를 수립
      2. 효율화: 우아한형제들의 비즈니스를 이해하고, 사내 정보를 쉽고 빠르게 검색할 수 있는 기술을 개발
      3. 접근성: 누구나 쉽고 빠르게 질문할 수 있도록 웹 기반이 아닌 업무 전용 메신저 슬랙의 대화창을 이용
      4. 자동화: 데이터 담당자가 없어도 365일 24시간 언제 어디서나 이용할 수 있는 자동화된 데이터 서비스를 지향


기반기술:
- LLM, RAG, LLMOps
- LLMOps: LLM의 배포, 관리, 모니터링을 위한 운영 기법과 도구들을 의미. 구체적으로는 모델 학습, 튜닝, 버전 관리, 성능 모니터링, 응답 속도 관리, 데이터 보안 및 비용 등 다양한 요소를 최적화하며, LLM 없이도 LLM 관련 엔지니어링 영역에 적용할 수 있음


아키텍처:
- Vector DB
  - 데이터 영역별 임베딩 인덱스를 적용
- RAG 기반의 멀티 체인
  - Router Supervisor 체인이 질문 의도를 파악하여 적합한 질문 유형으로 분류
  - 사용자 질문에 답변을 해줄 수 있는 멀티 체인(예: 쿼리문 생성, 쿼리문 해설, 쿼리 문법 검증, 테이블 해설, 로그 데이터 활용 안내 기능, 칼럼 및 테이블 활용 안내 기능 등)으로 매핑되어 최선의 답변을 생성
  - 멀티 체인 실행 시, 체인별 서치 알고리즘을 활용하여 retriever가 필요한 데이터를 선별적으로 추출


text-to-SQL:
- 성능이 부족함
  - 사내 도메인과 데이터 정책에 대한 이해도가 부족하고, 기본적으로 제공되는 retriever의 성능이 떨어져 불필요한 데이터가 포함
  - 환각도 발생
- '데이터 보강', '검색 알고리즘 개발', '프롬프트 엔지니어링', '실험 및 평가 시스템 구축'에 집중하여 새로운 구조를 개발


데이터 보강:
- 성능을 향상시키려면 어떤 문서를 수집하는지가 가장 중요
- NeurIPS'2023에서 발표된 'Data Ambiguity Strikes Back: How Documentation Improves GPT’s Text-to-SQL'에선 데이터 모호성 문제 개선의 중요성과 방법을 다루고 있음
  - Q. Data Ambiguity Strikes Back: How Documentation Improves GPT’s Text-to-SQL
- 이를 기반으로 테이블 메타 데이터를 풍부하게 할 방법들을 고안, 기존보다 고도화된 메타 데이터 생성 작업을 진행
  - 테이블의 목적과 특성, 칼럼의 상세 설명, 주요 값 및 키워드 등 기존에 기록되지 않았던 상세한 내용을 추가
  - 또한, 주로 사용되는 서비스와 그에 따른 질문들을 정리
  - 테이블 메타 데이터를 기반으로 테이블 *DDL* 데이터를 생성할 때 기존보다 훨씬 풍부한 DDL을 만들 수 있었음
  - 테이블 DDL 데이터 수집 작업은 사내 데이터 카탈로그 시스템이 잘 구축되어 있어, 손쉽게 정보를 추가하고 API를 통해 자동으로 최신 데이터를 수집할 수 있었음

> DDL (Data Definition Language, 데이터 정의어)
> 데이터베이스를 정의하는 언어를 말하며 데이터를 생성하거나 수정, 삭제 등 데이터의 전체 골격을 결정하는 역할의 언어를 말한다.
>
> CREATE: 데이터 베이스, 테이블 등을 생성하는 역할을 한다.
> ALTER: 테이블을 수정하는 역할을 한다.
> DROP: 데이터베이스, 테이블을 삭제하는 역할을 한다.
> TRUNCATE: 테이블을 초기화 시키는 역할을 한다.

- Domain specific 비즈니스 용어들이 많이 사용
  - 해당 용어는 서비스 또는 조직별로 다를 수 있어, 커뮤니케이션 혼동이 없도록 표준화와 관리가 필수적
  - 데이터 거버넌스를 관리하는 조직이 있어 데이터와 비즈니스 용어를 잘 관리 중이었음
  - 따라서 기존에 구축되어 있던 비즈니스 표준 용어 사전을 활용하여 Text-to-SQL 전용 비즈니스 용어집을 구축
- 데이터 분석가가 기존에 생성해놓은 높은 품질의 쿼리문과 주요한 비즈니스 질문에 대한 쿼리문을 추가로 작성하여 few-shot SQL 예제 데이터 구축
  - 그리고 각 쿼리문에 해당하는 질문을 작성하여 쿼리문-질문 데이터셋을 구축
  - SQL 예제 데이터는 새롭게 변경되는 데이터 추출 기준과 급변하는 비즈니스 상황에 대응해야하므로 각 도메인 지식에 특화된 데이터 분석가의 관리가 필요
  - 해당 부분은 향후 도메인 데이터 분석가와 원활하게 의사소통을 하고 데이터를 관리할 수 있도록 협업 체계를 구축할 예정
- 데이터는 비정형 데이터 수집 파이프라인을 통해 매일 수집
  - 따라서 시시각각 변화하는 최신 데이터 정책을 자동 수집하여 빠르게 확인하고 서비스에 반영이 가능
  - 데이터가 증가함에 따라 업데이트 속도를 향상하기위해 VectorDB에 인덱싱을 적용, 변경된 부분만 업데이트
  - Q. VectorDB에 인덱싱이란?


검색 알고리즘 개발:
- 프롬프트 활용 시, 사용자 질문과 단계마다 적합한 검색 알고리즘을 활용해 데이터를 입력시키는 것이 중요
  - 질문이 모호하거나 짧고 명확하지 않은 경우, 질문을 구체화하는 것으로 이를 해결할 수 있음
  - 반드시 비즈니스 용어를 먼저 이해해야 하므로 질문의 의도와 관련이 있는 적절한 용어를 추출해야 함
  - 비슷하지만 의도와 관련이 없는 용어를 추출하면 오히려 잘못된 질문을 만들어낼 수 있기 때문에 의도를 정확히 파악할 수 있는 정교한 검색 알고리즘이 필요
- 이후 구체화된 질문으로 쿼리 작성에 필요한 정보를 추출
  - 테이블 및 칼럼 메타 정보, 테이블 DDL, few-shot SQL 등 다양한 정보를 활용
  - 이 단계에서는 수많은 정보 중 사용자 질문에 대답하기 가장 적합한 정보를 추출하는 것이 중요함
  - 즉, 질문의 문맥을 파악하여 가장 유사도가 높은 정보를 추출하거나, 특정한 단어가 포함된 정보를 선별하는 등 다양한 검색 알고리즘을 활용하여 조합해야 함
- few-shot SQL 예시도 질문과 가장 유사한 예시를 선별
  - 유사한 예시가 없으면 새롭게 추가


프롬프트 엔지니어링
- 데이터 분석가로 페르소나를 설정
  - 설정된 페르소나에 따라 결과물의 품질이 달라질 수 있으므로 원하는 역할과 기대하는 결과물에 대해 충분한 논의가 필요
- 프롬프트 구조 설계는 'ReAct: Synergizing Reasoning and Acting in Language Models'을 따름
  - LLM 기반 ReAct 방법은 다양한 벤치마크에서 모방 학습과 강화 학습에 비해 더 높은 답변 성능을 보여줌
  - ReAct 방법은 문제 해결 과정을 위한 순차적 추론 단계(chain-of-thought, COT)와 특정 작업 수행을 위한 도구 또는 행동으로 나뉨.
  - 이 두 가지 요소를 함께 활용하면 시너지가 발생하여 단일 방법만을 사용한 답변보다 더 정확한 답변이 나오게 됨
- 물어보새는 ReAct 방법을 응용하여 쿼리 생성 프롬프트를 개발
  - 사용자 질문에 적합한 쿼리를 생성하기 위해 단계별 추론 과정(COT)을 거침
  - 또한, 사용자 질문에 적합한 데이터를 동적으로 검색하여 선별
  - 이처럼 추론 과정과 검색 과정이 결합되면서 답변이 점점 정교해짐
  - 이를통해 단순 추론 기법만을 사용할 때보다 훨씬 더 정확한 답변을 제공


Data Discovery:
- 사용자 베타 테스트에서 얻은 인사이트를 바탕으로 사용자의 다양한 질문을 이해하고 답변할 수 있는 Data Discovery 기능을 추가
  - Text-to-SQL, 쿼리문과 테이블 해설, 로그 데이터 활용 안내 답변까지 구성원의 데이터 리터러시를 향상시키는 데 도움을 주는 다양한 정보 획득 기능으로 구성


질문 이해:
- 사용자의 질문 의도를 정확히 파악하고 적절한 답을 전달할 수 있는 Router Supervisor 체인을 구현
  - LangGraph의 Agent Supervisor에서 아이디어
    - Q. Agent Supervisor이란?
- 사용자 질문 개선
  - 질문 해석 능력 개선
    - 질문이 데이터와 얼마나 연관되어 있으며, 문제 해결을 위한 구체적인 단서를 얼마나 포함하고 있는지 평가
      - 질문 유형이 LLM을 통해 분류되더라도 오류가 있을 수 있으므로, 질문의 품질을 한 번 더 평가하는 단계
    - 질문 평가 기준을 수립하고, 평가 항목별 점수를 합산하여 종합 점수를 산출
    - 프롬프트 엔지니어링 기법을 활용해 기준을 충분히 달성했는지에 대한 일관성 있는 점수를 부여
    - 벡터 스토어를 통해 사내 용어와 질문을 결합하여 추상적이거나 전문적인 질문을 쉽게 이해할 수 있는 질문으로 변경하여 질문을 점수화
    - 일정 기준 점수와 분류 모델을 통과한 경우 정보 획득 단계로 넘어감
    - 통과하지 못한 질문은 적합한 질문 예시를 참고하여 조금 더 구체적으로 질문해 달라는 안내 문구를 자동으로 제공
    - Q. 질문을 어떻게 점수화 했다는 것인지?
  - 질문 생성 능력 개선
    - 사용자의 질문 생성 능력을 향상
    - 상황에 적합한 질문을 할 수 있도록 가이드를 제공했으나, 가이드북을 제대로 읽지 않음
    - 슬랙 앱 등록 시 튜토리얼같은 활용 안내 화면을 개발하여 직관적인 사용자 가이드 제공
    - 현재 제공 중인 기능과 기능별 대표 질문 및 예시 답변을 통해 어떤 질문을 할 수 있으며 어떤 답변을 받을 수 있는지 인지시킴
- 응답 속도가 중요하므로 주로 Single-Turn을 사용
- 데이터와 무관할 질문을 Text-to-SQL로 연결하면 엉뚱한 답변이 생성
  - 따라서 질문을 우선적으로 데이터 또는 비즈니스 관련 여부에 따라 분류
    - e.g., 날씨 문의, 안부 인사 -> 일반 대화로 분류하여 별도의 처리 없이 적절한 답변을 제공
    - LLM 기반의 프롬프트 엔지니어링으로 분류
  - 데이터/비즈니스 관련 질문은 어떤 정보 획득 유형이 적합한지 재분류
    - 해설 답변을 요구하는지 검증 답변을 요구하는지에 따라 상세하게 분류
    - 여러 질문은 기능의 우선순위와 문제 해결에 가장 적합한 답변을 우선적으로 제공

정보획득:
- 해설
  - 쿼리문 해설: 해당 쿼리문에 사용된 주요 비즈니스 조건, 주요 칼럼, 최종적으로 추출되는 정보, 그리고 해당 쿼리문의 활용 방법에 대한 정보를 제공
    1. 사용자의 질문에 포함된 테이블명을 추출
      - SQLGlot + 정규 표현식 활용, 질문을 분석하고 테이블명 추출
    2. DDL 벡터 스토어에서 테이블 정보 추출
    3. 일부 칼럼에 대한 정보만 추출하는 DDL 축소 로직을 적용
      - 프롬프트가 길수록 허위 생성이 발생할 가능성이 높아짐
      - 칼럼 수가 많은 경우 토큰 제한 에러 발생
      - 최종적으로 쿼리문에서 사용된 칼럼명과 함께 테이블의 키 정보, 파티션 정보 등 주요 칼럼만 추출하는 로직을 적용하여 축소된 DDL을 프롬프트에 입력
      - 기존 CoT 한계를 개선하기 위한 Plan and Solve Prompting을 적용해 쿼리와 테이블에 대한 해석 방법을 지시
  - 테이블 해설: 주요 칼럼, 칼럼 설명, 그리고 테이블의 활용 예시에 대한 정보를 제공
- 기술
  - 쿼리 문법 검증
    - 사용자가 작성한 쿼리문 문법의 정확성을 확인하고, 필요한 경우 칼럼명, 조건 값, 실행 최적화를 위한 개선 방안을 제안
    - 길고 복잡한 쿼리에서 오류가 발생했을 때 또는 사용자가 쿼리 작성에 익숙하지 않아 어떤 부분에서 잘못되었는지 모를 때 유용
  - 데이터 기술 지원
    - 쿼리 함수(e.g., 어제 일자 구하는 함수)와 데이터 과학, DB와 관련된 데이터 전문 지식 제공
- 활용
  - 테이블 및 컬럼 활용 안내
    - LLM을 통해 테이블 메타 데이터 고도화 수행
      - 고도화된 메타 데이터에는 테이블 목적, 특성, 주요 키워드 등에 대한 정보가 포함되어 질문과 관련된 테이블을 검색하는데 유용
      - 그러나 허위 생성 문제가 발생하여 테이블에 대한 정보에 오류도 발생
    - 사용자의 질문을 이해하기 위해 비즈니스 용어 사전과 토픽 모델링을 활용한 질문 구체화 체인을 구현
      - Q. 토픽 모델링은 또 어떻게..?
      - 비즈니스 용어 사전: 서비스 구조와 용어로 LLM이 사용자의 질문을 확장
      - 토픽 모델링: DDL을 구성하는 단어들을 기반으로 토픽 모델링 수행 후 선정된 토픽과 키워드를 질문 구체화 프롬프트에 입력
  - 로그데이터 활용 안내
    - 질문을 분석, 로그 체커에 있는 데이터를 탐색
      - e.g., "가게 상세 페이지 관련 로그에 대해 알려줘" -> `ShopDet`(가게 상세 페이지) 같은 로그 단어를 추출하여 관련 로그를 찾음
    - 이전과는 달리 테이블의 DDL 정보가 아닌 로그 체커 데이터 사용
    - 로그체커:

      | 로그 이름           | 로그 설명      | Screen Name | Group      | Event      | Type  | 새롭게 정의된 로그 설명    |
      |-----------------|------------|-------------|------------|------------|-------|------------------|
      | 가게 상세 > 쿠폰 받기   | 배포 일자 기록   | ShopDet     | Cpn        | CpnDown    | Click | 가게 상세 쿠폰 다운로드 클릭 |
      | 배민스토어 > 가게카드 노출 | 배포 앱 버전 기록 | Store       | SellerCard | SellerCard | Imp   | 배민스토어 셀러 카드 노출   |

      - 테이블 DDL과는 '로그 이름', '로그 설명'만 있고, 로그에 대한 설명이 없기 때문에 Screen Name, Group, Event, Type을 바탕으로 '로그 설명'을 생성
    - 로그 단어 체인: 질문과 로그 시스템 특화 용어를 연결
      - 질문과 로그 용어 사전의 단어 간 유사도를 계산하여 유사도가 높은 로그 단어들을 선별
      - LLM을 활용, 질문과 연관성이 높은 로그 단어를 최종 선정
        - LLM을 활용한 검색 방식은 기존의 알고리즘 기반 검색보다 더 유연하고 구현이 용이
    - 로그 검색 체인: 벡터 스토어 내 로그 정보 중 사용자가 원하는 로그만 선별
      - 로그 단어 체인에서 선별된 단어로 벡터 스토어에 로그를 검색
      - LLM으로 질문과 가장 연관성이 높은 로그를 선정
      - 허위 생성을 줄이기 위해 LLM에게 로그를 구분할 수 있는 고유 키값만 출력하도록 지시
      - 고유 키값알 벡터 스토어에 검색하여 최종 로그 제공
      - 






{: .align-center}{: width="300"}
