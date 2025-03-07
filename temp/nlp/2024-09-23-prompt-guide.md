---
title:  "프롬프트 엔지니어링 정리"
toc: true
toc_sticky: true
categories:
  - LLM
  - Prompt Engineering
  - Large Language Model
tags:
use_math: true
last_modified_at: 2024-09-23
---

## 들어가며

https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering

https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-completions#meta-prompts



## Basics

Azure의 경우 system, user, assistant로 나뉨:

### Instructions

모델에게 동작을 지시하는 것.


시스템 프롬프트, 메타 프롬프트, 그리고 태스크 프롬프트는 모두 모델의 동작을 조정하는 데 중요한 역할을 하며, 각자의 역할과 관계가 다릅니다. 이 셋은 상호 보완적으로 작용하여 모델이 특정 작업을 효과적으로 수행할 수 있도록 돕습니다.

1. 역할과 범위
시스템 프롬프트: 모델의 기본적인 역할과 태도를 설정하는 상위 지침입니다. 이 프롬프트는 전체 대화나 작업에서 모델의 일관된 동작 방식을 정의합니다. 시스템 프롬프트는 모델의 전반적인 성격, 목표, 또는 역할(예: 친절한 어시스턴트, 전문가 등)을 설정합니다.

예시: "You are a helpful and polite assistant who always provides clear and accurate information."
메타 프롬프트: 시스템 프롬프트가 설정한 틀 내에서, 모델이 특정 작업이나 질문에 대해 어떻게 대응할지를 더 구체적으로 지시합니다. 메타 프롬프트는 모델이 특정한 방식으로 작업을 처리하도록 유도하며, 시스템 프롬프트보다 더 구체적이고 특정한 상황에 초점을 맞춥니다.

예시: "When responding to a question, first summarize the key points, then provide a detailed explanation."
태스크 프롬프트: 모델이 수행해야 하는 특정 작업이나 질문을 명확하게 지시하는 프롬프트입니다. 태스크 프롬프트는 모델이 특정한 결과나 출력을 생성하도록 직접적으로 지시하며, 가장 구체적인 프롬프트 유형입니다.

예시: "Translate the following text into French: 'Good morning, how are you?'"
2. 관계
시스템 프롬프트는 상위 프레임워크: 시스템 프롬프트는 모델이 전체 대화나 작업에서 어떤 역할을 수행해야 하는지에 대한 상위 프레임워크를 설정합니다. 이것은 모델이 모든 작업에서 어떻게 행동할지에 대한 기본적인 가이드라인을 제공합니다.

메타 프롬프트는 구체적 행동 지침: 메타 프롬프트는 시스템 프롬프트의 가이드라인에 따라 특정 작업을 수행할 때의 행동 방식을 구체화합니다. 즉, 시스템 프롬프트가 "어떻게 행동해야 하는가?"를 정의한다면, 메타 프롬프트는 "어떻게 이 작업을 수행할 것인가?"를 정의합니다.

태스크 프롬프트는 직접적인 작업 지시: 태스크 프롬프트는 메타 프롬프트가 정의한 방식에 따라 모델이 구체적으로 어떤 작업을 수행해야 하는지 직접적으로 지시합니다. 이는 특정한 질문에 답하거나, 명령을 수행하거나, 작업을 완료하는 데 사용됩니다.

3. 예시로 본 관계
시스템 프롬프트:

"You are a friendly and knowledgeable travel advisor."
메타 프롬프트:

"When asked about a travel destination, first provide an overview of the place, then highlight key attractions, and finally give practical advice for travelers."
태스크 프롬프트:

"Describe the top attractions in Paris for first-time visitors."


## Meta prompting

개요
논문: Meta Prompting for AI Systems (Zhang et al., 2024)


소개
태스크와 문제의 구체적인 세부 내용보다는 구조적(structural)/구문적(syntactical) 측면에 집중하는 프롬프팅 기법
기존의 내용 중심 방법보다 정보의 형태와 패턴을 강조하는 것



주요 특징

구조 지향적(Structure-oriented): 구체적인 내용보다는 문제와 해결책의 형식과 패턴을 우선시
구문 중심(Syntax-focused): 구문을 예상되는 반응이나 해결책의 지침으로 사용함
추상적 예(Abstract example): 특정 세부 사항에 초점을 맞추지 않고 문제와 해결책의 구조를 설명하는프레임워크로서 추상적인 예를 사용
다재다능함(Versatile): 다양한 도메인에 적용 가능하며 광범위한 문제에 대해 구조화된 응답을 제공
범주적 접근 방식(Categorical approach): 프롬프트의 구성 요소 중 범주화와 논리적 배열을 강조하기 위해 유형 이론(type theory)을 활용










{: .align-center}{: width="300"}
