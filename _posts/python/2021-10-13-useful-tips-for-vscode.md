---
title:  "Python사용 시 도움되는 VSCode 팁 모음"
toc: true
toc_sticky: true

categories:
  - Python
tags:
  - VSCode
use_math: true
last_modified_at: 2025-06-19
---

## 들어가며

이번 시간에는 VScode에서 유용하게 사용할 수 있는 팁을 공유해보도록 하자.

## Debugging

VScode로 디버깅하는 방법을 알아보자. 

우선 좌측 메뉴바에서 실행 및 디버그(`Ctrl+<Shift>+D`)를 눌러보자. 그러면 `launch.json`파일 만들기를 할 수 있을 것이다. 아래의 형식처럼 이름(`name`)과 파이썬파일(`program`), argparse로 받는 인자(`args`)를 넣어주도록 하자. 나의 경우엔 wikipedia 데이터 파싱 코드를 디버깅해보았다. 또한, working directory를 세팅해야 되는 경우도 있으므로, 이를 `cwd`에다가 넣어주도록 하자

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "create_wikicorpus test",
            "type": "python",
            "request": "launch",
            "program": "source_code/path/file_name.ext",
            "console": "integratedTerminal",
            "cwd": "source_code/path/",
            "args": [
                "--output_path",
                "../datasets/pre-train/wiki.txt"
            ]
        }
    ]
}
```

## Formatter, Linter, Type checker

포매터와 린터, 타입체커는 파이썬을 개발하는데 반필수적인 기능으로, 코드 가독성과 유지보수에 도움이 된다.

| 도구 종류            | 목적                                | 대표 도구               | VSCode에서의 역할                               |
| ---------------- | --------------------------------- | ------------------- | ------------------------------------------ |
| **Linter**       | 코드의 **문법 오류, 스타일 위반, 잠재적 버그**를 탐지 | `pylint`, `flake8`  | 빨간 밑줄, 경고 메시지 표시                           |
| **Type Checker** | 변수, 함수의 **타입 오류** 확인              | `mypy`, `pyright`   | `"str" expected but got "int"` 같은 타입 에러 표시 |
| **Formatter**    | **코드 자동 정렬 및 스타일 정리**             | `black`, `autopep8` | `Shift+Alt+F` 또는 저장 시 코드 자동 정리             |

여기서 린터와 타입체커는 상당히 유사해보이는데, 핵심 차이점은 다음과 같다:
  
| 항목        | Linter                         | Type Checker               |
| --------- | ------------------------------ | -------------------------- |
| **목적**    | **문법**, 스타일, 잠재적 버그 탐지         | **타입 오류** 탐지               |
| **기반**    | 코드 스타일 가이드 (PEP8 등)            | 타입 힌트 (`x: int`) 기반 정적 분석  |
| **예시**    | 사용하지 않는 변수, 들여쓰기 오류, 변수명 규칙 위반 | `str`에 숫자 더하기 같은 타입 불일치 오류 |
| **대표 도구** | `pylint`, `flake8`             | `mypy`, `pyright`          |


한번 예시를 살펴보자

```py
def add(x, y):
    return x +  y
```

- formatter: `x +  y` -> `x + y`로 y 앞 공백 제거
- linter: 사용하지 않는 변수, 잘못된 네이밍 등을 경고
- type checker: `add("1", 2) 같이 타입 안 맞을 때 오류 표시`

이들의 상세설정은 `settings.json`에서 다음과 같은 방식으로 진행할 수 있다.

```json
{
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.banditEnabled": false,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.formatting.blackArgs": [
        "--line-length",
        "140"
    ],
    "python.linting.lintOnSave": true,
    "python.linting.flake8Args": [
        "--max-line-length=140",
        "--ignore=W291",
    ],
    "git.ignoreLegacyWarning": true,
    "python.languageServer": "Pylance",
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## Setting 종류 차이

VScode에서 세팅을 하다보면 다음과 같이 Default Settings, User Settings, Workspace Settings 세 종류의 세팅이 보인다.
해당 세팅의 차이는 다음과 같다:

| 설정 종류                  | 저장 위치                             | 적용 범위          | 우선순위    |
| ---------------------- | --------------------------------- | -------------- | ------- |
| **Default Settings**   | 내장 (읽기 전용)                        | VSCode 전체의 기본값 | ❌ (최하위) |
| **User Settings**      | `settings.json` (사용자 계정)          | 모든 프로젝트에 적용    | 중간      |
| **Workspace Settings** | `.vscode/settings.json` (프로젝트 폴더) | 해당 워크스페이스에만 적용 | ✅ (최우선) |
