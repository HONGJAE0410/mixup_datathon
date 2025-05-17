# 🧪 MixUp_감자탕 : Grammar Error Correction Promptathon 

본 레포지토리는 Grammar Error Correction Promptathon  실험을 재현하고 확장하기 위한 코드 및 가이드를 제공합니다.


## 📌 프로젝트 개요

* **목표**: 멀티턴 프롬프트와 RAG를 활용해 Solar Pro API의 한국어 맞춤법 교정 성능을 개선한다. 
* **접근 전략**:
    
    1) 한국어 문법 오류 유형을 정의하여 교정을 실시한 뒤, EDA를 통해 주로 발견되는 오류를 한 번 더 교정하도록 요구하는 **멀티턴 구성**
    2) **RAG** 기반 train 데이터의 유사 문장 검색 후 **few shot 활용**

* **주요 실험 내용**:

  * ***실험 진행 방식 작성***
    1) 싱글 턴을 사용하여 프롬프트 엔지니어링 (문법 오류 정의 / CoT 설계 / Few-Shot) 
    2) RAG 전후의 성능 변화 확인 
    3) EDA를 통해 싱글턴으로 교정하지 못한 오류 포착
    4) 멀티 턴으로 교정하지 못한 오류에 집중하여 교정
    
---

## ⚙️ 환경 세팅 & 실행 방법

### 1. 사전 준비 

```bash
git clone https://github.com/HONGJAE0410/mixup_datathon.git
cd your-repo/code
```

### 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 실험 실행

```bash
cd ..
>Mac : export PYTHONPATH=$(pwd)
>Window : set PYTHONPATH=%cd%
python code/main.py
```

> 입력 파일:
> - `./data/train.csv`: 훈련 데이터 (err_sentence, cor_sentence 포함)
> - `./data/test.csv`: 테스트용 문장 목록 (id, err_sentence 포함)

> 출력 파일:
> - `submission_baseline.csv`: 교정 결과 저장 파일


---


## 🚧 실험의 한계 및 향후 개선

* **한계**:

  * LLM이 자체적으로 **교정 방법의 해설**을 제공한 경우가 간혹 발생 
  * 하나의 문장을 처리하는데 **8초 정도** 소요


* **향후 개선 방향**:

  * 해설을 제공하는 경우, 특수기호와 함께 설명하는 경우가 많았으므로 **정규표현식 등의 후처리 방법**을 통해 배제
  * **배치를 나누어 처리**함으로써 시간 소요 감소

---

## 📂 폴더 구조

```
📁 code/
├── main.py              # 메인 실행 파일
├── config.py            # 설정 파일
├── requirements.txt     # 필요한 패키지 목록
├── __init__.py         # 패키지 초기화 파일
├── utils/              # 유틸리티 함수들
│   ├── __init__.py     # utils 패키지 초기화
│   ├── experiment.py   # 실험 실행 및 API 호출
│   └── metrics.py      # 평가 지표 계산
└── prompts/            # 프롬프트 템플릿 저장
    ├── __init__.py     # prompts 패키지 초기화
    └── templates.py    # 프롬프트 템플릿 정의
```
