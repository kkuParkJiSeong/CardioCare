# CardioCare 심혈관 질환 예측 파이프라인 (Cardiovascular Risk Prediction CDSS)

이 리포지토리는 건국대학교 글로컬캠퍼스 기계학습 기말고사대체과제물로 개발된 심혈관 질환 스크리닝 파이프라인입니다. 임상 현장에서 의료진의 판단을 돕는 의사결정 보조 시스템(CDSS)을 구현하며, MLOps 구성 요소인 특성 공학, 실험 추적(MLflow), 모니터링 및 드리프트 탐지, 단위 테스트, 그리고 컨테이너화를 포함하고 있습니다.

---

## 📂 프로젝트 구조 (Project Directory Structure)

```text
CardioCare/
│
├── .github/workflows/
│   └── ci.yml             # Github Actions CI 구성 (Push 이벤트 시 테스트 자동화)
│
├── data/
│   └── processed.cleveland.data  # 심혈관 질환 원본 데이터셋
│
├── src/
│   ├── models/
│   │   └── final_model.pkl   # 학습이 완료된 최종 튜닝 파이프라인 모델 (pickle)
│   ├── preprocessing.py   # 결측값 처리, 이상치 IQR 클리핑, 임상 검증 로직
│   ├── train.py           # 모델 학습, 5-Fold 교차검증, GridSearchCV, MLflow 로깅
│   ├── inference.py       # 학습된 모델을 활용한 배치 파일 단위 추론 스크립트
│   └── monitor.py         # 드리프트 시뮬레이션, KS 검정 탐지, 시계열 시각화 스크립트
│
├── tests/
│   ├── __init__.py        # python -m unittest 자동 탐색용 패키지화 파일
│   └── test_pipeline.py   # 파이프라인 기능 검증을 위한 4가지 핵심 단위 테스트
│
├── Dockerfile             # python:3.12-slim 기반 컨테이너화 설정 파일
├── requirements.txt       # 프로젝트 패키지 의존성 목록
├── report.md              # 문제 정의, 설계 결정, 모델 비교 및 모니터링 서술형 보고서
└── README.md              # 본 안내 문서
```

---

## 🔄 주요 파이프라인 단계 (Pipeline Steps)

### 1. 데이터 전처리 및 임상값 범위 검증 (`preprocessing.py`)

### 2. 피처 엔지니어링 및 모델 학습 (`train.py`)

### 3. 배치 파일 추론 (`inference.py`)

---

## 🚀 실행 및 검증 방법 (Appendix: Verification Guide)

제출 시점에 채점자가 파이프라인을 검증할 수 있도록 제공되는 6가지 "완료" 기준 실행 방법입니다.

### 1) 최종 보고서 검토 (report.md / report.pdf)
* 프로젝트 루트 경로에 위치한 **[report.pdf](file:///d:/Programings/projects/CardioCare/report.pdf)** 파일을 열고 무엇을, 왜 만들었는지 즉시 확인할 수 있습니다.

### 2) 저장소 복제 및 의존성 설치 후 한 줄 학습 실행
저장소를 복제하고 의존성을 설치한 뒤, 한 줄 명령어를 통해 전체 데이터 로드, 전처리, 3종 모델 학습 비교 및 하이퍼파라미터 튜닝 파이프라인을 실행합니다:
```bash
# 1. 저장소 복제 및 디렉토리 이동
git clone https://github.com/kkuParkJiSeong/CardioCare.git
cd CardioCare

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 파이프라인 학습 실행
python src/train.py
```

### 3) MLflow를 통한 3개 이상의 실행 기록 및 아티팩트 확인
학습 완료 후 아래 명령어로 로컬 MLflow 대시보드를 열어 3개 이상의 실행 기록과 각각 저장된 평가지표, Confusion Matrix 이미지 아티팩트를 확인할 수 있습니다:
```bash
mlflow ui
```

### 4) 단위 테스트(Unittest)가 모두 통과하는지 확인
아래 단일 명령어로 구현된 4가지 단위 테스트(Shape 일치, 확률 범위, 임상 범위 유효성 검증, 시드 결정론 검증)가 모두 통과(`OK`)하는 것을 확인합니다:
```bash
python -m unittest
```

### 5) Docker 빌드 및 컨테이너 기반 샘플 입력 추론 실행
`Dockerfile`을 빌드한 뒤 제공되는 소량 배치 샘플 입력 파일(`sample_input.csv`)을 인풋으로 전달해 예측 결과를 정상적으로 출력받습니다:
```bash
# Docker 이미지 빌드 (태그 cardiocare:1.0)
docker build -t cardiocare:1.0 .

# sample_input.csv 에 대한 컨테이너 추론 실행 및 결과 저장 (사용 환경에 맞게 선택 실행)
# 1) Windows PowerShell:
docker run --rm -v ${pwd}:/workspace cardiocare:1.0 --input /workspace/sample_input.csv --output /workspace/sample_output.csv

# 2) macOS / Linux / Git Bash:
docker run --rm -v $(pwd):/workspace cardiocare:1.0 --input /workspace/sample_input.csv --output /workspace/sample_output.csv

# 3) Windows CMD:
docker run --rm -v %cd%:/workspace cardiocare:1.0 --input /workspace/sample_input.csv --output /workspace/sample_output.csv
```

### 6) 드리프트 모니터링 실행 (`monitor.py`)
아래 명령어를 실행하여 데이터 분포 변화(Drift)가 정상적으로 플래그(`[DRIFT DETECTED!]`)되고, 그에 상응하는 정확도 하락이 보고되는 것을 확인합니다:
```bash
python src/monitor.py
```
