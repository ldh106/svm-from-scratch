# 📊 SVM (Support Vector Machine) from Scratch

> 빅데이터 수학 프로젝트 (3인 팀)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![SVM](https://img.shields.io/badge/Algorithm-SVM_%2B_SMO-FF6F00?style=flat-square)
![No Library](https://img.shields.io/badge/sklearn.svm-Not_Used-critical?style=flat-square)

> ⚠️ **이 레포는 코드 구현 파트를 정리한 것입니다.** 과제 조건상 `sklearn.svm`,
> `fitcsvm` 등 기성 SVM 함수 사용이 금지되어, Soft-margin SVM과 SMO 알고리즘을
> 표준 라이브러리만으로 직접 구현했습니다. 보고서(이론 정리, 실험 분석,
> 결론)는 팀원들과 함께 작성했으며, 본인은 코드 구현과 개요 작성을
> 담당했습니다.

## 📌 프로젝트 목적

기존 머신러닝 라이브러리 없이 SVM을 직접 구현하여, 마진 최대화 문제와 제약
최적화 이론(라그랑주, KKT 조건)이 실제 알고리즘·코드 수준에서 어떻게
구현되는지 확인하는 것이 목표입니다. 정확도 경쟁이 아니라 **SVM이 제약
최적화 문제로 모델링되는 이유와 SMO의 동작 원리를 검증**하는 데 중점을
두었습니다.

## 🧩 문제 정의

상담 기반 서비스에서 고객의 **구매(전환) 여부**를 예측하는 이진 분류
문제로 정의했습니다.

| 변수 | 의미 |
|---|---|
| `response_time_min` | 상담 응답 시간 (분) |
| `quote_amount_10k` | 견적 금액 (만원 단위) |
| `qna_count` | 상담 질문-응답 횟수 |
| `converted` | 구매 전환 여부 (1=전환, 0=비전환) — 레이블 |

학습 데이터 80개(`train_set.csv`)로 학습하고, 레이블 없는 20개
(`test_set.csv`)에 대해 예측을 수행합니다.

## 🧮 구현 방법 (수식 ↔ 코드 대응)

**Primal Problem (Soft-margin)**

```
min(w,b,ξ)  1/2‖w‖² + C·Σξ_i
s.t.        y_i(wᵀx_i + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
```

**Dual Problem (SMO가 최적화하는 대상)**

```
max(α)  Σα_i - 1/2·ΣΣ α_i α_j y_i y_j K(x_i, x_j)
s.t.    0 ≤ α_i ≤ C,  Σα_i y_i = 0
```

| 단계 | 수식 | 코드 위치 |
|---|---|---|
| 결정 함수 | `f(x) = wᵀx + b` | `decision_function_one()` |
| 오차 계산 | `E_i = f(x_i) - y_i` | `_E()` |
| KKT 위반 검사 | `y_iE_i < -tol & α_i<C` 등 | `fit()` 내 if문 |
| α 허용범위 [L,H] | `y_i≠y_j`/`y_i=y_j` 두 경우 | `fit()` |
| η 계산 (선형 커널) | `η = 2K_ij - K_ii - K_jj` | `fit()` |
| α_j 갱신+clip | `α_j_new = α_j - y_j(E_i-E_j)/η` | `fit()` |
| b 갱신 | `b1, b2` 중 선택/평균 | `fit()` |
| w 증분 업데이트 | `w += Δα·y·x` | `add_scaled()` |

모듈 구성: `data_utils.py`(로딩/표준화/평가) · `svm_smo.py`(SMO 핵심 로직) ·
`main.py`(학습→평가→예측 파이프라인 제어).

## 🧪 실험 결과

| 항목 | 결과 |
|---|---|
| 학습 정확도 | 93.75% (80개 중 75개 정답) |
| 혼동행렬 [[TN,FP],[FN,TP]] | [[40, 1], [4, 35]] |
| w (표준화 공간 기준) | [-2.1806, -0.8220, +0.6995] |
| b | -0.3426 |

**가중치 해석**
- 응답 시간(-2.18, 절댓값 최대): 응답이 빠를수록 전환 가능성 ↑
- 견적 금액(-0.82): 견적이 높을수록 전환 가능성 ↓
- 상담 횟수(+0.70): 상담이 활발할수록 전환 가능성 ↑

**테스트셋 예측 (20개, 결정경계 근처 케이스 포함하도록 설계)**

| 구간 | 예측 | 판정 이유 |
|---|---|---|
| 0~11번 | 1 (전환) | 응답 빠름 + 상담 횟수 많음 → f(x) > 0 |
| 12~17번 | 0 (비전환) | 응답 느림 + 견적 높음 → f(x) < 0 |
| 18~19번 | 1 (전환) | 경계 근처지만 전환 조건 충족 |

## ⚠️ 재현 검증 중 발견한 불일치

포트폴리오 정리를 위해 코드를 다시 실행하며 보고서 수치를 검증하는 과정에서,
**서포트 벡터 개수**가 보고서 기재값과 다르다는 것을 발견했습니다.

- 보고서 기재값: 15개 (전체 80개 중 18.75%)
- 재실행 결과: **23개** (전체 80개 중 28.75%) — threshold를 1e-6~0.5까지
  바꿔봐도 동일하게 나옴

정확도(93.75%), 혼동행렬, w/b 값, test 예측 결과는 보고서와 정확히
일치하는 것으로 보아 **모델 자체는 동일하게 재현됨**을 확인했습니다. 다만
서포트 벡터 개수는 팀원들이 보고서를 작성하는 과정에서 차이가 생긴 것으로
추정되며, 정확한 원인은 확인하지 못했습니다. 자세한 내용은
[NOTES.md](NOTES.md)를 참고하세요.

## 📂 파일 구조

```
svm-from-scratch/
├── README.md
├── NOTES.md            # 재현 검증 과정에서 발견한 불일치 기록
├── data_utils.py        # CSV 로딩, 표준화, 평가 지표
├── svm_smo.py            # Soft-margin Linear SVM + SMO 구현
├── main.py               # 학습 → 평가 → 예측 파이프라인
├── train_set.csv         # 학습 데이터 (80개, 과목에서 제공)
└── test_set.csv          # 테스트 데이터 (20개, 레이블 없음)
```

## ▶️ 실행 방법

```bash
python main.py
```

`train_set.csv`, `test_set.csv`가 같은 폴더에 있으면 학습 요약과 테스트
예측 결과가 콘솔에 출력됩니다.

## 🛠 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

외부 머신러닝/수치 라이브러리 없이 표준 라이브러리(`csv`, `math`, `random`)만
사용했습니다.
