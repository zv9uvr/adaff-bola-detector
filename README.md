# 🛡️ ADAFF v8 — AI 기반 BOLA 공격 탐지 시스템

BOLA(Broken Object Level Authorization) 공격을 실시간으로 탐지하는 AI 방어 시스템입니다.
Transformer 기반 모델이 사용자의 API 요청 패턴을 분석해 공격 여부를 판별합니다.

---

## 📁 파일 구조

```
📁 parsing/
├── ai_server_v8.py                ← FastAPI AI 추론 서버
├── bola.interceptor.ts            ← NestJS 방어 인터셉터 (실서비스 연동용)
├── adaff_transformer_v8.keras     ← 학습된 Transformer 모델
├── adaff_artifacts_v8.pkl         ← 전처리 스케일러 및 피처 메타데이터
├── find_best_threshold.py         ← 최적 임계값 탐색 스크립트
├── test_bola_model.py             ← 모델 성능 테스트 스크립트
├── X_test.npy                     ← 테스트 입력 데이터 (1,233개 시퀀스)
├── y_test.npy                     ← 테스트 라벨 (공격 156개 / 정상 1,077개)
└── 환경세팅_가이드.md              ← 설치 가이드
```

---

## ⚙️ 환경 설정

**Python 3.11 필수**

```bash
py -3.11 -m pip install fastapi uvicorn tensorflow keras scikit-learn numpy requests
```

---

## 🚀 서버 실행

```bash
py -3.11 -m uvicorn ai_server_v8:app --port 8000
```

성공 시 출력:
```
✅ 모델 및 스케일러 로드 완료. 서버 시작!
INFO: Uvicorn running on http://127.0.0.1:8000
```

Swagger UI 확인: http://127.0.0.1:8000/docs

---

## 🔌 API 엔드포인트

### `POST /predict`
단일 시퀀스 공격 여부 판별

**Request**
```json
{
  "features": [[...19개 피처...] × 20개 윈도우]
}
```

**Response**
```json
{
  "is_attack": true,
  "score": 0.9923
}
```

### `POST /predict_batch`
다수 시퀀스 일괄 판별 (테스트용)

**Request**
```json
{
  "samples": [ [[...19개 피처...] × 20], ... ]
}
```

**Response**
```json
{
  "results": [
    { "is_attack": true, "score": 0.9923 },
    { "is_attack": false, "score": 0.0012 }
  ]
}
```

---

## 🧪 모델 테스트

서버 실행 상태에서:
```bash
py -3.11 test_bola_model.py
```

출력 예시:
```
[⭕] 실제: 🚨 공격 | 예측: 🚨 공격 | 확률: 99.5%  (#770)
[⭕] 실제: ✅ 정상 | 예측: ✅ 정상 | 확률: 0.0%   (#0)

정밀도 (Precision) : 88.9%
재현율 (Recall)    : 82.1%
F1 Score           : 85.3%
```

---

## 🎯 최적 임계값 탐색

서버 없이 로컬에서 직접 실행:
```bash
py -3.11 find_best_threshold.py
```

임계값별 정밀도/재현율/F1 비교표와 최적값을 출력합니다.
탐색 결과에 따라 `ai_server_v8.py`의 `THRESHOLD` 값을 수정하세요.

---

## 📊 모델 성능 (임계값 0.90 기준)

| 지표 | 값 |
|------|----|
| 정밀도 (Precision) | 88.9% |
| 재현율 (Recall) | 82.1% |
| F1 Score | 85.3% |
| TP (공격 → 공격) | 128 |
| FP (정상 → 공격) | 16 |
| FN (공격 → 정상) | 28 |
| TN (정상 → 정상) | 1,061 |

---

## 🏗️ 시스템 구조

```
실제 사용자 요청
      ↓
 NestJS 서버
 bola.interceptor.ts
  ├─ 고위험 경로 확인 (LLM 생성 룰)
  ├─ 최근 20개 요청 피처 추출
  └─ AI 서버로 판별 요청
      ↓
 FastAPI AI 서버 (ai_server_v8.py)
  ├─ Transformer 모델 추론
  └─ 공격 확률 반환
      ↓
 공격이면 → 403 차단
 정상이면 → 요청 통과
```

---

## 📐 입력 피처 (19개)

| 피처 | 설명 |
|------|------|
| path_var_short/mid/long | 경로 변수 다양성 (단기/중기/장기) |
| path_ent_short/mid/long | 경로 엔트로피 (단기/중기/장기) |
| status_ent_short/mid/long | 응답 상태 엔트로피 |
| path_change | 경로 변경 빈도 |
| consecutive_errors | 연속 에러 횟수 |
| is_4xx / is_5xx | 4xx/5xx 응답 여부 |
| req_rate_10s | 10초 내 요청 속도 |
| unique_paths_cumul | 누적 고유 경로 수 |
| cross_resource_ratio | 타 리소스 접근 비율 |
| path_revisit_rate | 경로 재방문 비율 |
| is_sensitive_path | 민감 경로 여부 |
| has_id_param | ID 파라미터 포함 여부 |
