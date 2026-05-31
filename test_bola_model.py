"""
BOLA 탐지 모델 테스트 스크립트 (배치 버전 - 빠름)
"""
import numpy as np
import requests

SERVER_URL = "http://localhost:8000"
BATCH_SIZE = 64

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

attack_idx = np.where(y_test == 1)[0]
normal_idx = np.where(y_test == 0)[0]

print("=" * 60)
print("  BOLA 탐지 모델 테스트")
print(f"  전체: {len(X_test)}개  공격: {len(attack_idx)}개  정상: {len(normal_idx)}개")
print("=" * 60)

# 공격 5개 + 정상 5개 샘플 미리보기
preview = [(i, "🚨 공격") for i in attack_idx[:5]] + [(i, "✅ 정상") for i in normal_idx[:5]]
for idx, label in preview:
    resp = requests.post(f"{SERVER_URL}/predict", json={"features": X_test[idx].tolist()})
    r = resp.json()
    pred = "🚨 공격" if r["is_attack"] else "✅ 정상"
    mark = "⭕" if pred == label else "❌"
    print(f"  [{mark}] 실제: {label} | 예측: {pred} | 확률: {r['score']*100:.1f}%  (#{idx})")

print("=" * 60)

# 전체 배치 평가
print("\n📊 전체 테스트셋 평가 중...")
all_preds = []
for start in range(0, len(X_test), BATCH_SIZE):
    batch = X_test[start:start+BATCH_SIZE].tolist()
    resp = requests.post(f"{SERVER_URL}/predict_batch", json={"samples": batch})
    results = resp.json()["results"]
    all_preds.extend([r["is_attack"] for r in results])
    print(f"  {min(start+BATCH_SIZE, len(X_test))}/{len(X_test)} 완료...", end="\r")

print()
tp = fp = tn = fn = 0
for pred, actual in zip(all_preds, y_test):
    if pred and actual:       tp += 1
    elif pred and not actual: fp += 1
    elif not pred and actual: fn += 1
    else:                     tn += 1

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n  정밀도 (Precision) : {precision*100:.1f}%")
print(f"  재현율 (Recall)    : {recall*100:.1f}%")
print(f"  F1 Score           : {f1*100:.1f}%")
print(f"\n  TP(공격→공격): {tp}  FP(정상→공격): {fp}")
print(f"  FN(공격→정상): {fn}  TN(정상→정상): {tn}")