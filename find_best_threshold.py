"""
BOLA 탐지 모델 - 최적 임계값 탐색 스크립트
서버 없이 로컬에서 직접 모델 추론 (빠름)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import keras
import pickle

# ── 모델 로드 ──────────────────────────────────────────
@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="gelu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(d_model)])
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = x + self.drop1(self.att(self.ln1(x), self.ln1(x), training=training), training=training)
        return x + self.drop2(self.ffn(self.ln2(x), training=training), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads,
                        "ff_dim": self.ff_dim, "dropout": self.dropout_rate})
        return config

class FlexibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            dummy = type(name, (), {
                '__init__': lambda self, *a, **kw: None,
                '__getattr__': lambda self, n: None,
                '__setstate__': lambda self, s: self.__dict__.update(s) if isinstance(s, dict) else None,
            })
            return dummy

print("🧠 모델 로딩 중...")
model = keras.models.load_model(
    'adaff_transformer_v8.keras',
    custom_objects={'TransformerBlock': TransformerBlock}
)

# ── 데이터 로드 ────────────────────────────────────────
X_test = np.load("X_test.npy")   # 이미 스케일링된 데이터
y_test = np.load("y_test.npy")

# ── 전체 확률 점수 한 번에 추론 ────────────────────────
print("📊 전체 샘플 추론 중...")
scores = model.predict(X_test, batch_size=128, verbose=0).flatten()
print(f"   추론 완료! 총 {len(scores)}개\n")

# ── 임계값별 성능 비교 ─────────────────────────────────
thresholds = np.arange(0.1, 1.0, 0.05)

print(f"{'임계값':>6} | {'정밀도':>7} | {'재현율':>7} | {'F1':>7} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4}")
print("-" * 65)

best_f1 = 0
best_thresh = 0
best_row = None

rows = []
for t in thresholds:
    preds = scores >= t
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    rows.append((t, precision, recall, f1, tp, fp, fn, tn))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_row = rows[-1]

for t, p, r, f, tp, fp, fn, tn in rows:
    marker = " ◀ 현재" if abs(t - 0.8374) < 0.03 else ""
    best_marker = " ★ 최적" if abs(t - best_thresh) < 0.001 else ""
    print(f"  {t:.2f}  | {p*100:>6.1f}% | {r*100:>6.1f}% | {f*100:>6.1f}% | {tp:>4} | {fp:>4} | {fn:>4} | {tn:>4}{marker}{best_marker}")

print("-" * 65)
t, p, r, f, tp, fp, fn, tn = best_row
print(f"\n🏆 최적 임계값: {t:.2f}")
print(f"   정밀도: {p*100:.1f}%  재현율: {r*100:.1f}%  F1: {f*100:.1f}%")
print(f"   TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
print(f"\n→ ai_server_v8.py 의 THRESHOLD = {t:.4f} 으로 변경하세요")
