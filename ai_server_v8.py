from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import keras
import pickle
from typing import List

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings('ignore')  # 서버 상단에 추가

@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model=64, num_heads=4, ff_dim=128, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="gelu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(d_model)
        ])
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


app = FastAPI()

print("🧠 BOLA 탐지 모델(v8) 로딩 중...")
model = keras.models.load_model(
    'adaff_transformer_v8.keras',
    custom_objects={'TransformerBlock': TransformerBlock}
)
with open('adaff_artifacts_v8.pkl', 'rb') as f:
    artifacts = FlexibleUnpickler(f).load()
    scaler = artifacts['scaler']

print("✅ 모델 및 스케일러 로드 완료. 서버 시작!")

THRESHOLD = 0.9000

class TrafficSequence(BaseModel):
    features: list  # 20 x 19


class BatchSequence(BaseModel):
    samples: List[list]  # N x 20 x 19


@app.post("/predict")
def predict_bola(data: TrafficSequence):
    try:
        seq = np.array(data.features)
        seq_scaled = scaler.transform(seq)
        pred = model.predict(np.expand_dims(seq_scaled, axis=0), verbose=0)
        score = float(pred[0][0])
        is_attack = score >= THRESHOLD
        if is_attack:
            print(f"🚨 [방어막 가동] 위험도: {score:.4f} -> 즉각 차단!")
        return {"is_attack": is_attack, "score": score}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_batch")
def predict_batch(data: BatchSequence):
    """여러 샘플을 한 번에 처리 - 테스트용"""
    try:
        batch = np.array(data.samples)          # (N, 20, 19)
        N = batch.shape[0]
        # 샘플별로 scaler 적용 (StandardScaler는 2D만 처리)
        batch_scaled = np.stack([scaler.transform(batch[i]) for i in range(N)])
        preds = model.predict(batch_scaled, verbose=0)  # (N, 1)
        scores = preds[:, 0].tolist()
        results = [
            {"is_attack": float(s) >= THRESHOLD, "score": round(float(s), 4)}
            for s in scores
        ]
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}