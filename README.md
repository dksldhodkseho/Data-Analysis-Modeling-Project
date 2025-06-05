# 📱 스마트폰 센서 기반 모션 분류 프로젝트

> 스마트폰 센서 데이터를 활용하여 정적/동적 행동을 분류하고, 각각의 세부 행동을 예측하는 딥러닝 기반 모델 개발

![main-banner](./assets/banner.png)

---

## 🎯 프로젝트 목표

- 스마트폰 센서 데이터를 통해 **정적 / 동적 행동을 구분**
- 각각의 범주 내에서 **세부 행동 분류**
- 최종 목표: 웨어러블 및 스마트폰 앱에 활용 가능한 **실시간 행동 감지 모델 구축**

---

## 🧪 데이터 및 탐색

### 🔹 데이터 개요
- 스마트폰 내장 센서 (가속도, 자이로스코프) 기반 시계열 데이터
- 6가지 활동 label:  
  `STANDING`, `SITTING`, `LAYING`, `WALKING`, `WALKING_UP`, `WALKING_DOWN`

### 📊 주요 분석 결과

- 정적/동적 행동을 기준으로 Feature 중요도 분석  
- 전체적으로 **가속도(acc)** 기반 feature의 영향력이 높음  
- 주파수 도메인에서 추출한 feature가 핵심 지표

![feature-importance](./assets/feature_importance.png)

---

## 🧠 모델 구조

### 🟢 1단계: 정적 / 동적 이진 분류

- **모델 구조**: Dense → Dropout(0.2) → Dense  
- **Loss**: Binary Crossentropy  
- **Scaler**: MinMaxScaler  
- **Epoch**: 5~10 실험  
- **최고 정확도**: 96.87%

### 🔵 2단계: 세부 행동 분류 (Multiclass)

- 정적/동적 각각 별도 모델 (Softmax 활용)
- LabelEncoder 사용하여 숫자 라벨 매핑
- Dropout 적용 시 과적합 방지 효과 확인

![model-architecture](./assets/model_diagram.png)

---

## 🌀 전체 예측 흐름도

```python
def predict(test_df):
    x, y = preprocessing(test_df)
    is_dynamic = predict_active_or_static(x)
    if is_dynamic:
        return active_model.predict(x)
    else:
        return static_model.predict(x)
