# 📱 스마트폰 센서 기반 모션 분류 프로젝트

> 스마트폰 센서 데이터를 활용하여 정적/동적 행동을 분류하고, 각각의 세부 행동을 예측하는 딥러닝 기반 모델 개발

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

### 🔍 주요 분석 결과

- 정적/동적 행동을 기준으로 Feature 중요도 분석  
- 전체적으로 **가속도(acc)** 기반 feature의 영향력이 높음  
- 주파수 도메인에서 추출한 feature가 핵심 지표

---

## 🧠 모델 구조

### 🟢 1단계: 정적 / 동적 이진 분류

- 모델 구조: Dense → Dropout(0.2) → Dense  
- Loss 함수: Binary Crossentropy  
- 데이터 정규화: MinMaxScaler  
- 데이터 분할: Train/Validation = 8:2  
- 최고 정확도: 96.87%

### 🔵 2단계: 세부 행동 분류 (Multiclass)

- 정적/동적 각각 별도 모델 (Softmax 기반)
- LabelEncoder로 라벨 숫자화
- Dropout을 활용하여 과적합 방지

---

## 🌀 예측 흐름 요약 (Pseudo Code)

```python
def predict(test_df):
    x, y = preprocessing(test_df)
    is_dynamic = predict_active_or_static(x)
    if is_dynamic:
        return active_model.predict(x)
    else:
        return static_model.predict(x)

1. 입력 데이터 전처리 및 정규화  
2. 정적/동적 분류 → 세부 행동 분류  
3. 최종 예측 결과 반환  

---

## 🏆 결과 요약

| 실험 모델                      | 테스트 정확도 |
|-------------------------------|----------------|
| Dropout 0.2, Epoch 5          | 90.41%         |
| Hidden Layer 5, Epoch 5       | 92.52%         |
| Hidden Layer 3, Epoch 10      | **96.87%**     |
| Dropout 0.5, Epoch 10         | 94.29%         |

📌 **Epoch 수 조정이 성능 향상에 가장 결정적인 요소로 확인됨**

---

## 💡 활용 방안

- 🏃 피트니스 앱 → 실시간 운동 인식 및 피드백  
- 🩺 노약자 낙상 감지 및 웨어러블 연동  
- 😴 수면 자세 분석  
- 🚗 운전자 졸음 감지, 보행자 안전 개선  

---

## ✅ To Do

- 테스트 데이터 기반 실시간 예측 시각화  
- Flask or Streamlit으로 데모 앱 배포  
- 웨어러블 디바이스 연동 모듈 추가  
