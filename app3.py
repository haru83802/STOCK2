import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 주식 데이터 가져오기
def get_stock_data(ticker, start="2020-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end)
    return data

# 데이터 준비 함수
def prepare_data(data, time_step=60):
    data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X).reshape(-1, time_step, 1)
    y = np.array(y)
    return X, y, scaler

# LSTM 모델 생성
def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit UI
st.title("📈 주식 가격 예측 웹앱")
ticker = st.text_input("종목 코드 입력 (예: AAPL, TSLA, 005930.KS)", "AAPL")

if st.button("예측 시작"):
    st.write(f"📊 {ticker} 주식 데이터 로드 중...")
    
    try:
        data = get_stock_data(ticker)
        st.line_chart(data['Close'])
        
        X, y, scaler = prepare_data(data)
        model = create_model((X.shape[1], 1))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        plt.figure(figsize=(12,6))
        plt.plot(data.index[-len(predictions):], predictions, label="예측 가격")
        plt.plot(data.index[-len(y):], scaler.inverse_transform(y.reshape(-1, 1)), label="실제 가격")
        plt.legend()
        st.pyplot(plt)
    
    except Exception as e:
        st.error(f"오류 발생: {e}")

