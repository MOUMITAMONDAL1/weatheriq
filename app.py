import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import pytz

# --- Stylish Background & Font ---
st.set_page_config(page_title="Weather Forecast", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        font-size: 16px;
    }

    .block-container {
        backdrop-filter: blur(8px);
        background-color: rgba(255, 255, 255, 0.65);
        padding: 2rem;
        border-radius: 15px;
    }

    .stMetric {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        padding: 10px;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Title ---
st.title("🌦️ Weather Forecast Web App")

# --- API Configuration ---
API_KEY = 'bdf53fb14cacdd1cc84c85ba4ccbe98c'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# --- Fetch Current Weather Data ---
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'Wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed']
    }

# --- Read Historical Weather Data ---
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    return df

# --- Prepare Data for Classification ---
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrowk'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le

# --- Train Rain Prediction Model ---
def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("✅ Accuracy for Rain Model:", accuracy_score(y_test, y_pred))
    return model

# --- Prepare Data for Regression ---
def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

# --- Train Regression Model ---
def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- Predict Future Values ---
def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

# --- Main App Logic ---
def main():
    city = st.text_input('🏙️ Enter a City Name:', 'Kolkata')

    if st.button('Get Weather Forecast'):
        current_weather = get_current_weather(city)

        historical_data = read_historical_data('weather.csv')

        # Train Rain Prediction Model
        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)

        # Convert wind degrees to direction
        wind_deg = current_weather['Wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75), ("N", 348.75, 360)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['Wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp'],
        }

        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]

        # Future prediction
        X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        # Time setup
        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # --- Display Results ---
        st.subheader(f"🌍 Weather in {current_weather['city']}, {current_weather['country']}")
        st.metric("🌡️ Current Temp", f"{current_weather['current_temp']}°C")
        st.metric("🤔 Feels Like", f"{current_weather['feels_like']}°C")
        st.metric("💧 Humidity", f"{current_weather['humidity']}%")
        st.write("📋 Weather Condition:", current_weather['description'].capitalize())
        st.success(f"☔ Rain Tomorrow: {'Yes' if rain_prediction else 'No'}")

        # --- Stylish Future Temperature Display ---
        st.subheader("📈 Future Temperature Forecast")
        temp_cols = st.columns(5)
        for i, col in enumerate(temp_cols):
            col.metric(label=f"{future_times[i]}", value=f"{round(future_temp[i], 1)}°C")

        # --- Stylish Future Humidity Display ---
        st.subheader("💧 Future Humidity Forecast")
        humidity_cols = st.columns(5)
        for i, col in enumerate(humidity_cols):
            col.metric(label=f"{future_times[i]}", value=f"{round(future_humidity[i], 1)}%")

# --- Run App ---
if __name__ == "__main__":
    main()


