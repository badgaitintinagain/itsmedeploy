import joblib
import pandas as pd
import streamlit as st
import os

# Load model and encoders
model_path = 'randforest.joblib'
encoders_path = 'label_encoders.joblib'

def load_object(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading file '{path}': {e}")
        st.stop()

model = load_object(model_path)
label_encoders = load_object(encoders_path)

# Access each encoder
try:
    le_wind_gust_dir = label_encoders['WindGustDir']
    le_wind_dir_9am = label_encoders['WindDir9am']
    le_wind_dir_3pm = label_encoders['WindDir3pm']
    le_rain_today = label_encoders['RainToday']
except KeyError as e:
    st.error(f"Missing encoder in label encoders: {e}")
    st.stop()

# App title and description
st.title('พยากรณ์สภาพอากาศ')
st.markdown("""
    ใช้ฟอร์มด้านล่างเพื่อป้อนข้อมูลสภาพอากาศและคาดการณ์ว่าพรุ่งนี้จะมีฝนหรือไม่
    โปรดระบุข้อมูลที่เป็นไปตามช่วงที่กำหนด
""")

# Create a form with styling
with st.form(key='weather_form'):
    st.header("ข้อมูลที่ใช้ในการพยากรณ์")

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        min_temp = st.slider('MinTemp (°C)', min_value=-20.0, max_value=60.0, value=0.0, step=0.1)
        max_temp = st.slider('MaxTemp (°C)', min_value=-20.0, max_value=60.0, value=0.0, step=0.1)
        rainfall = st.slider('Rainfall (mm)', min_value=0.0, max_value=400.0, value=0.0, step=0.1)
        wind_gust_speed = st.slider('WindGustSpeed (km/h)', min_value=0.0, max_value=150.0, value=0.0, step=0.1)
        wind_speed_9am = st.slider('WindSpeed9am (km/h)', min_value=0.0, max_value=150.0, value=0.0, step=0.1)
        wind_speed_3pm = st.slider('WindSpeed3pm (km/h)', min_value=0.0, max_value=150.0, value=0.0, step=0.1)
        humidity_9am = st.slider('Humidity9am (%)', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        humidity_3pm = st.slider('Humidity3pm (%)', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        temp_9am = st.slider('Temp9am (°C)', min_value=-20.0, max_value=60.0, value=0.0, step=0.1)
        temp_3pm = st.slider('Temp3pm (°C)', min_value=-20.0, max_value=60.0, value=0.0, step=0.1)

    with col2:
        wind_gust_dir = st.selectbox('WindGustDir',
                                     options=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'NNW', 'NNE', 'ESE', 'SSE',
                                              'SSW', 'WSW', 'WNW'])
        wind_dir_9am = st.selectbox('WindDir9am',
                                    options=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'NNW', 'NNE', 'ESE', 'SSE',
                                             'SSW', 'WSW', 'WNW'])
        wind_dir_3pm = st.selectbox('WindDir3pm',
                                    options=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'NNW', 'NNE', 'ESE', 'SSE',
                                             'SSW', 'WSW', 'WNW'])
        rain_today = st.selectbox('RainToday', options=['No', 'Yes'])

    # Submit button
    submit_button = st.form_submit_button(label='ทำการพยากรณ์')

# Prepare data for prediction
if submit_button:
    input_data = pd.DataFrame({
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'WindGustDir': [wind_gust_dir],
        'WindGustSpeed': [wind_gust_speed],
        'WindDir9am': [wind_dir_9am],
        'WindDir3pm': [wind_dir_3pm],
        'WindSpeed9am': [wind_speed_9am],
        'WindSpeed3pm': [wind_speed_3pm],
        'Humidity9am': [humidity_9am],
        'Humidity3pm': [humidity_3pm],
        'Temp9am': [temp_9am],
        'Temp3pm': [temp_3pm],
        'RainToday': [rain_today]
    })

    # Transform categorical features
    def safe_transform(encoder, column):
        try:
            return encoder.transform(input_data[column])
        except ValueError:
            st.warning(f"Warning: The value '{input_data[column].values[0]}' for column '{column}' was not seen during training.")
            # Fallback to a default value if unseen
            return encoder.transform([encoder.classes_[0]])

    input_data['WindGustDir'] = safe_transform(le_wind_gust_dir, 'WindGustDir')
    input_data['WindDir9am'] = safe_transform(le_wind_dir_9am, 'WindDir9am')
    input_data['WindDir3pm'] = safe_transform(le_wind_dir_3pm, 'WindDir3pm')
    input_data['RainToday'] = safe_transform(le_rain_today, 'RainToday')

    # Predict
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:  # Assuming 1 means 'Yes' and 0 means 'No'
            st.success("พรุ่งนี้จะมีฝน")
            image_path = 'picture/sad.jpg'
            if os.path.isfile(image_path):
                st.image(image_path, caption='Sad! ☔', use_column_width=True)
            else:
                st.warning("Image for 'Rain' not found.")
        else:
            st.success("พรุ่งนี้จะไม่มีฝน")
            image_path = 'picture/yay.jpg'
            if os.path.isfile(image_path):
                st.image(image_path, caption='Yay! 🌤️', use_column_width=True)
            else:
                st.warning("Image for 'No Rain' not found.")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
