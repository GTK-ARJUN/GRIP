import streamlit as st
import joblib

st.title('Score detection using the hours of study')

model=joblib.load('modjob.pkl')

x_input=round(st.number_input('Enter the study time in hours'),2)

y_predicted=model.predict([[x_input]]).ravel()
y_predicted=round(y_predicted[0],2)

if y_predicted>100:
    y_predicted=100

st.header('Results')
st.write(f'Student studying for {x_input} hours will get marks around {y_predicted} %')
