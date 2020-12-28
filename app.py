import streamlit as st
import joblib
import sklearn
import webbrowser


option=['Supervised', 'Unsupervised', 'Decision tree Classifier']
x=st.sidebar.selectbox('Select machine learning type\model', options=option)

def supervised():
    st.header(option[0])
    st.subheader('To predict the percentage of an student based on the no. of study hours.')
    if st.sidebar.button('code'):
        webbrowser.open_new_tab('https://bit.ly/37P3RJ0')
    model = joblib.load('modjob.pkl')
    x_input = round(st.number_input('Enter the study time in hours'), 2)
    y_predicted = model.predict([[x_input]]).ravel()
    y_predicted = round(y_predicted[0], 2)
    if y_predicted > 100:
        y_predicted = 100
    elif y_predicted < 0:
        y_predicted = 0
    st.header('Prediction Result')
    st.write(f'Student studying for {x_input} hours will get marks around {y_predicted} %')

def unsupervised():
    st.header(option[1])
    st.subheader('Predict the optimum number of clusters and to predict the group for given sample')
    st.subheader("Enter the details below")
    st.sidebar.image('iris.png',width=280,
                    caption='Iris-flower')
    if st.sidebar.button('code'):
        webbrowser.open_new_tab('https://bit.ly/3gsrXxl')
    mms = joblib.load('mms_iris_scale.pkl')
    model = joblib.load('iris_pred.pkl')
    sl = st.number_input('Sepal Length (Range 4.3 - 7.9)')
    sw = st.number_input('Sepal Width (Range 2.0 - 4.4)')
    pl = st.number_input('Petal Length (Range 1.0 - 6.9)')
    pw = st.number_input('Petal Width (Range 0.1 - 2.5)')
    scaled_value = mms.transform([[sl, sw, pl, pw]])
    cluster_class = model.predict(scaled_value)[0]
    st.write(cluster_class)

def Decision_tree_Classifier():
    st.header(option[2])
    st.subheader('To classify given data of iris using Decisiontree')
    st.subheader("Enter the details below")
    st.sidebar.image('iris.png',width=280,
        caption='Iris-flower')
    if st.sidebar.button('code'):
        webbrowser.open_new_tab('https://bit.ly/3mWu4fa')


    target_enc=joblib.load('iris_target_enc.pkl')
    model=joblib.load('iris_dtree_clf.pkl')
    sl = st.number_input('Sepal Length (Range 4.3 - 7.9)')
    sw = st.number_input('Sepal Width (Range 2.0 - 4.4)')
    pl = st.number_input('Petal Length (Range 1.0 - 6.9)')
    pw = st.number_input('Petal Width (Range 0.1 - 2.5)')
    y=model.predict([[sl, sw, pl, pw]])[0]
    st.write(target_enc.classes_[y])


if x==option[0]:
    supervised()
elif x==option[1]:
    unsupervised()
elif x==option[2]:
    Decision_tree_Classifier()


