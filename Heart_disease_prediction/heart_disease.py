import streamlit as s
import pandas as pd
import numpy as np
import pickle
import random

s.header('Heart Disease Prediction Using Machine Learning')

c = '''Heart Disease Prediction using Machine Learning Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:

**Logistic Regression**

**Naive Bayes**

**Support Vector Machine (Linear)**

**K-Nearest Neighbors**

**Decision Tree**

**Random Forest**

**XGBoost**

**Artificial Neural Network (1 Hidden Layer, Keras)**
'''


s.markdown(c)

s.image('https://media.clinicaladvisor.com/images/2017/03/29/heartillustrationts51811362_1191108.jpg')
with open('Heart_disease_prediction/model_joblib.pkl','rb') as f:
    chatgpt = pickle.load(f)


#laod data 
url = '''https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'''
df = pd.read_csv(url)
print('Done')

s.sidebar.header('Select features to predict Heart Disease')
s.sidebar.image('https://cdn.dribbble.com/users/2154580/screenshots/6452241/atemlos_loop_heart_v1.0_chriseff_dribbble.gif')


all_values = []
final_value = [all_values]


for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

    var =s.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)


ans = chatgpt.predict(final_value)[0]
all_values = []
import time

progress_bar = s.progress(0)
placeholder = s.empty()
placeholder.subheader('Predicting Heart Disease')

place = s.empty()
place.image('https://i.makeagif.com/media/1-17-2024/dw-jXM.gif',width = 200)

random.seed(12)
for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    placeholder.empty()
    place.empty()
    s.success(body)
    progress_bar = s.progress(0)
else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    s.warning(body)
    progress_bar = s.progress(0)
