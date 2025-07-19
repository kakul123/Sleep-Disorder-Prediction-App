#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[2]:


df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')



# In[3]:


df.isnull().sum()


# In[4]:


## treat the null vaues
df['Sleep Disorder'].fillna(df['Sleep Disorder'].mode()[0], inplace=True)
df['Sleep Disorder'].value_counts()


# In[5]:


## encoded the rows
le_gender = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])

df = pd.get_dummies(df, columns=['BMI Category'], drop_first=True, dtype=int)

## encode the target column
le_sleep_disorder = LabelEncoder()

df['Sleep Disorder'] = le_sleep_disorder.fit_transform(df['Sleep Disorder'])



# In[6]:


## drop the column personID, Occupation
df = df.drop(columns=['Person ID', 'Occupation'])  # optional: drop ID too


# In[7]:





# In[8]:


## blood pressure divided into two parts systolic bp and distolic bp 
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
## drop the bp column
df = df.drop(columns=['Blood Pressure'])


# In[9]:


## add the pulse pressure column
df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']


# In[10]:


df['High_BP'] = ((df['Systolic_BP'] >= 130) | (df['Diastolic_BP'] >= 85)).astype(int)
df['Low_BP'] = ((df['Systolic_BP'] < 90) | (df['Diastolic_BP'] < 60)).astype(int)



# ## Perform the eda

# In[11]:


## Step 1 Basic Overview

print(df.shape)
print(df.info())
print(df.describe())
print(df['Sleep Disorder'].value_counts())


# In[53]:


## Step 2: Check Missing Values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# In[12]:


## Step 3: Feature Correlation with Sleep Disorder
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with Sleep Disorder")
plt.show()


# In[13]:


## Step 4: Distribution Plots
numerical_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                  'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure']

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[14]:


## Step 5: Count Plots for Categorical Features

categorical_cols = ['Gender', 'BMI Category_Normal Weight', 'BMI Category_Obese', 
                    'BMI Category_Overweight', 'High_BP', 'Low_BP', 'Sleep Disorder']

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f'Count of {col}')
    plt.show()


# In[15]:


##  Step 6: Boxplots to Compare with Target
features_to_compare = ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Pulse_Pressure']

for col in features_to_compare:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Sleep Disorder', y=col, data=df)
    plt.title(f'{col} vs Sleep Disorder')
    plt.show()


# In[16]:


## Step 7: Pairplot
# Only include relevant features
eda_cols = ['Age', 'Sleep Duration', 'Stress Level', 'Heart Rate', 'Sleep Disorder']
sns.pairplot(df[eda_cols], hue='Sleep Disorder', palette='husl')
plt.show()


# In[17]:


## Step 8: Group Statistics
print(df.groupby('Sleep Disorder')[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Pulse_Pressure']].mean())


# ## Training and testing the data
# 

# In[22]:


#Split the data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)


# In[23]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression 

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train_scaled, y_train)

#DecisionTree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)


# In[24]:


models = {'Logistic Regression' : lr, 'Decision Tree' : dt, 'Random Forest' : rf}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

for name, model in models.items():
    y_pred = model.predict(x_test)
    print(f"{name}")
    print("Accuracy: ", accuracy_score(y_test,y_pred))
    print("Precision: ", precision_score(y_test,y_pred,average='weighted'))
    print("Recall Score: ", recall_score(y_test,y_pred,average='weighted'))
    print("F1 Score: ", f1_score(y_test,y_pred,average='weighted'))


# In[26]:


from sklearn.metrics import confusion_matrix
for name, model in models.items():
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# In[33]:


## Save model
import joblib
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl') 


# ## Streamlit

# In[34]:


import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛌 Sleep Disorder Prediction App")

st.markdown("### Please enter your health and lifestyle details:")

# --- Input Fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100, 25)
sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 5)
physical_activity = st.slider("Physical Activity Level (1-10)", 1, 10, 5)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 70)
daily_steps = st.number_input("Daily Steps", 0, 30000, 7000)
bmi_category = st.selectbox("BMI Category", ["Normal Weight", "Overweight", "Obese"])
systolic = st.number_input("Systolic Blood Pressure", 80, 200, 120)
diastolic = st.number_input("Diastolic Blood Pressure", 50, 130, 80)

# --- Feature Engineering ---
pulse_pressure = systolic - diastolic
high_bp = 1 if diastolic > 85 else 0
low_bp = 1 if systolic < 90 or diastolic < 60 else 0

# --- Manual Encoding ---
gender = 1 if gender == "Male" else 0
bmi_normal = 1 if bmi_category == "Normal Weight" else 0
bmi_overweight = 1 if bmi_category == "Overweight" else 0
bmi_obese = 1 if bmi_category == "Obese" else 0

# --- Final Input Vector ---
input_data = np.array([[gender, age, sleep_duration, quality_of_sleep,
                        physical_activity, stress_level, heart_rate, daily_steps,
                        bmi_normal, bmi_overweight, bmi_obese,
                        systolic, diastolic, pulse_pressure,
                        high_bp, low_bp]])

# --- Scale the Input ---
input_scaled = scaler.transform(input_data)


# ## Button

# In[35]:


if st.button("Predict Sleep Disorder"):
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]  # Get class probabilities

    # Display prediction
    if prediction == 0:
        st.error("⚠️ Likely Insomnia")
    elif prediction == 1:
        st.warning("😵‍💫 Likely Sleep Apnea")
    else:
        st.success("✅ No Sleep Disorder Detected")

    # Show confidence scores
    st.subheader("🧠 Model Confidence")
    classes = model.classes_
    for cls, prob in zip(classes, probabilities):
        label = "Insomnia" if cls == 0 else "Sleep Apnea" if cls == 1 else "No Disorder"
        st.write(f"{label}: {prob*100:.2f}%")

    # Health tips
    st.subheader("💡 Personalized Health Tips")
    if prediction == 0:
        st.markdown("""
        - Try maintaining a consistent sleep schedule.
        - Avoid screens (mobile/laptop) 1 hour before bedtime.
        - Reduce caffeine and heavy meals in the evening.
        - Try meditation or relaxation techniques.
        """)
    elif prediction == 1:
        st.markdown("""
        - Maintain a healthy weight to reduce apnea risk.
        - Sleep on your side instead of your back.
        - Avoid alcohol and smoking.
        - Consult a doctor for a sleep study.
        """)
    else:
        st.markdown("""
        - Keep up the good habits!
        - Continue moderate physical activity.
        - Sleep at least 7–8 hours per night.
        - Manage stress with exercise, journaling or mindfulness.
        """)

    # Optional: Plot bar chart of probabilities
    st.subheader("📊 Prediction Probability Chart")
    import matplotlib.pyplot as plt

    # Get predicted probabilities and class names
probabilities = model.predict_proba(input_scaled)[0]
classes = model.classes_

# Convert numeric class names to readable labels
label_map = {0: "Insomnia", 1: "Sleep Apnea", 2: "No Disorder"}
labels = [label_map[c] for c in classes]

# Dynamically assign color based on number of classes
color_map = {
    "Insomnia": "red",
    "Sleep Apnea": "orange",
    "No Disorder": "green"
}
colors = [color_map[label] for label in labels]

# Bar chart
fig, ax = plt.subplots()
ax.bar(labels, probabilities * 100, color=colors)
ax.set_ylabel('Confidence (%)')
ax.set_title('Prediction Confidence')
st.pyplot(fig)


# In[36]:





# In[ ]:




