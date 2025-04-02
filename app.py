import streamlit as st
import pandas as pd
import sklearn
import xgboost
import pickle

df = pd.read_csv('Cleaned_data.csv')

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("onehot_model.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)
categorical_columns = ['Gender', 'Highest Education Level', 'Preferred Subjects in Highschool/College', 
                        'Participation in Extracurricular Activities', 'Previous Work Experience (If Any)', 
                        'Preferred Work Environment', 'Leadership Experience', 'Networking & Social Skills', 
                        'Tech-Savviness', 'Motivation for Career Choice ']
st.title("Career Choice")

with st.form("Data_collection"):
    age = st.selectbox("How old are you?", list(range(18, 26)), index=0)
    
    gender = st.selectbox("Gender: ", list(df['Gender'].unique()), index=0)

    highest_education = st.selectbox("Highest Educational qualification: ", list(df['Highest Education Level'].unique()), index=0)

    cgpa = st.number_input("What is your CGPA in your highest qualification? ", min_value=0, max_value=10, value=0)

    stream = st.selectbox("What stream did you study in PU?", list(df['Preferred Subjects in Highschool/College'].unique()), index=0)

    activities = st.selectbox("Have you participated in any extra curricular activities?", list(df['Participation in Extracurricular Activities'].unique()), index = 0)

    work_experience = st.selectbox("Any previous Work experience?", list(df['Previous Work Experience (If Any)'].unique()), index=0)

    prefered_work_environment = st.selectbox("Prefered Work Environment", list(df['Preferred Work Environment'].unique()), index=0)

    risk = st.number_input("How likely are you willing to take risk, 10 being the highest and 0 being the lowest?", min_value=0, max_value=10, value=0)

    leadership = st.selectbox("Do you have any experience in leading? ", list(df['Leadership Experience'].unique()), index=0)

    networking = st.selectbox("Networking & Social Skills", list(df['Networking & Social Skills'].unique()), index=0)

    tech_savviness = st.selectbox("Tech-Savviness", list(df['Tech-Savviness'].unique()), index=0)

    financial_stability = st.number_input("How stable is your family financially?, 10 being the highest and 0 being the lowest?", min_value=0, max_value=10, value=0)

    motivation = st.selectbox("Motivation for Career Choice", list(df['Motivation for Career Choice '].unique()), index=0)

    siblings = st.number_input("How many siblings do you have?", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Submit")

if submitted:
    temp_dict = {
    'Age': [int(age)],
    'Gender': [gender],
    'Highest Education Level': [highest_education],
    'Preferred Subjects in Highschool/College': [stream],
    'Participation in Extracurricular Activities': [activities],
    'Previous Work Experience (If Any)': [work_experience],
    'Preferred Work Environment': [prefered_work_environment],
    'Risk-Taking Ability ': [int(risk)],
    'Leadership Experience': [leadership],
    'Networking & Social Skills': [networking],
    'Tech-Savviness': [tech_savviness],
    'Financial Stability - self/family (1 is low income and 10 is high income)': [int(financial_stability)],
    'Motivation for Career Choice ': [motivation],
    'Number of Siblings': [int(siblings)],
    'CGPA': [cgpa]
    }

    df_to_predict = pd.DataFrame(temp_dict)
    temp = onehot_encoder.transform(df_to_predict[categorical_columns])
    temp_df = pd.DataFrame(temp, columns = onehot_encoder.get_feature_names_out(categorical_columns))
    df_to_predict = pd.concat([df_to_predict.drop(columns=categorical_columns), temp_df], axis=1)
    
    predicted_value = model.predict(df_to_predict)
    
    data = encoder.inverse_transform(predicted_value)

    st.write("You would probably become a "+ data)