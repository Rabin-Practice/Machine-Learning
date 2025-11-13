import pickle
import warnings
import streamlit as st
from streamlit_option_menu import option_menu

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Trying to unpickle.*')

# loading the saved models
with open('diabetes_model.sav', 'rb') as f:
    diabetes_model = pickle.load(f)
with open('heart_disease_model.sav', 'rb') as f:
    heart_disease_model = pickle.load(f)
with open('parkinsons_model.sav', 'rb') as f:
    parkinsons_model = pickle.load(f)

# Default values (dataset means for non-critical features)
DEFAULT_VALUES = {
    'diabetes': {
        'Pregnancies': 3.8,
        'SkinThickness': 20.5,
        'Insulin': 79.8,
        'DiabetesPedigreeFunction': 0.47
    },
    'heart': {
        'fbs': 0,  # Fasting blood sugar (0 or 1)
        'restecg': 1,  # Resting ECG (0, 1, or 2)
        'exang': 0,  # Exercise induced angina (0 or 1)
        'oldpeak': 1.0,  # ST depression
        'slope': 1,  # Slope (0, 1, or 2)
        'ca': 0,  # Major vessels (0-3)
        'thal': 2  # Thalassemia (0, 1, 2, or 3)
    },
    'parkinsons': {
        'MDVP:Fhi(Hz)': 197.1,
        'MDVP:Flo(Hz)': 116.3,
        'MDVP:Jitter(Abs)': 0.00006,
        'MDVP:RAP': 0.003,
        'MDVP:PPQ': 0.0035,
        'Jitter:DDP': 0.009,
        'MDVP:Shimmer': 0.03,
        'MDVP:Shimmer(dB)': 0.31,
        'Shimmer:APQ3': 0.016,
        'Shimmer:APQ5': 0.02,
        'MDVP:APQ': 0.022,
        'Shimmer:DDA': 0.048,
        'NHR': 0.025,
        'DFA': 0.82,
        'spread1': -5.7,
        'spread2': 0.23,
        'D2': 2.38
    }
}

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
st.title("Multiple Disease Prediction System")
st.markdown("---")
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    st.subheader('Diabetes Prediction')
    st.markdown("Enter the following essential information:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=500, value=120, step=1)
        BMI = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    
    with col2:
        BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=80, step=1)
        Age = st.number_input('Age', min_value=0, max_value=120, value=30, step=1)
    
    # code for Prediction
    diab_diagnosis = ''
    
    if st.button('Get Diabetes Test Result', type='primary', use_container_width=True):
        try:
            # Use defaults for non-essential features
            Pregnancies = DEFAULT_VALUES['diabetes']['Pregnancies']
            SkinThickness = DEFAULT_VALUES['diabetes']['SkinThickness']
            Insulin = DEFAULT_VALUES['diabetes']['Insulin']
            DiabetesPedigreeFunction = DEFAULT_VALUES['diabetes']['DiabetesPedigreeFunction']
            
            # Make prediction with all features
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, 
                                                       SkinThickness, Insulin, BMI, 
                                                       DiabetesPedigreeFunction, Age]])
            
            if (diab_prediction[0] == 1):
                diab_diagnosis = '⚠️ The person is diabetic'
            else:
                diab_diagnosis = '✅ The person is not diabetic'
        except Exception as e:
            diab_diagnosis = f'Error: {str(e)}'
        
    if diab_diagnosis:
        st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    st.subheader('Heart Disease Prediction')
    st.markdown("Enter the following essential information:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=50, step=1)
        sex = st.selectbox('Sex', options=[('Male', 1), ('Female', 0)], format_func=lambda x: x[0])
        sex_value = sex[1]
        cp = st.selectbox('Chest Pain Type', 
                         options=[('Typical Angina', 0), ('Atypical Angina', 1), 
                                 ('Non-anginal Pain', 2), ('Asymptomatic', 3)],
                         format_func=lambda x: x[0])
        cp_value = cp[1]
    
    with col2:
        trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=0, max_value=250, value=130, step=1)
        chol = st.number_input('Cholesterol Level (mg/dL)', min_value=0, max_value=600, value=240, step=1)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=150, step=1)
    
    # code for Prediction
    heart_diagnosis = ''
    
    if st.button('Get Heart Disease Test Result', type='primary', use_container_width=True):
        try:
            # Use defaults for non-essential features
            fbs = DEFAULT_VALUES['heart']['fbs']
            restecg = DEFAULT_VALUES['heart']['restecg']
            exang = DEFAULT_VALUES['heart']['exang']
            oldpeak = DEFAULT_VALUES['heart']['oldpeak']
            slope = DEFAULT_VALUES['heart']['slope']
            ca = DEFAULT_VALUES['heart']['ca']
            thal = DEFAULT_VALUES['heart']['thal']
            
            # Make prediction with all features
            heart_prediction = heart_disease_model.predict([[age, sex_value, cp_value, trestbps, 
                                                             chol, fbs, restecg, thalach,
                                                             exang, oldpeak, slope, ca, thal]])                          
            
            if (heart_prediction[0] == 1):
                heart_diagnosis = '⚠️ The person is having heart disease'
            else:
                heart_diagnosis = '✅ The person does not have any heart disease'
        except Exception as e:
            heart_diagnosis = f'Error: {str(e)}'
        
    if heart_diagnosis:
        st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    st.subheader("Parkinson's Disease Prediction")
    st.markdown("Enter the following essential voice analysis parameters:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz) - Average vocal fundamental frequency', 
                            min_value=80.0, max_value=300.0, value=150.0, step=0.1)
        Jitter_percent = st.number_input('MDVP:Jitter(%) - Variation in fundamental frequency', 
                                        min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.4f")
        HNR = st.number_input('HNR - Harmonics to Noise Ratio', 
                             min_value=0.0, max_value=50.0, value=20.0, step=0.1)
    
    with col2:
        RPDE = st.number_input('RPDE - Nonlinear dynamical complexity measure', 
                              min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        DFA = st.number_input('DFA - Signal fractal scaling exponent', 
                             min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        PPE = st.number_input('PPE - Pitch period entropy', 
                             min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    if st.button("Get Parkinson's Test Result", type='primary', use_container_width=True):
        try:
            # Use defaults for non-essential features
            defaults = DEFAULT_VALUES['parkinsons']
            
            # Make prediction with all features
            parkinsons_prediction = parkinsons_model.predict([[fo, defaults['MDVP:Fhi(Hz)'], defaults['MDVP:Flo(Hz)'], 
                                                               Jitter_percent, defaults['MDVP:Jitter(Abs)'], 
                                                               defaults['MDVP:RAP'], defaults['MDVP:PPQ'], 
                                                               defaults['Jitter:DDP'], defaults['MDVP:Shimmer'], 
                                                               defaults['MDVP:Shimmer(dB)'], defaults['Shimmer:APQ3'], 
                                                               defaults['Shimmer:APQ5'], defaults['MDVP:APQ'], 
                                                               defaults['Shimmer:DDA'], defaults['NHR'], HNR, RPDE,
                                                               DFA, defaults['spread1'], defaults['spread2'], 
                                                               defaults['D2'], PPE]])                          
            
            if (parkinsons_prediction[0] == 1):
                parkinsons_diagnosis = "⚠️ The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "✅ The person does not have Parkinson's disease"
        except Exception as e:
            parkinsons_diagnosis = f'Error: {str(e)}'
        
    if parkinsons_diagnosis:
        st.success(parkinsons_diagnosis)

def set_bg_from_url(url, opacity=1):
    
    footer = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
    <footer>
        <div style='visibility: visible;margin-top:7rem;justify-content:center;display:flex;'>
            <p style="font-size:1.1rem;">
                &nbsp;
                <a href="https://www.linkedin.com/in/mohamedshaad">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                    </svg>          
                </a>
                &nbsp;
                <a href="https://github.com/shaadclt">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
            </p>
        </div>
    </footer>
"""
    st.markdown(footer, unsafe_allow_html=True)
    
    
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image from URL
set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768", opacity=0.875)
