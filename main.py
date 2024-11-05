import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Custom CSS
with open('style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load the pre-trained model with caching
@st.cache_resource
def load_model(model_path):
    """Helper function to load the pre-trained model."""
    model = joblib.load(model_path)
    return model

logistic_model = load_model("models/logistic-regression-model.pkl")
decision_tree_model = load_model("models/decision-tree-model.pkl")


# Load dataset function
data = pd.read_csv('dataset/diabetes.csv')

# Define the split size for 20%
test_data = int(len(data) * 0.2)

# Create training and testing datasets
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']                # Target variable

# Take the first 20% of the data as the test set
X_test = X.iloc[:test_data]       # First 20% of rows
y_test = y.iloc[:test_data]       # Target variable

# Function to download the dataset
def download_data(file_path):
    """Download dataset as a CSV file."""
    with open(file_path, "rb") as f:
        st.download_button(
            label="Download Dataset",
            data=f,
            file_name=os.path.basename(file_path),
            mime="text/csv",
            help="Click here to download the full dataset"
        )

# Function to reset the input fields
def reset_inputs():
    # Reset all input fields to their default values
    st.session_state.pregnancies = 0.0
    st.session_state.glucose = 0.0
    st.session_state.bloodPressure = 0.0
    st.session_state.skinThickness = 0.0
    st.session_state.insulin = 0.0
    st.session_state.bmi = 0.0
    st.session_state.diabetesPedigreeFunction = 0.0
    st.session_state.age = 0

# Initialize session state variables if they don't exist
if 'pregnancies' not in st.session_state:
    st.session_state.pregnancies = 0.0
if 'glucose' not in st.session_state:
    st.session_state.glucose = 0.0
if 'bloodPressure' not in st.session_state:
    st.session_state.bloodPressure = 0.0
if 'skinThickness' not in st.session_state:
    st.session_state.skinThickness = 0.0
if 'insulin' not in st.session_state:
    st.session_state.insulin = 0.0
if 'bmi' not in st.session_state:
    st.session_state.bmi = 0.0
if 'diabetesPedigreeFunction' not in st.session_state:
    st.session_state.diabetesPedigreeFunction = 0.0
if 'age' not in st.session_state:
    st.session_state.age = 0

# Tabs for navigation
tabs = st.tabs(["HOME", "ABOUT ME", "DIAGNOSIS"])

# Home Page
def home():
    with tabs[0]:
        st.header("Diabetes Prediction App")
        st.subheader("Welcome to the Diabetes Prediction Tool")

        st.write("""
        This application uses machine learning algorithms to predict the likelihood of diabetes in individuals based on certain health parameters.

        #### Features:
        - **User-friendly Interface**: Easily input your health parameters for quick predictions.
        - **Instant Results**: Get immediate feedback on your diabetes risk.
        - **Downloadable Dataset**: Access the complete dataset for your reference.

        #### How it Works:
        1. Enter your health data in the provided fields.
        2. Click on the **Predict** button to see your results.
        3. Optionally, reset the form using the **Reset** button.

        #### Get Started:
        Navigate to the **Diagnosis** section to start using the tool.
        """)

        st.image("img/symptoms.jpg", use_column_width=True)  # Replace with an appropriate image path

# About Us Page
def about():
    with tabs[1]:
        st.subheader("About Me")
        st.write("##### Hello! I'm Divyansh Kumar Singh")
        st.write("I am a final-year engineering student at UIET, Panjab University, specializing in Computer Science and Engineering.")

        st.write("I am passionate about leveraging technology to solve real-world problems. My interests include machine learning, software development, and data science.")

        st.write("Feel free to connect with me on my social media:")

        # GitHub link
        st.markdown("[GitHub](https://github.com/singhdivyanshdishu)")  # Replace with your GitHub link

        # LinkedIn link
        st.markdown("[LinkedIn](https://www.linkedin.com/in/singh-divyansh-dishu-645311201/)")  # Replace with your LinkedIn link

# Diagnosis Page
def diagnosis_page():
    with tabs[2]:
        st.subheader("Diabetes Prediction App")

        if data is not None:
            st.write("Here are the first 5 entries of the dataset:")
            st.dataframe(data.head())  # Display the first 5 rows of the dataset

            # Provide a download button for the dataset
            download_data("dataset/diabetes.csv")


        # User inputs for prediction
        st.subheader("Enter patient details:")

        # Reset button
        if st.button('Reset Data', key='reset_button'):
            reset_inputs()  # Call the reset function

        # Create input fields linked to session state
        pregnancies = st.number_input('Pregnancies', key='pregnancies')
        glucose = st.number_input('Glucose', key='glucose')
        bloodPressure = st.number_input('Blood Pressure', key='bloodPressure')
        skinThickness = st.number_input('Skin Thickness', key='skinThickness')
        insulin = st.number_input('Insulin', key='insulin')
        bmi = st.number_input('BMI', key='bmi')
        diabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', key='diabetesPedigreeFunction')
        age = st.number_input('Age', key='age')

        # Dropdown menu for model selection
        model_choice = st.selectbox("Select a Model:",
                                    options=["Decision Tree","Logistic Regression"])  # In future I will add more

        # Choose the model based on user selection
        if model_choice == "Decision Tree":
            model = decision_tree_model
            model_name = "Decision Tree"  # Custom name
        elif model_choice == "Logistic Regression":
            model = logistic_model
            model_name = "Logistic Regression"  # Custom name

        # Model Predict Button
        if st.button('Predict', key='predict_button'):
            with st.spinner('Processing...'):
                # Create a DataFrame with the input values
                input_data = pd.DataFrame({
                    'Pregnancies': [pregnancies],
                    'Glucose': [glucose],
                    'BloodPressure': [bloodPressure],
                    'SkinThickness': [skinThickness],
                    'Insulin': [insulin],
                    'BMI': [bmi],
                    'DiabetesPedigreeFunction': [diabetesPedigreeFunction],
                    'Age': [age]
                })

                # Make the prediction using the DataFrame
                pred = model.predict(input_data)
                result = "Positive" if pred == [1] else "Negative"

                # Display result using st.success or st.error
                if result == "Positive":
                    st.error(f"The prediction is: **{result}**")
                else:
                    st.success(f"The prediction is: **{result}**")

                # Calculate accuracy
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Display accuracy
                st.info(f'{model_name} Model Accuracy: **{accuracy * 100:.2f}%**')


# Page Routing
home()
about()
diagnosis_page()
