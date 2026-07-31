import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="<---repo id---->/tourism-package-prediction", filename="model.pkl")
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Prediction")
st.write("""
This application predicts whether a customer is likely to purchase
the newly introduced **Wellness Tourism Package** based on customer
demographics and previous interactions with the company.
""")

# User input
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox(
    "City Tier",
    [1,2,3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried","Small Business","Large Business","Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [3,4,5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single","Married","Divorced"]
)

numberoftrips = st.number_input(
    "Number of Trips per Year",
    min_value=0,
    max_value=20,
    value=2
)

passport = st.selectbox(
    "Passport Available",
    [0,1]
)

owncar = st.selectbox(
    "Own Car",
    [0,1]
)

numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=5,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=500000,
    value=30000
)

pitchsatisfactionscore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

productpitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

numberoffollowups = st.number_input(
    "Number of Follow Ups",
    min_value=0,
    max_value=10,
    value=2
)

durationofpitch = st.number_input(
    "Duration of Pitch (Minutes)",
    min_value=5,
    max_value=60,
    value=20
)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "ProductPitched": productpitched,
    "NumberOfFollowups": numberoffollowups,
    "DurationOfPitch": durationofpitch
}])

# Predict button
if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ This customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.error("❌ This customer is unlikely to purchase the Wellness Tourism Package.")
