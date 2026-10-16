#Part A: Imports and Page Setup
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
BASE = os.path.dirname(os.path.abspath(__file__))
# Page configuration — must be the very first Streamlit command
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


#Part B: Load Data and Models (once, at the top)
# Load the processed dataset
df = pd.read_csv(os.path.join(BASE, "data", "student_features.csv"))
# Load the two trained models saved from your notebooks
rf_model = pickle.load(open(os.path.join(BASE, "model_rf.pkl"), "rb"))
clf_model = pickle.load(open(os.path.join(BASE, "model_classifier.pkl"), "rb"))

# Folders holding the saved tables and figures (one level above the dashboard folder)
TABLES = os.path.join(BASE, "..", "outputs", "tables")
FIGURES = os.path.join(BASE, "..", "outputs", "figures")

#Part C: Sidebar Navigation
st.sidebar.title("🎓 Student Performance Dashboard")
section = st.sidebar.radio("Menu", [
    "Overview",
    "Study Habits",
    "Lifestyle & Social",
    "Family & Demographics",
    "Statistical Analysis",
    "Model Performance",
    "Predict My Grade"
])


#Part D: Overview Page
if section == "Overview":
    st.title("📊 Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average G3", f"{df['G3'].mean():.2f}")
    c2.metric("Pass Rate", f"{(df['pass_fail'].mean()*100):.1f}%")
    c3.metric("Total Students", len(df))

    fig = px.histogram(df, x="G3", nbins=15, title="Final Grade Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.pie(df, names="performance_category", title="Performance Category Breakdown")
    st.plotly_chart(fig2, use_container_width=True)


#Part E: Study Habits, Lifestyle, Family Pages
elif section == "Study Habits":
    st.title("📚 Study Habits")

    fig = px.box(df, x="studytime_label", y="G3", title="Study Time vs Final Grade")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df, x="absences", y="G3", title="Absences vs Final Grade", trendline="ols")
    st.plotly_chart(fig2, use_container_width=True)

elif section == "Lifestyle & Social":
    st.title("🎉 Lifestyle & Social Factors")

    fig = px.box(df, x="goout", y="G3", title="Going Out vs Final Grade")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, x="Walc", y="G3", title="Weekend Alcohol Consumption vs Final Grade")
    st.plotly_chart(fig2, use_container_width=True)

elif section == "Family & Demographics":
    st.title("👨‍👩‍👧 Family & Demographic Factors")

    fig = px.box(df, x="Medu", y="G3", title="Mother's Education vs Final Grade")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, x="sex", y="G3", title="Gender vs Final Grade")
    st.plotly_chart(fig2, use_container_width=True)


#Part F: Statistical Analysis Page
elif section == "Statistical Analysis":
    st.title("📈 Statistical Analysis")

    st.subheader("Correlation with Final Grade (G3)")
    corr_table = pd.read_csv(os.path.join(TABLES, "correlation_matrix.csv"), index_col=0)
    st.dataframe(corr_table["G3"].sort_values(ascending=False))

    st.subheader("Model Comparison (Regression)")
    reg_comparison = pd.read_csv(os.path.join(TABLES, "model_comparison_leakage.csv"))
    st.dataframe(reg_comparison)

    st.subheader("Model Comparison (Classification)")
    clf_comparison = pd.read_csv(os.path.join(TABLES, "classification_comparison.csv"))
    st.dataframe(clf_comparison)


#Part G: Model Performance Page
elif section == "Model Performance":
    st.title("🤖 Model Performance")

    st.subheader("Actual vs Predicted (Random Forest Regressor)")
    st.image(os.path.join(FIGURES, "rf_actual_vs_predicted.png"))

    st.subheader("Feature Importance — Regression")
    st.image(os.path.join(FIGURES, "16_feature_importance_regression.png"))

    st.subheader("Feature Importance — Classification")
    st.image(os.path.join(FIGURES, "17_feature_importance_classification.png"))

    st.subheader("Confusion Matrix — Logistic Regression")
    st.image(os.path.join(FIGURES, "confusion_matrix_logreg_a.png"))


#Part H: Predict My Grade Page
elif section == "Predict My Grade":
    st.title("🔮 Predict My Grade")
    st.write("Adjust the sliders below to see a predicted final grade and pass/fail outcome.")

    # Collect user inputs matching the training features
    studytime = st.slider("Study Time (1=low, 4=high)", 1, 4, 2)
    failures = st.slider("Past Class Failures", 0, 3, 0)
    absences = st.slider("Number of Absences", 0, 75, 5)
    goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
    age = st.slider("Age", 15, 22, 17)
    traveltime = st.slider("Travel Time to School (1-4)", 1, 4, 1)
    Medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
    Fedu = st.slider("Father's Education (0-4)", 0, 4, 2)

    # Build a single-row DataFrame matching the EXACT columns the model was trained on
    input_dict = {col: 0 for col in rf_model.feature_names_in_}  # start all at 0
    input_dict['studytime'] = studytime
    input_dict['failures'] = failures
    input_dict['absences'] = absences
    input_dict['goout'] = goout
    input_dict['age'] = age
    input_dict['traveltime'] = traveltime
    input_dict['Medu'] = Medu
    input_dict['Fedu'] = Fedu

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        predicted_g3 = rf_model.predict(input_df)[0]
        predicted_pass = clf_model.predict(input_df)[0]

        st.success(f"Predicted Final Grade (G3): {predicted_g3:.1f} / 20")
        if predicted_pass == 1:
            st.info("Predicted Outcome: ✅ Pass")
        else:
            st.warning("Predicted Outcome: ❌ Fail")