# Analysis and Prediction of Student Academic Performance

A data science mini project (VSEC - Data Science Laboratory, BIT23VS01) at
Pimpri-Chinchwad College of Engineering, Department of Information Technology.

## Team
- Mrunmayee Ajay Kharche (125B1F229)
- Dipesh Kishor Shelar (125B1F208)
- Kunal Yeole (125B1F224)

Guide: Mrs. Jyoti Manoorkar, Course Coordinator

## About the project
This project studies how study habits, lifestyle, family and demographic factors
are associated with students' final grade (G3), using statistics and machine
learning. Two scenarios are compared: Model A (with G1/G2 grades) and
Model B (without them) to show data leakage.

## Dataset
UCI Student Performance dataset (Mathematics, 395 students):
https://archive.ics.uci.edu/dataset/320/student+performance
Source study: P. Cortez and A. Silva (2008).

## Folder structure
- data/raw - original dataset
- data/processed - cleaned and encoded data
- notebooks - 12 Jupyter notebooks (cleaning, EDA, statistics, models)
- outputs - saved figures and tables
- dashboard - Streamlit app (app.py) and saved models
- report, ppt - project report and presentation

## Methods
Descriptive statistics, EDA, correlation, hypothesis tests (t-tests, ANOVA),
Linear Regression, Random Forest, Logistic Regression, Decision Tree.

## Key results
- Average G3 = 10.42, pass rate = 67.1%
- Random Forest R2: 0.815 (with G1/G2) vs 0.245 (without)
- Logistic Regression accuracy: 88.6% vs 65.8%

## Run locally
pip install -r requirements.txt
cd dashboard
streamlit run app.py

## Live dashboard
<paste your Streamlit link here after deployment>