# Healthcare Appointment No-Show Prediction

## Introduction
Missed healthcare appointments lead to revenue loss and decreased efficiency. This project aims to predict no-shows to help clinics optimize schedules and improve patient outreach.

## Project Overview
We build a machine learning model to predict whether a patient will attend their appointment using demographic, health, and behavioral features. The model highlights key factors like SMS reminders, age, and day of the week.

## Tools Used
- Python (Pandas, Scikit-learn, Seaborn)
- Power BI

## Project Structure
- `data/`: Contains the dataset used for analysis
- `notebooks/`: Jupyter notebooks for data analysis and modeling
- `scripts/`: Python scripts for data processing and model building
- `visualizations/`: Output visualizations and Power BI dashboard

## Steps Involved
1. **Data Cleaning & EDA**: Handled missing values, encoded categorical variables, and visualized trends.
2. **Feature Engineering**: Created features like lead time, appointment weekday, and SMS_received flag.
3. **Modeling**: Trained a Decision Tree classifier and evaluated using accuracy, precision, recall, and confusion matrix.
4. **Dashboarding**: Built a Power BI dashboard showing no-show patterns by age group, gender, and neighborhood.
5. **Optimization Suggestions**: Recommended overbooking slots on high-risk days and automating SMS follow-ups.

## Results
The model achieved ~78% accuracy and revealed that SMS reminders and shorter wait times reduce no-shows. Clinics can use the dashboard for targeted intervention strategies.