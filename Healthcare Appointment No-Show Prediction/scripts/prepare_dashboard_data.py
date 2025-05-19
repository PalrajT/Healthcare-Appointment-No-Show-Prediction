# Prepare Data for Power BI Dashboard

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def prepare_dashboard_data(input_file='../data/healthcare_appointments.csv', output_folder='../visualizations'):
    """
    Process the healthcare appointment dataset and prepare data for Power BI dashboard.
    
    Args:
        input_file (str): Path to the input CSV file
        output_folder (str): Path to save the output files
    """
    print(f"Preparing dashboard data from {input_file}...")
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Load data
        df = pd.read_csv(input_file)
        
        # Convert date columns to datetime
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
        
        # Feature engineering
        if 'LeadTime' not in df.columns:
            df['LeadTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        
        if 'AppointmentDayOfWeek' not in df.columns:
            df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.day_name()
        
        # Convert target variable to binary if needed
        if df['No-show'].dtype == object:
            df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
        
        # Create age groups
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
        
        # Create lead time groups
        df['LeadTimeGroup'] = pd.cut(df['LeadTime'], bins=[0, 7, 14, 30, 100], labels=['0-7 days', '8-14 days', '15-30 days', '30+ days'])
        
        # Generate summary tables for the dashboard
        
        # 1. No-show rates by age group
        age_no_show = pd.crosstab(df['AgeGroup'], df['No-show'], normalize='index') * 100
        age_no_show.columns = ['Show Rate', 'No-show Rate']
        age_no_show.reset_index().to_csv(f"{output_folder}/no_show_by_age.csv", index=False)
        
        # 2. No-show rates by gender
        gender_no_show = pd.crosstab(df['Gender'], df['No-show'], normalize='index') * 100
        gender_no_show.columns = ['Show Rate', 'No-show Rate']
        gender_no_show.reset_index().to_csv(f"{output_folder}/no_show_by_gender.csv", index=False)
        
        # 3. No-show rates by neighborhood
        neighborhood_no_show = pd.crosstab(df['Neighbourhood'], df['No-show'], normalize='index') * 100
        neighborhood_no_show.columns = ['Show Rate', 'No-show Rate']
        neighborhood_no_show.reset_index().to_csv(f"{output_folder}/no_show_by_neighborhood.csv", index=False)
        
        # 4. No-show rates by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_no_show = pd.crosstab(df['AppointmentDayOfWeek'], df['No-show'], normalize='index') * 100
        day_no_show = day_no_show.reindex(day_order)
        day_no_show.columns = ['Show Rate', 'No-show Rate']
        day_no_show.reset_index().to_csv(f"{output_folder}/no_show_by_day.csv", index=False)
        
        # 5. No-show rates by SMS received
        sms_no_show = pd.crosstab(df['SMS_received'], df['No-show'], normalize='index') * 100
        sms_no_show.columns = ['Show Rate', 'No-show Rate']
        sms_no_show.reset_index().to_csv(f"{output_folder}/no_show_by_sms.csv", index=False)
        
        # 6. No-show rates by lead time group
        lead_time_no_show = pd.crosstab(df['LeadTimeGroup'], df['No-show'], normalize='index') * 100
        lead_time_no_show.columns = ['Show Rate', 'No-show Rate']
        lead_time_no_show.reset_index().to_csv(f"{output_folder}/no_show_by_lead_time.csv", index=False)
        
        # 7. No-show rates by health conditions
        health_conditions = ['Hypertension', 'Diabetes', 'Alcoholism']
        for condition in health_conditions:
            condition_no_show = pd.crosstab(df[condition], df['No-show'], normalize='index') * 100
            condition_no_show.columns = ['Show Rate', 'No-show Rate']
            condition_no_show.reset_index().to_csv(f"{output_folder}/no_show_by_{condition.lower()}.csv", index=False)
        
        # 8. Export full processed dataset for Power BI
        df.to_csv(f"{output_folder}/processed_appointments.csv", index=False)
        
        # 9. Train model and export feature importance
        X = df.drop(['No-show', 'PatientID', 'ScheduledDay', 'AppointmentDay', 'AgeGroup', 'LeadTimeGroup'], axis=1, errors='ignore')
        y = df['No-show']
        
        # Handle categorical features
        categorical_features = [col for col in ['Gender', 'Neighbourhood', 'AppointmentDayOfWeek'] if col in X.columns]
        numerical_features = [col for col in X.columns if col not in categorical_features]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create and train model pipeline
        dt_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        dt_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = dt_pipeline.predict(X_test)
        
        # Evaluate model
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        # Export metrics
        pd.DataFrame([metrics]).to_csv(f"{output_folder}/model_metrics.csv", index=False)
        
        # Export confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_df = pd.DataFrame(conf_matrix, 
                              columns=['Predicted Show', 'Predicted No-show'],
                              index=['Actual Show', 'Actual No-show'])
        conf_df.to_csv(f"{output_folder}/confusion_matrix.csv")
        
        # Create a sample Power BI template file description
        with open(f"{output_folder}/power_bi_instructions.txt", 'w') as f:
            f.write("""# Healthcare Appointment No-Show Dashboard Instructions

## Data Files
The following CSV files have been prepared for your Power BI dashboard:

1. processed_appointments.csv - Full processed dataset
2. no_show_by_age.csv - No-show rates by age group
3. no_show_by_gender.csv - No-show rates by gender
4. no_show_by_neighborhood.csv - No-show rates by neighborhood
5. no_show_by_day.csv - No-show rates by day of week
6. no_show_by_sms.csv - No-show rates by SMS received status
7. no_show_by_lead_time.csv - No-show rates by lead time group
8. no_show_by_hypertension.csv - No-show rates by hypertension status
9. no_show_by_diabetes.csv - No-show rates by diabetes status
10. no_show_by_alcoholism.csv - No-show rates by alcoholism status
11. model_metrics.csv - Model performance metrics
12. confusion_matrix.csv - Confusion matrix for model evaluation

## Suggested Dashboard Pages

1. **Overview Page**
   - Key metrics: Overall no-show rate, total appointments
   - No-show trends by day of week (bar chart)
   - No-show rates by age group (column chart)
   - No-show rates by gender (donut chart)

2. **Demographic Analysis Page**
   - No-show rates by neighborhood (map visualization)
   - No-show rates by age and gender (matrix visualization)
   - Health conditions impact on no-show rates (multi-row card)

3. **Operational Insights Page**
   - Impact of SMS reminders (card visualization)
   - Lead time analysis (line chart)
   - Appointment scheduling recommendations (table)

4. **Prediction Model Page**
   - Model performance metrics (cards)
   - Confusion matrix (matrix visualization)
   - Feature importance (bar chart)
   - Prediction tool (what-if parameter)

## Implementation Steps
1. Import all CSV files into Power BI
2. Create relationships between tables as needed
3. Create calculated measures for key metrics
4. Design the dashboard pages following the structure above
5. Add slicers for interactive filtering by date, age group, gender, etc.
6. Create tooltips for additional insights
7. Publish and share the dashboard
""")
        
        print(f"Dashboard data preparation complete. Files saved to {output_folder}")
        print(f"Generated {len(os.listdir(output_folder))} files for Power BI dashboard")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found. Please run the generate_sample_data.py script first.")
    except Exception as e:
        print(f"Error preparing dashboard data: {str(e)}")

if __name__ == "__main__":
    prepare_dashboard_data()