# Healthcare Appointment No-Show Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the healthcare appointment dataset.
    
    Args:
        file_path (str): Path to the dataset CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("Loading and preprocessing data...")
    
    # For demonstration purposes, we'll create a sample dataset if file doesn't exist
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample dataset for demonstration.")
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data
        ages = np.random.randint(0, 100, n_samples)
        genders = np.random.choice(['M', 'F'], n_samples)
        scheduled_dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
        appointment_dates = scheduled_dates + pd.to_timedelta(np.random.randint(1, 30, n_samples), unit='D')
        sms_received = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        neighborhoods = np.random.choice(['Downtown', 'Uptown', 'Midtown', 'Suburb', 'Rural'], n_samples)
        no_show = np.random.choice(['No', 'Yes'], n_samples, p=[0.8, 0.2])  # 20% no-show rate
        
        # Create dataframe
        df = pd.DataFrame({
            'PatientID': range(1, n_samples + 1),
            'Age': ages,
            'Gender': genders,
            'ScheduledDay': scheduled_dates,
            'AppointmentDay': appointment_dates,
            'SMS_received': sms_received,
            'Neighbourhood': neighborhoods,
            'Hypertension': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'Diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Alcoholism': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'Handicap': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
            'No-show': no_show
        })
        
        # Save the sample dataset
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Created and saved sample dataset to {file_path}")
    
    # Data cleaning
    print("Cleaning data...")
    
    # Convert date columns to datetime
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    # Feature engineering
    print("Engineering features...")
    
    # Calculate lead time (days between scheduling and appointment)
    df['LeadTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    
    # Extract day of week
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.day_name()
    
    # Convert target variable to binary
    if df['No-show'].dtype == object:
        df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})
    
    # Handle missing values
    df = df.fillna({
        'Age': df['Age'].median(),
        'Gender': df['Gender'].mode()[0],
        'Neighbourhood': df['Neighbourhood'].mode()[0],
        'SMS_received': 0
    })
    
    # Remove any remaining rows with missing values
    df = df.dropna()
    
    print("Data preprocessing complete.")
    return df

# Function for exploratory data analysis
def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
    """
    print("\nPerforming exploratory data analysis...")
    
    # Basic statistics
    print("\nDataset shape:", df.shape)
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check target variable distribution
    no_show_counts = df['No-show'].value_counts(normalize=True) * 100
    print(f"\nNo-show distribution: {no_show_counts[1]:.2f}% no-shows, {no_show_counts[0]:.2f}% shows")
    
    # Analyze key features
    print("\nAnalyzing key features...")
    
    # Age distribution by no-show status
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='No-show', y='Age', data=df)
    plt.title('Age Distribution by No-show Status')
    plt.savefig('visualizations/age_distribution.png')
    
    # Lead time analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='LeadTime', hue='No-show', bins=30, kde=True)
    plt.title('Lead Time Distribution by No-show Status')
    plt.savefig('visualizations/lead_time_distribution.png')
    
    # Day of week analysis
    plt.figure(figsize=(12, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_no_show = pd.crosstab(df['AppointmentDayOfWeek'], df['No-show'], normalize='index') * 100
    day_no_show = day_no_show.reindex(day_order)
    day_no_show[1].plot(kind='bar')
    plt.title('No-show Rate by Day of Week')
    plt.ylabel('No-show Rate (%)')
    plt.savefig('visualizations/day_of_week_no_show.png')
    
    # SMS received analysis
    plt.figure(figsize=(10, 6))
    sms_no_show = pd.crosstab(df['SMS_received'], df['No-show'], normalize='index') * 100
    sms_no_show[1].plot(kind='bar')
    plt.title('No-show Rate by SMS Received')
    plt.xlabel('SMS Received (1=Yes, 0=No)')
    plt.ylabel('No-show Rate (%)')
    plt.savefig('visualizations/sms_no_show.png')
    
    print("EDA complete. Visualizations saved to 'visualizations' folder.")

# Function to build and evaluate the model
def build_model(df):
    """
    Build and evaluate a machine learning model for no-show prediction.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    print("\nBuilding prediction model...")
    
    # Prepare features and target
    X = df.drop(['No-show', 'PatientID', 'ScheduledDay', 'AppointmentDay'], axis=1)
    y = df['No-show']
    
    # Handle categorical features
    categorical_features = ['Gender', 'Neighbourhood', 'AppointmentDayOfWeek']
    numerical_features = ['Age', 'LeadTime', 'SMS_received', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train model pipeline
    print("Training Decision Tree model...")
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_pipeline.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Show', 'No-show'],
                yticklabels=['Show', 'No-show'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix.png')
    
    # Feature importance
    if hasattr(dt_pipeline.named_steps['classifier'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                for col in cols:
                    feature_names.extend([f"{col}_{cat}" for cat in trans.categories_[0]])
        
        # Get feature importances
        importances = dt_pipeline.named_steps['classifier'].feature_importances_
        
        # Plot top 10 features
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importances)[-10:]
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importances')
        plt.savefig('visualizations/feature_importance.png')
    
    print("Model building and evaluation complete.")
    return dt_pipeline, {'accuracy': accuracy, 'precision': precision, 'recall': recall}

# Function to generate optimization suggestions
def generate_suggestions(df, metrics):
    """
    Generate optimization suggestions based on the analysis.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        metrics (dict): Model evaluation metrics
    """
    print("\nGenerating optimization suggestions...")
    
    # Calculate no-show rates by different factors
    day_no_show = pd.crosstab(df['AppointmentDayOfWeek'], df['No-show'], normalize='index') * 100
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_no_show = day_no_show.reindex(day_order)
    
    high_risk_days = day_no_show[1].nlargest(3).index.tolist()
    low_risk_days = day_no_show[1].nsmallest(3).index.tolist()
    
    # SMS effectiveness
    sms_no_show = pd.crosstab(df['SMS_received'], df['No-show'], normalize='index') * 100
    sms_effectiveness = sms_no_show[1][0] - sms_no_show[1][1] if 1 in sms_no_show[1].index else 0
    
    # Lead time analysis
    lead_time_groups = pd.cut(df['LeadTime'], bins=[0, 7, 14, 30, 100], labels=['0-7 days', '8-14 days', '15-30 days', '30+ days'])
    lead_time_no_show = pd.crosstab(lead_time_groups, df['No-show'], normalize='index') * 100
    
    # Print suggestions
    print("\nOptimization Suggestions:")
    print(f"1. Schedule Optimization:")
    print(f"   - High-risk days for no-shows: {', '.join(high_risk_days)}")
    print(f"   - Consider overbooking by {metrics['precision']*100:.1f}% on these days")
    print(f"   - Low-risk days: {', '.join(low_risk_days)}")
    
    print(f"\n2. Patient Communication:")
    if sms_effectiveness > 0:
        print(f"   - SMS reminders reduce no-show rate by approximately {sms_effectiveness:.1f}%")
        print(f"   - Implement automated SMS reminders for all appointments")
        print(f"   - Consider sending multiple reminders (3 days before and day before)")
    
    print(f"\n3. Appointment Lead Time Management:")
    print(f"   - Optimal appointment lead time: {lead_time_no_show[1].idxmin()} (lowest no-show rate)")
    print(f"   - Try to schedule appointments within this timeframe when possible")
    
    print(f"\n4. Targeted Interventions:")
    print(f"   - Model accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"   - Use the prediction model to identify high-risk patients")
    print(f"   - Implement phone call confirmations for patients with >50% no-show probability")
    
    print("\nOptimization suggestions complete.")

# Main function
def main():
    import os
    
    # Create directories if they don't exist
    for directory in ['data', 'visualizations']:
        os.makedirs(directory, exist_ok=True)
    
    # File path for dataset
    file_path = 'data/healthcare_appointments.csv'
    
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    
    # Perform EDA
    perform_eda(df)
    
    # Build and evaluate model
    model, metrics = build_model(df)
    
    # Generate optimization suggestions
    generate_suggestions(df, metrics)
    
    print("\nHealthcare Appointment No-Show Prediction project completed successfully!")
    print("Check the 'visualizations' folder for output visualizations.")

if __name__ == "__main__":
    main()