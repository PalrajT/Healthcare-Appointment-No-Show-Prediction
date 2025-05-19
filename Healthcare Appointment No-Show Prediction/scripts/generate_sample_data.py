# Generate Sample Healthcare Appointment Dataset

import pandas as pd
import numpy as np
import os

def generate_sample_data(n_samples=1000, output_file='../data/healthcare_appointments.csv'):
    """
    Generate a sample healthcare appointment dataset with realistic features.
    
    Args:
        n_samples (int): Number of appointments to generate
        output_file (str): Path to save the CSV file
    """
    print(f"Generating sample dataset with {n_samples} appointments...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate patient IDs (some patients have multiple appointments)
    patient_count = int(n_samples * 0.7)  # 70% unique patients
    patient_ids = np.random.randint(10000, 99999, patient_count)
    selected_patient_ids = np.random.choice(patient_ids, n_samples)
    
    # Generate demographic data
    ages = np.random.randint(0, 100, n_samples)
    genders = np.random.choice(['M', 'F'], n_samples)
    
    # Generate appointment dates
    base_date = pd.Timestamp('2022-01-01')
    scheduled_dates = [base_date + pd.Timedelta(days=np.random.randint(0, 180)) for _ in range(n_samples)]
    
    # Generate lead times (days between scheduling and appointment)
    lead_times = np.random.exponential(scale=10, size=n_samples).astype(int) + 1
    lead_times = np.clip(lead_times, 1, 60)  # Clip to reasonable range
    
    # Calculate appointment dates based on scheduled dates and lead times
    appointment_dates = [scheduled_dates[i] + pd.Timedelta(days=lead_times[i]) for i in range(n_samples)]
    
    # Generate SMS reminder flag (70% received SMS)
    sms_received = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Generate neighborhoods
    neighborhoods = np.random.choice(
        ['Downtown', 'Uptown', 'Midtown', 'Suburb', 'Rural', 'West End', 'East Side', 'North District', 'South District'], 
        n_samples
    )
    
    # Generate health conditions
    hypertension = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    diabetes = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    alcoholism = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    handicap = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02])
    
    # Generate appointment types
    appointment_types = np.random.choice(
        ['General Checkup', 'Specialist Consultation', 'Follow-up', 'Vaccination', 'Screening'], 
        n_samples, 
        p=[0.4, 0.25, 0.2, 0.1, 0.05]
    )
    
    # Generate no-show status with dependencies on other features
    # Base probability
    no_show_prob = np.ones(n_samples) * 0.2
    
    # Adjust based on SMS (decrease if SMS received)
    no_show_prob[sms_received == 1] -= 0.05
    
    # Adjust based on lead time (increase for longer lead times)
    no_show_prob[lead_times > 30] += 0.1
    no_show_prob[lead_times < 7] -= 0.05
    
    # Adjust based on age (younger patients miss more appointments)
    no_show_prob[ages < 30] += 0.05
    no_show_prob[ages > 65] -= 0.05
    
    # Adjust based on appointment type
    for i, apt_type in enumerate(appointment_types):
        if apt_type == 'Follow-up':
            no_show_prob[i] -= 0.03
        elif apt_type == 'Specialist Consultation':
            no_show_prob[i] -= 0.05
    
    # Clip probabilities to valid range
    no_show_prob = np.clip(no_show_prob, 0.05, 0.95)
    
    # Generate no-show status
    no_show = np.array(['No' if np.random.random() > prob else 'Yes' for prob in no_show_prob])
    
    # Create dataframe
    df = pd.DataFrame({
        'PatientID': selected_patient_ids,
        'Age': ages,
        'Gender': genders,
        'ScheduledDay': scheduled_dates,
        'AppointmentDay': appointment_dates,
        'LeadTime': lead_times,
        'AppointmentType': appointment_types,
        'SMS_received': sms_received,
        'Neighbourhood': neighborhoods,
        'Hypertension': hypertension,
        'Diabetes': diabetes,
        'Alcoholism': alcoholism,
        'Handicap': handicap,
        'No-show': no_show
    })
    
    # Add day of week
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.day_name()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample dataset saved to {output_file}")
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Total appointments: {n_samples}")
    print(f"No-show rate: {(df['No-show'] == 'Yes').mean() * 100:.2f}%")
    print(f"SMS received: {(df['SMS_received'] == 1).mean() * 100:.2f}%")
    
    return df

if __name__ == "__main__":
    # Generate a larger dataset for more realistic analysis
    generate_sample_data(n_samples=5000, output_file='../data/healthcare_appointments.csv')