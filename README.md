# AI Enabled Visa Status Prediction and Processing Time Estimator

## Infosys Springboard Project - Milestone 1

### Project Overview
This project builds a machine learning model to predict:
1. **Visa Status** - Whether an Indian visa application will be Approved, Rejected, or Pending
2. **Processing Time** - How many days it will take to process the visa application

### Milestone 1: Data Collection & Preprocessing (Completed)

#### What was done:
1. **Data Collection**: Generated synthetic dataset of 2000 Indian visa applications
2. **Missing Value Handling**: Filled null values using median (numeric) and mode (categorical)
3. **Categorical Encoding**: Applied label encoding and one-hot encoding
4. **Target Labels**: Created `processing_time_days` (numeric) and `visa_status_encoded` (0/1/2)

### Project Structure
```
├── data/
│   ├── raw/
│   │   └── visa_applications_raw.csv      # Original dataset with missing values
│   ├── processed/
│   │   ├── visa_applications_cleaned.csv  # Final cleaned dataset
│   │   └── encoding_mappings.txt          # Category to number mappings
│   └── README.md                          # Data dictionary
├── src/
│   ├── generate_synthetic_data.py         # Script to generate visa data
│   └── data_preprocessing.py              # Preprocessing pipeline
├── reports/
│   └── data_summary.txt                   # Dataset statistics
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

### How to Run

1. Install dependencies:
```bash
pip install pandas numpy
```

2. Generate raw data (already done):
```bash
python src/generate_synthetic_data.py
```

3. Run preprocessing (already done):
```bash
python src/data_preprocessing.py
```

### Dataset Features

| Feature | Description |
|---------|-------------|
| application_id | Unique ID for each application |
| visa_type | Type: Tourist, Business, Employment, Student, Medical, etc. |
| applicant_age | Age of the applicant |
| gender | Male or Female |
| nationality | Country of origin |
| processing_center | Indian city processing the application |
| financial_proof_usd | Financial documents amount in USD |
| processing_time_days | **TARGET**: Days to process (7-45 days) |
| visa_status | **TARGET**: Approved, Rejected, or Pending |

### Author
Rahul Makwana , Navprabhat Mishra - Infosys Springboard Intern
