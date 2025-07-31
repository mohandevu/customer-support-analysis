# Customer Support Ticket Analysis

This project performs an end-to-end analysis of customer support tickets using Python, including:

- Exploratory Data Analysis (EDA)
- Word Cloud for textual analysis
- Regression model to predict customer satisfaction

## Recommended Project Structure:
- customer-support-analysis/
├── data/
│   └── Customer_Support_Tickets.csv         # (optional placeholder or link in README)
├── outputs/
│   ├── *.png                                # All visualization images
│   ├── regression_results.csv
│   └── regression_coefficients.csv
├── analysis.py                              # Your script (you can rename it)
├── requirements.txt
├── .gitignore
└── README.md


## Features

- 📊 Visual insights into customer behavior and support metrics
- 💬 Text analysis of ticket descriptions
- 📈 Linear regression to estimate satisfaction ratings

- ## Usage
Place your CSV file in the data/ folder and run:

python analysis.py
Output visualizations and results will be saved to the outputs/ folder.

## Outputs
Department-wise ticket distribution
Age and product analysis
Resolution time distribution
Regression results with coefficients

## File Structure
Customer_Support_Tickets.csv       # Input file (not provided here)
analysis.py                        # Main script
outputs/                           # Contains all result files

## Requirements

Install the dependencies with:

pip install -r requirements.txt
