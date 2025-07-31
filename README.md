# customer-support-analysis
A Python-based data analysis project for exploring and modeling customer support ticket data. It includes detailed EDA, text analysis (word cloud), and a linear regression model to predict customer satisfaction ratings based on multiple features.

# Customer Support Ticket Analysis

This project performs an end-to-end analysis of customer support tickets using Python, including:

- Exploratory Data Analysis (EDA)
- Word Cloud for textual analysis
- Regression model to predict customer satisfaction

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

## Usage
Place your CSV file in the data/ folder and run:

bash
Copy
Edit
python analysis.py
Output visualizations and results will be saved to the outputs/ folder.

## Outputs
Department-wise ticket distribution
Age and product analysis
Resolution time distribution
Regression results with coefficients

## File Structure
bash
Copy
Edit
Customer_Support_Tickets.csv       # Input file (not provided here)
analysis.py                        # Main script
outputs/                           # Contains all result files

## Requirements

pandas
numpy
matplotlib
seaborn
wordcloud
scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
