import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Data Loading and Initial Preparation
# =============================================
print("Loading and preparing data...")
file_path = 'Customer_Support_Tickets.csv'
data = pd.read_csv(file_path)

# Set style for visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================
# Exploratory Data Analysis (EDA) Visualizations
# =============================================
print("\nPerforming exploratory data analysis...")

# 1.Ticket Distribution by Department
plt.figure(figsize=(10, 6))
department_counts = data['Department'].value_counts()
ax = sns.barplot(x=department_counts.index, y=department_counts.values, palette="viridis")
plt.title('Ticket Distribution by Department', fontsize=16)
plt.xlabel('Department', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')
plt.tight_layout()
plt.savefig('department_distribution.png')
plt.show()

# 2. Customer Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Customer Age'], bins=30, kde=True, color='skyblue')
plt.title('Customer Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(data['Customer Age'].mean(), color='red', linestyle='--', label=f'Mean: {data["Customer Age"].mean():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.show()

# 3. Product Analysis
plt.figure(figsize=(10, 8))
product_counts = data['Product Purchased'].value_counts().sort_values(ascending=True)
ax = sns.barplot(x=product_counts.values, y=product_counts.index, palette="magma")
plt.title('Ticket Counts by Product Purchased', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Product', fontsize=14)
for i, v in enumerate(product_counts.values):
    ax.text(v + 3, i, str(v), color='black', va='center')
plt.tight_layout()
plt.savefig('product_distribution.png')
plt.show()

# 4. Time to Resolution
data['Date Created'] = pd.to_datetime(data['Date Created'], dayfirst=True, errors='coerce')
data['Date Resolved'] = pd.to_datetime(data['Date Resolved'], dayfirst=True, errors='coerce')
data['Resolution Time (hours)'] = (data['Date Resolved'] - data['Date Created']).dt.total_seconds() / 3600

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(data[data['Resolution Time (hours)'] <= 1]['Resolution Time (hours)'], 
             bins=20, kde=True, color='green', ax=ax1)
ax1.set_title('Immediate Resolutions (0-1 hour)')
ax1.set_xlabel('Resolution Time (hours)')

# Longer resolutions (>1 hour)
sns.histplot(data[data['Resolution Time (hours)'] > 1]['Resolution Time (hours)'], 
             bins=30, kde=True, color='green', ax=ax2)
ax2.set_title('Longer Resolutions (>1 hour)')
ax2.set_xlabel('Resolution Time (hours)')

plt.tight_layout()
plt.savefig('resolution_time_split_view.png')
plt.show()

# 5. Customer Satisfaction Rating
plt.figure(figsize=(10, 6))
sns.histplot(data['Customer Satisfaction Rating'], bins=5, kde=True, discrete=True, color='purple')
plt.title('Customer Satisfaction Rating Distribution', fontsize=16)
plt.xlabel('Rating (1-5)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(range(1, 6))
plt.tight_layout()
plt.savefig('satisfaction_distribution.png')
plt.show()

# 6. Ticket Priority Distribution
plt.figure(figsize=(10, 6))
priority_counts = data['Ticket Priority'].value_counts()
ax = sns.barplot(x=priority_counts.index, y=priority_counts.values, palette="rocket")
plt.title('Ticket Priority Distribution', fontsize=16)
plt.xlabel('Priority Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')
plt.tight_layout()
plt.savefig('priority_distribution.png')
plt.show()

# 7. Correlation Heatmap
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 8))
    corr = data[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

# =============================================
# Text Analysis
# =============================================
print("\nPerforming text analysis...")

# 8. Word Cloud for Ticket Descriptions
if 'Ticket Description' in data.columns:
    text = ' '.join(desc for desc in data['Ticket Description'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Ticket Descriptions', fontsize=16)
    plt.tight_layout()
    plt.savefig('word_cloud.png')
    plt.show()

# =============================================
# Regression Analysis
# =============================================
print("\nPerforming regression analysis...")

# Convert date columns
data['First Response Time'] = pd.to_datetime(data['First Response Time'], dayfirst=True, errors='coerce')
data['Time to Resolution'] = pd.to_datetime(data['Time to Resolution'], dayfirst=True, errors='coerce')
data['Date of Purchase'] = pd.to_datetime(data['Date of Purchase'], dayfirst=True, errors='coerce')

# Feature Engineering
data['First_Response_Hours'] = (data['First Response Time'] - data['Date Created']).dt.total_seconds() / 3600
data['Resolution_Hours'] = (data['Time to Resolution'] - data['First Response Time']).dt.total_seconds() / 3600

# Drop irrelevant columns
data.drop(columns=[
    'Ticket ID', 'Customer Name', 'Customer Email', 'Customer Gender',
    'Product Purchased', 'Ticket Subject', 'Ticket Description',
    'Resolution', 'Assigned Engineer', 'Unnamed: 21'
], inplace=True, errors='ignore')

# Drop rows missing target or key engineered features
data.dropna(subset=['Customer Satisfaction Rating', 'First_Response_Hours', 'Resolution_Hours'], inplace=True)

# One-hot encode categorical features
categorical_cols = ['Ticket Type', 'Ticket Status', 'Ticket Priority', 'Ticket Channel', 'Department', 'Sentiment']
data = pd.get_dummies(data, columns=categorical_cols)

# Define features and target
X = data.drop(columns=[
    'Customer Satisfaction Rating',
    'Date of Purchase',
    'First Response Time',
    'Time to Resolution',
    'Date Resolved',
    'Date Created'
])

y = data['Customer Satisfaction Rating']

# Diagnostic: Check for remaining NaNs
print("\nMissing values in features before model training:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# Fix: Fill remaining NaNs with 0 or use imputation
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_rounded = np.clip(np.round(y_pred), 1, 5)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {round(mse, 2)}")
print(f"R-squared: {round(r2, 2)} — This means {round(r2 * 100)}% of the variability in customer satisfaction ratings can be explained by your model.")

# Coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nRegression Coefficients:")
print(coef_df)

# Sample predictions
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted (Scaled)': y_pred_rounded
})
print("\nSample Predictions:")
print(results_df.head())

# Export
results_df.to_csv('regression_results.csv', index=False)
coef_df.to_csv('regression_coefficients.csv', index=False)

print("\nAnalysis complete! All visualizations and results saved.")
