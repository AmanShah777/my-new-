#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer # Just use simple imputer as it is not available
#%% Download telecom data file
url='https://raw.githubusercontent.com/shoaibulhaq/MyClassData/refs/heads/main/telecom.csv'
data=pd.read_csv(url)
#Create a copy of the original dataframe
df=data.copy()
#%% Data visualization
plt.subplot(3, 3, 1)
sns.histplot(data=df, x='age', bins=30)
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Count')

# 2. Monthly Charges vs Tenure
plt.subplot(3, 3, 2)
sns.scatterplot(data=df, x='tenure_months', y='monthly_charges')
plt.title('Monthly Charges vs Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Monthly Charges ($)')

# 3. Data Usage Distribution
plt.subplot(3, 3, 3)
sns.histplot(data=df, x='data_usage_gb', bins=30)
plt.title('Distribution of Data Usage')
plt.xlabel('Data Usage (GB)')
plt.ylabel('Count')

# 4. Contract Type Distribution
plt.subplot(3, 3, 4)
contract_counts = df['contract_type'].value_counts()
sns.barplot(x=contract_counts.index, y=contract_counts.values)
plt.title('Distribution of Contract Types')
plt.xticks(rotation=45)
plt.xlabel('Contract Type')
plt.ylabel('Count')

# 5. Support Calls Distribution
plt.subplot(3, 3, 5)
sns.countplot(data=df, x='support_calls')
plt.title('Distribution of Support Calls')
plt.xlabel('Number of Support Calls')
plt.ylabel('Count')

# 6. Churn Analysis by Contract Type
plt.subplot(3, 3, 6)
sns.barplot(data=df, x='contract_type', y='churn')
plt.title('Churn Rate by Contract Type')
plt.xticks(rotation=45)
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate')

# 7. Box Plot of Monthly Charges by Internet Service
plt.subplot(3, 3, 7)
sns.boxplot(data=df, x='internet_service', y='monthly_charges')
plt.title('Monthly Charges by Internet Service')
plt.xticks(rotation=45)
plt.xlabel('Internet Service')
plt.ylabel('Monthly Charges ($)')

# 8. Payment Method Distribution
plt.subplot(3, 3, 8)
payment_counts = df['payment_method'].value_counts()
sns.barplot(x=payment_counts.index, y=payment_counts.values)
plt.title('Distribution of Payment Methods')
plt.xticks(rotation=45)
plt.xlabel('Payment Method')
plt.ylabel('Count')

# 9. Correlation between Numerical Variables
plt.subplot(3, 3, 9)
numerical_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                 'call_minutes', 'data_usage_gb', 'support_calls']
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            square=True, cbar=True)
plt.title('Correlation Matrix')
#%% Introducing different types of missing data
# 1. MCAR: Completely random missing values in age
mcar_mask=np.random.choice([True, False], size=len(data), p=[0.1, 0.9])
df.loc[mcar_mask, 'age'] = np.nan

#Listwise deletion
df_listwise=df.dropna()

#Pairwise deletion
# Select relevant numeric variables 
vars_to_correlate = ['age', 'tenure_months', 'monthly_charges', 'data_usage_gb']
numeric_df = df[vars_to_correlate]

# Calculate correlation matrix using pairwise deletion
#The corr() method implicitly handles missing data using pairwise deletion, ensuring
#that each correlation coefficient is calculated based on the available data for that
#specific parir of variables.
correlation_matrix = numeric_df.corr(method='pearson')
   
#Simple imputation using sklearn Simple Imputer
simple_imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_simple_imputed = df.copy()
df_simple_imputed[numeric_cols] = simple_imputer.fit_transform(df[numeric_cols])

#Multiple imputation
mice_imputer = IterativeImputer(random_state=42)
df_mice_imputed = df.copy()
df_mice_imputed[numeric_cols] = mice_imputer.fit_transform(df[numeric_cols])

#Visualize imputation
fig,axes=plt.subplots(1,3)
sns.histplot(df.age,kde=True,ax=axes[0],label='Original')
sns.histplot(df_simple_imputed.age,kde=True,ax=axes[1],label='Simple Imputed')
sns.histplot(df_mice_imputed.age,kde=True,ax=axes[2],label='Multiple Imputed')

#%% 2. MAR: Missing values in data_usage_gb dependent on contract_type
df=data.copy()
mar_mask = (data['contract_type'] == 'Month-to-Month') & (np.random.random(len(data)) < 0.2)
df.loc[mar_mask, 'data_usage_gb'] = np.nan

#Simple mean imputation using pandas fillna
df['data_usage_gb']=df.groupby('contract_type')['data_usage_gb'].transform(lambda x: x.fillna(x.mean()))

#Regression imputation
#Regenerate the missing data in df
df.loc[mar_mask, 'data_usage_gb'] = np.nan
# Create mask for missing data_usage_gb values
missing_mask = df['data_usage_gb'].isna()

# Get complete cases for training
complete_cases = ~missing_mask

# Select features for imputation model
features = ['monthly_charges', 'call_minutes', 'tenure_months', 'support_calls']
x = df.loc[complete_cases, features]
y = df.loc[complete_cases, 'data_usage_gb']

# Fit regression model
reg_model = LinearRegression()
reg_model.fit(x, y)

# Predict missing values
X_missing = df.loc[missing_mask, features]
imputed_values = reg_model.predict(X_missing)

# Fill missing values with predictions
df.loc[missing_mask, 'data_usage_gb'] = imputed_values

#%% 3. MNAR: Missing values in monthly_charges more likely for higher values
df=data.copy()
mnar_mask = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)) & (np.random.random(len(df)) < 0.3)
df.loc[mnar_mask, 'monthly_charges'] = np.nan

# Create mask for missing monthly_charges
missing_mask = df['monthly_charges'].isna()

# Get complete cases for training
complete_cases = ~missing_mask

# Select features for imputation - using variables likely correlated with charge amounts
features = ['tenure_months', 'data_usage_gb', 'call_minutes', 'support_calls', 'total_charges']
x = df.loc[complete_cases, features]
y = df.loc[complete_cases, 'monthly_charges']

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train Random Forest model - better for non-random missingness patterns
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_scaled, y)

# Scale features for missing cases and predict
x_missing = scaler.transform(df.loc[missing_mask, features])
imputed_values = rf_model.predict(x_missing)

# Fill missing values with predictions
df.loc[missing_mask, 'monthly_charges'] = imputed_values

#%% Variable scaling
scaler = StandardScaler()
numeric_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'call_minutes', 'data_usage_gb']
df_scaled = data.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])   

# Create visualization
fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(15, 4*len(numeric_cols)))
fig.suptitle('Distribution Before and After Scaling', fontsize=16, y=1.02)

for idx, col in enumerate(numeric_cols):
    # Original distribution
    sns.histplot(data[col], kde=True, ax=axes[idx, 0])
    axes[idx, 0].set_title(f'Original {col}')
    axes[idx, 0].set_xlabel('Value')
    axes[idx, 0].set_ylabel('Count')
    
    # Scaled distribution
    sns.histplot(df_scaled[col], kde=True, ax=axes[idx, 1])
    axes[idx, 1].set_title(f'Scaled {col}')
    axes[idx, 1].set_xlabel('Value')
    axes[idx, 1].set_ylabel('Count')
    
plt.tight_layout()
#%% Variable transformation for skewed variables
df_transformed = data.copy()
df_transformed['data_usage_gb'] = np.log1p(df_transformed['data_usage_gb'])

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the original data
sns.histplot(df['data_usage_gb'], ax=axes[0], kde=True)
axes[0].set_title('Original Data Distribution')
axes[0].set_xlabel('Data Usage (GB)')

# Plot the transformed data
sns.histplot(df_transformed['data_usage_gb'], ax=axes[1], kde=True)
axes[1].set_title('Transformed Data Distribution')
axes[1].set_xlabel('Log1p Transformed Data Usage')
#%% Dimensionality reduction by using PCA
numeric_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'call_minutes', 'data_usage_gb']
#PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data[numeric_cols])

# Calculate explained variance ratio
explained_variance = pca.explained_variance_ratio_ * 100

# Plot explained variance
plt.bar(range(1, 3), explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by Principal Components')
plt.show()

# Create scatter plot with first two components
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% variance explained)')
plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% variance explained)')
plt.title('PCA: First Two Principal Components')
plt.grid(True, linestyle='--', alpha=0.7)

# Create component loadings visualization
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=numeric_cols
)

sns.heatmap(loadings, annot=True, cmap='RdBu', center=0)
plt.title('PCA Component Loadings')

# Print feature contributions
print("\nFeature Contributions to Principal Components:")
for i, pc in enumerate(['PC1', 'PC2']):
    print(f"\n{pc} major contributors:")
    pc_loadings = loadings[pc].abs().sort_values(ascending=False)
    for feat, val in pc_loadings.items():
        print(f"{feat}: {val:.3f}")

#%% Dimensionality reduction by using t-SNE 
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data[numeric_cols])
    
# Create DataFrame for visualization
tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE 1', 't-SNE 2'])

scatter = sns.scatterplot(data=tsne_df,x='t-SNE 1',y='t-SNE 2',alpha=0.6)   
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

#%% Create interaction terms
df_interactions = data.copy()
df_interactions['usage_per_charge'] = df['data_usage_gb'] / df['monthly_charges']
df_interactions['calls_per_tenure'] = df['support_calls'] / df['tenure_months']
#%%   Feature selection
numeric_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'call_minutes', 'data_usage_gb']

# Filter method using SelectKBest
selector = SelectKBest(score_func=f_classif, k=3)
selected_features = selector.fit_transform(data[numeric_cols], data.churn)
selected_feature_names = np.array(numeric_cols)[selector.get_support()]

# Get F-scores and p-values
f_scores, p_values = f_classif(data[numeric_cols], data.churn)

# Create DataFrame for visualization
feature_scores = pd.DataFrame({
    'Feature': numeric_cols,
    'F_Score': f_scores,
    'P_Value': p_values,
    'Selected': [feat in selected_feature_names for feat in numeric_cols]
})

# Sort by F-score
feature_scores = feature_scores.sort_values('F_Score', ascending=False)

# Visualization 1: F-scores bar plot
bars = plt.bar(range(len(feature_scores)), feature_scores['F_Score'])
plt.xticks(range(len(feature_scores)), feature_scores['Feature'], rotation=45)
plt.title('Feature Selection: F-scores for Each Variable')
plt.xlabel('Features')
plt.ylabel('F-score')

# Color selected features differently
for i, bar in enumerate(bars):
    if feature_scores.iloc[i]['Selected']:
        bar.set_color('darkblue')
    else:
        bar.set_color('lightgray')

plt.tight_layout()
plt.show()

# Print detailed results
print("\nFeature Selection Results:")
print("-" * 50)
print(feature_scores.to_string(index=False))
print("\nSelected Features:", ', '.join(selected_feature_names))
#%% Categorical encoding
df_encoded = data.copy()

# Label encoding for binary variables
le = LabelEncoder()
df_encoded['streaming_tv'] = le.fit_transform(df['streaming_tv'])
df_encoded['streaming_movies'] = le.fit_transform(df['streaming_movies'])

# One-hot encoding for multinomial variables
df_encoded = pd.get_dummies(df_encoded, columns=['contract_type', 'payment_method', 'internet_service'])

