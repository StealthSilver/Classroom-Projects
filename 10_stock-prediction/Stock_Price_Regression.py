import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (10, 5)

df = pd.read_csv('data.csv', parse_dates=['Date'])
print('Shape:', df.shape)
df.head()

df.info()

df.describe().T

print('Missing values:\n', df.isnull().sum(), sep='')
print('\nDuplicate rows:', df.duplicated().sum())
print('Companies:', df['Company'].nunique(), '| Months:', df['Date'].nunique())

# Remove duplicates & rows with missing values (none expected, but kept for safety)
df = df.drop_duplicates().dropna().reset_index(drop=True)

# Sanity checks: prices and volume must be positive, High >= Low
df = df[(df[['Open','High','Low','Close','Volume']] > 0).all(axis=1) & (df['High'] >= df['Low'])]

# Feature engineering
df['Month'] = df['Date'].dt.month
df['Monthly_Return_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
df = df.sort_values(['Company','Date']).reset_index(drop=True)
print('Clean shape:', df.shape)

# Outlier check with IQR on Close price
q1, q3 = df['Close'].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df['Close'] > q3 + 1.5*iqr) | (df['Close'] < q1 - 1.5*iqr)]
print(f'High-price outliers: {len(outliers)} rows ({len(outliers)/len(df):.1%}) - genuine expensive stocks, so we keep them')

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Close'], bins=40, kde=True, ax=ax[0])
ax[0].set_title('Close Price Distribution')
sns.histplot(np.log(df['Close']), bins=40, kde=True, color='green', ax=ax[1])
ax[1].set_title('Log of Close Price Distribution')
plt.show()

sector_avg = df.groupby('Sector')['Close'].mean().sort_values()
sector_avg.plot(kind='barh', color=sns.color_palette('viridis', len(sector_avg)))
plt.title('Average Close Price by Sector'); plt.xlabel('Avg Close Price')
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='Sector', y='Close')
plt.xticks(rotation=30); plt.title('Close Price Spread by Sector')
plt.show()

monthly = df.groupby('Date')['Close'].mean()
plt.plot(monthly.index, monthly.values, marker='o', linewidth=2)
plt.title('Average Close Price of All Companies - 12 Months')
plt.xlabel('Month'); plt.ylabel('Avg Close Price'); plt.xticks(rotation=30)
plt.show()

sector_trend = df.pivot_table(index='Date', columns='Sector', values='Close', aggfunc='mean')
sector_trend.plot(figsize=(12, 6), marker='o')
plt.title('Average Close Price by Sector over Time'); plt.ylabel('Avg Close Price')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()

top10 = df.groupby('Company')['Close'].mean().nlargest(10).sort_values()
top10.plot(kind='barh', color='teal')
plt.title('Top 10 Companies by Avg Close Price')
plt.show()

num_cols = ['Open','High','Low','Close','Volume','EPS','PE_Ratio','Market_Cap_Mn','Revenue_Growth_Pct','Debt_to_Equity']
plt.figure(figsize=(11, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
for a, col in zip(ax.ravel(), ['Open','EPS','Volume','Revenue_Growth_Pct']):
    sns.scatterplot(data=df, x=col, y='Close', alpha=0.4, ax=a)
    a.set_title(f'{col} vs Close')
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
vol = df.groupby('Date')['Volume'].mean()
ax[0].bar(vol.index.strftime('%b %y'), vol.values, color='orange')
ax[0].set_title('Average Trading Volume per Month')
ax[0].tick_params(axis='x', rotation=45)
sns.histplot(df['Monthly_Return_Pct'], bins=40, kde=True, color='purple', ax=ax[1])
ax[1].set_title('Monthly Return (%) Distribution')
plt.tight_layout(); plt.show()

df.groupby('Sector')['Monthly_Return_Pct'].mean().sort_values().plot(kind='barh', color='crimson')
plt.title('Avg Monthly Return (%) by Sector')
plt.show()

features = ['Open','High','Low','Volume','EPS','Revenue_Growth_Pct','Debt_to_Equity','Month']
X = pd.get_dummies(df[features + ['Sector']], columns=['Sector'], drop_first=True)
y = df['Close']

cutoff = pd.Timestamp(df['Date'].sort_values().unique()[-2])          # last 2 months = test
train_idx = df['Date'] < cutoff
X_train, X_test = X[train_idx], X[~train_idx]
y_train, y_test = y[train_idx], y[~train_idx]
print('Train:', X_train.shape, '| Test:', X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print('Intercept:', round(model.intercept_, 4))
coef = pd.Series(model.coef_, index=X.columns).sort_values()
coef

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f'R² Score : {r2:.4f}')
print(f'MAE      : {mae:.2f}')
print(f'RMSE     : {rmse:.2f}')
print(f'MAPE     : {mape:.2f}%')

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
# Actual vs Predicted
ax[0].scatter(y_test, y_pred, alpha=0.5)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[0].set_title('Actual vs Predicted'); ax[0].set_xlabel('Actual'); ax[0].set_ylabel('Predicted')
# Residual distribution
residuals = y_test - y_pred
sns.histplot(residuals, bins=30, kde=True, ax=ax[1], color='green')
ax[1].set_title('Residual Distribution')
# Residuals vs Predicted
ax[2].scatter(y_pred, residuals, alpha=0.5, color='orange')
ax[2].axhline(0, color='red', linestyle='--')
ax[2].set_title('Residuals vs Predicted'); ax[2].set_xlabel('Predicted'); ax[2].set_ylabel('Residual')
plt.tight_layout(); plt.show()

# Feature importance (coefficients)
coef.plot(kind='barh', figsize=(9, 6), color=np.where(coef > 0, 'seagreen', 'indianred'))
plt.title('Linear Regression Coefficients'); plt.show()

sample_company = 'Apex Technologies' if 'Apex Technologies' in df['Company'].values else df['Company'].iloc[0]
comp = df[df['Company'] == sample_company].copy()
comp_X = pd.get_dummies(comp[features + ['Sector']], columns=['Sector'], drop_first=True)
comp_X = comp_X.reindex(columns=X.columns, fill_value=0)
comp['Predicted_Close'] = model.predict(comp_X).round(2)
comp['Error'] = (comp['Close'] - comp['Predicted_Close']).round(2)
print('Sample company:', sample_company, '| Sector:', comp['Sector'].iloc[0])
comp[['Date','Open','Close','Predicted_Close','Error']].reset_index(drop=True)

plt.figure(figsize=(11, 5))
plt.plot(comp['Date'], comp['Close'], marker='o', label='Actual Close', linewidth=2)
plt.plot(comp['Date'], comp['Predicted_Close'], marker='s', linestyle='--', label='Predicted Close', linewidth=2)
plt.axvline(cutoff.to_pydatetime(), color='gray', linestyle=':', label='Train | Test split')
plt.title(f'{sample_company} - Actual vs Predicted Close Price')
plt.xlabel('Month'); plt.ylabel('Price'); plt.legend(); plt.xticks(rotation=30)
plt.show()

new_stock = {
    'Open': 150.0, 'High': 158.0, 'Low': 147.0, 'Volume': 2_500_000,
    'EPS': 6.5, 'Revenue_Growth_Pct': 3.2, 'Debt_to_Equity': 0.8,
    'Month': 11, 'Sector': 'Technology'
}
new_df = pd.get_dummies(pd.DataFrame([new_stock]), columns=['Sector'])
new_df = new_df.reindex(columns=X.columns, fill_value=0)
print(f"Predicted Close Price: {model.predict(new_df)[0]:.2f}")
