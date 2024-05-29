import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

data = pd.read_csv('homework_sample_big.csv')
factors = pd.read_csv('factors_char_list.csv')

#### Split into train, validation and test

VARS = list(factors['variable'].values)
TARGET = ['stock_ret']
n_years_training = 10
train_years = data['year'].unique()[:n_years_training]
val_years = data['year'].unique()[n_years_training:n_years_training+2]
test_years = data['year'].unique()[n_years_training+2:]
print(train_years)
print(val_years)
print(test_years)


train = data[data['year'].isin(train_years)]
val = data[data['year'].isin(val_years)]
test = data[data['year'].isin(test_years)]

x_train = train[VARS]
y_train = train[TARGET]

x_val = val[VARS]
y_val = val[TARGET]

x_test = test[VARS]
y_test = test[TARGET]



#### Analyzing target

sns.histplot(y_train['stock_ret'], bins=100)
plt.title('Stock Returns Distribution')
plt.show()


from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

model_xgb = XGBRegressor(n_estimators=1000, random_state=123, max_depth=16, learning_rate=0.01)
model_lgbm = LGBMRegressor(n_estimators=1000, random_state=123, max_depth=10, num_leaves=2**10)
model_catboost = CatBoostRegressor(n_estimators=1000, random_state=123)
model_xgb.fit(x_train,y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=10)
model_lgbm.fit(x_train,y_train)
model_catboost.fit(x_train,y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=10)

##


def get_metrics(x, y, model):
    predictions = model.predict(x)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    metrics = {'mae': mae, 'mse': mse, 'r2': r2}
    return metrics

metrics_train_lgbm = get_metrics(x_train, y_train, model_lgbm)
metrics_val_lgbm = get_metrics(x_val, y_val, model_lgbm)
metrics_test_lgbm = get_metrics(x_test, y_test, model_lgbm)

metrics_train_xgb = get_metrics(x_train, y_train, model_xgb)
metrics_val_xgb = get_metrics(x_val, y_val, model_xgb)
metrics_test_xgb = get_metrics(x_test, y_test, model_xgb)

metrics_train_catboost = get_metrics(x_train, y_train, model_catboost)
metrics_val_catboost = get_metrics(x_val, y_val, model_catboost)
metrics_test_catboost = get_metrics(x_test, y_test, model_catboost)


### Comparison

print('Train:', metrics_train_lgbm)
print('Validation:', metrics_val_lgbm)  
print('Test:', metrics_test_lgbm)


print('Train:', metrics_train_xgb)
print('Validation:', metrics_val_xgb)  
print('Test:', metrics_test_xgb)



print('Train:', metrics_train_catboost)
print('Validation:', metrics_val_catboost)  
print('Test:', metrics_test_catboost)

###


test['pred_lgbm'] = model_lgbm.predict(x_test)
test['pred_xgb'] = model_xgb.predict(x_test)
test['pred_catboost'] = model_catboost.predict(x_test)
sns.histplot(test['pred_lgbm'], bins=100)
plt.title('Predictions Distribution')
plt.show()

sns.histplot(test['pred_xgb'], bins=100)
plt.title('Predictions Distribution')
plt.show()


sns.histplot(test['pred_catboost'], bins=100)
plt.title('Predictions Distribution')
plt.show()


import pandas as pd
import numpy as np

dates = np.unique(test['date'].values)

# Initialize portfolio and strategy performance tracking based on predictions
initial_cash = 1 
portfolio_value = initial_cash
cash = initial_cash
leverage = 1.5
strategy_returns_lgbm = []
n = 20
for date in dates:  
    test_date = test[test['date']==date]
    
    test_positive = test_date[test_date['pred_lgbm']>0]
    test_positive.sort_values(by = 'pred_lgbm',ascending=False, inplace=True)
    test_positive = test_positive.head(n)
    test_positive['investment_return'] = test_positive['stock_exret'] * leverage
    test_positive['portfolio_value'] = test_positive['investment_return']
    portfolio_value = test_positive['portfolio_value'].sum()
    # Track the portfolio value and returns
    #strategy_returns_lgbm.append((portfolio_value - initial_cash) / initial_cash)
    strategy_returns_lgbm.append(portfolio_value)
    cash = portfolio_value  # Update cash for the next month

# Calculate performance metrics
strategy_returns_lgbm = np.array(strategy_returns_lgbm)
sharpe_ratio = np.mean(strategy_returns_lgbm) / np.std(strategy_returns_lgbm) * np.sqrt(12)  # Annualized Sharpe ratio

print(f"Final portfolio value: {portfolio_value:.2f}")
print(f"Annualized Sharpe ratio: {sharpe_ratio:.2f}")

plt.figure(figsize=(15, 6))
sns.lineplot(x = [str(i) for i in dates], y = strategy_returns_lgbm)
plt.title('Strategy Returns with prediction LGBM model')
plt.xticks(rotation=90)
plt.show()




# Initialize portfolio and strategy performance tracking based on predictions
initial_cash = 1 
portfolio_value = initial_cash
cash = initial_cash
leverage = 1.5
strategy_returns_xgb = []


for date in dates:  
    test_date = test[test['date']==date]
    
    test_positive = test_date[test_date['pred_xgb']>0]
    test_positive.sort_values(by = 'pred_xgb',ascending=False, inplace=True)
    test_positive = test_positive.head(n)
    test_positive['investment_return'] = test_positive['stock_exret'] * leverage
    test_positive['portfolio_value'] = test_positive['investment_return']
    portfolio_value = test_positive['portfolio_value'].sum()
    # Track the portfolio value and returns
    #strategy_returns_xgb.append((portfolio_value - initial_cash) / initial_cash)
    strategy_returns_xgb.append(portfolio_value)
    cash = portfolio_value  # Update cash for the next month

# Calculate performance metrics
strategy_returns = np.array(strategy_returns_xgb)
sharpe_ratio = np.mean(strategy_returns_xgb) / np.std(strategy_returns_xgb) * np.sqrt(12)  # Annualized Sharpe ratio

print(f"Final portfolio value: {portfolio_value:.2f}")
print(f"Annualized Sharpe ratio: {sharpe_ratio:.2f}")

plt.figure(figsize=(15, 6))
sns.lineplot(x = [str(i) for i in dates], y = strategy_returns_xgb)
plt.title('Strategy Returns with prediction XGB model')
plt.xticks(rotation=90)
plt.show()



# Initialize portfolio and strategy performance tracking based on predictions
initial_cash = 1 
portfolio_value = initial_cash
cash = initial_cash
leverage = 1.5
strategy_returns_catboost = []


for date in dates:  
    test_date = test[test['date']==date]
    
    test_positive = test_date[test_date['pred_catboost']>0]
    test_positive.sort_values(by = 'pred_catboost',ascending=False, inplace=True)
    test_positive = test_positive.head(n)
    test_positive['investment_return'] = test_positive['stock_exret'] * leverage
    test_positive['portfolio_value'] = test_positive['investment_return']
    portfolio_value = test_positive['portfolio_value'].sum()
    # Track the portfolio value and returns
    #strategy_returns_catboost.append((portfolio_value - initial_cash) / initial_cash)
    strategy_returns_catboost.append(portfolio_value)
    cash = portfolio_value  # Update cash for the next month

# Calculate performance metrics
strategy_returns = np.array(strategy_returns_catboost)
sharpe_ratio = np.mean(strategy_returns_catboost) / np.std(strategy_returns_catboost) * np.sqrt(12)  # Annualized Sharpe ratio

print(f"Final portfolio value: {portfolio_value:.2f}")
print(f"Annualized Sharpe ratio: {sharpe_ratio:.2f}")

plt.figure(figsize=(15, 6))
sns.lineplot(x = [str(i) for i in dates], y = strategy_returns_catboost)
plt.title('Strategy Returns with prediction Catboost model')
plt.xticks(rotation=90)
plt.show()


# Initialize portfolio and strategy performance tracking based on random choice
initial_cash = 1 
portfolio_value = initial_cash
cash = initial_cash
leverage = 1.5
strategy_returns = []
for date in dates:  

    test_date = test[test['date']==date]

    test_date = test_date.sample(n, random_state=123)
    test_date['investment_return'] = test_date['stock_exret'] * leverage
    test_date['portfolio_value'] = test_date['investment_return']
    portfolio_value = test_date['portfolio_value'].sum()
    # Track the portfolio value and returns
    #strategy_returns.append((portfolio_value - initial_cash) / initial_cash)
    strategy_returns.append(portfolio_value)
    cash = portfolio_value  # Update cash for the next month

# Calculate performance metrics
strategy_returns = np.array(strategy_returns)
sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(12)  # Annualized Sharpe ratio

print(f"Final portfolio value: {portfolio_value:.2f}")
print(f"Annualized Sharpe ratio: {sharpe_ratio:.2f}")


plt.figure(figsize=(15, 6))
sns.lineplot(x = [str(i) for i in dates], y = strategy_returns)
plt.title('Strategy Returns with random model')
plt.xticks(rotation=90)
plt.show()



### Final results
dict_results = {
    'random_choice': strategy_returns,
    'lgbm':strategy_returns_lgbm,
    'xgb':strategy_returns_xgb,
    'catboost':strategy_returns_catboost
}

df_results = pd.DataFrame(dict_results)
df_results.describe()


### Cumulative return

cumulative_returns = df_results.cumsum()
cumulative_returns