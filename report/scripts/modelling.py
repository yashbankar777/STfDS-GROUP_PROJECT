# # Preprocessing

! pip install -r requirements.txt -q


import pandas as pd
import numpy as np

df = pd.read_csv('./data/dataset.csv')

df.head()

print(df.shape)
df.info()
df.describe()

df.head()

# ## Feature Engineering

df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

df['HOUR'] = df['DATE_TIME'].dt.hour
df['DAY_OF_WEEK'] = df['DATE_TIME'].dt.dayofweek
df['MIN_SIN'] = np.sin(2 * np.pi * df['DATE_TIME'].dt.minute / 60)
df['MIN_COS'] = np.cos(2 * np.pi * df['DATE_TIME'].dt.minute / 60)
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
df['MONTH'] = df['DATE_TIME'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

df['SEASON'] = df['MONTH'].apply(get_season)
df = pd.get_dummies(df, columns=['SEASON'], prefix='SEASON')

df.shape

import altair as alt
import pandas as pd

# Calculate the correlation matrix for numerical columns
corr_matrix = df.select_dtypes('number').corr().reset_index().melt(id_vars='index')
corr_matrix.columns = ['Variable 1', 'Variable 2', 'Correlation']

# Create the heatmap
base = alt.Chart(corr_matrix).encode(
    x=alt.X('Variable 1:O', title=None),
    y=alt.Y('Variable 2:O', title=None)
)

heatmap = base.mark_rect().encode(
    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1]))
)

# Add text labels for the correlation coefficients
text = base.mark_text().encode(
    text=alt.Text('Correlation:Q', format='.2f'),
    color=alt.condition(
        "abs(datum.Correlation) > 0.5",  # Using a string expression instead
        alt.value('white'),
        alt.value('black')
    )
)

chart = (heatmap + text).properties(
    width=600,
    height=600
)

chart.save('./figs/multicollinearity_chart.pdf')

display(chart)

## Splitting the Dataset

from sklearn.model_selection import train_test_split


y_target = 'GROSS_DEMAND_MW'  # set the target variable
# y_target = 'TOTAL_DEMAND'  # set the target variable
# y_target = 'ROOFTOP_SOLAR_MW'  # set the target variable

df = df.sort_values('DATE_TIME')
df.drop(columns=[
    'TOTAL_DEMAND',
    'ROOFTOP_SOLAR_MW'
], inplace=True, errors='ignore')
df.drop(columns=['DATE_TIME'], inplace=True, errors='ignore')

n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

# X_train, X_temp = train_test_split(df, test_size=0.4, random_state=42)
# X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

X_train = df.iloc[:train_end]
X_val = df.iloc[train_end:val_end]
X_test = df.iloc[val_end:]

def plot_demand_dist(df, title):
    """
    Generates a binned distribution chart for GROSS_DEMAND_MW.
    """
    return alt.Chart(df).mark_bar().encode(
        alt.X("GROSS_DEMAND_MW:Q", bin=alt.Bin(maxbins=30), title="Gross Demand (MW)"),
        alt.Y("count()", title="Frequency"),
        tooltip=["count()"],
        color=alt.value('#0066ff')
    ).properties(
        width=250,
        height=200,
        title=title
    )

# Execute for all datasets
chart_train = plot_demand_dist(X_train, "X_train Distribution")
chart_val = plot_demand_dist(X_val, "X_val Distribution")
chart_test = plot_demand_dist(X_test, "X_test Distribution")

# Display side-by-side
final_dist_chart = (chart_train | chart_val | chart_test)
final_dist_chart.save('./figs/dist_chart.png')
display(final_dist_chart)

y_train, y_val, y_test = (
    X_train.pop(y_target),
    X_val.pop(y_target),
    X_test.pop(y_target),
)

df_names = ['X_train', 'X_val', 'X_test']


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# ## Data Transformation


# ### One-hot Encoding (OHE)

from sklearn.preprocessing import OneHotEncoder

ohe_encoder = OneHotEncoder()

for name in df_names:
    df_ = globals()[name]

    globals()[name] = df_.copy()


# ### Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=['number']).columns

scaler.fit(X_train[numeric_cols])

for name in df_names:
    df_ = globals()[name]
    
    df_[numeric_cols] = scaler.transform(df_[numeric_cols])

    for col in df_.columns:
        if df_[col].dtype == 'bool':
            df_[col] = df_[col].astype('category').cat.codes    

    globals()[name] = df_.copy()

from sklearn.linear_model import Ridge

lin_reg = Ridge(
    alpha=1.0,
    fit_intercept=True,
    random_state=42
)

lin_reg.fit(X_train, y_train)

lin_train_preds = lin_reg.predict(X_train)
lin_val_preds = lin_reg.predict(X_val)
lin_test_preds = lin_reg.predict(X_test)

print(f"Model Intercept: {lin_reg.intercept_}")

# %%
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_train_preds = xgb_model.predict(X_train)
xgb_val_preds = xgb_model.predict(X_val)
xgb_test_preds = xgb_model.predict(X_test)

# %%
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_state=42
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False
)

cat_train_preds = cat_model.predict(X_train)
cat_val_preds = cat_model.predict(X_val)
cat_test_preds = cat_model.predict(X_test)

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_metrics_row(y_train, p_train, y_val, p_val, y_test, p_test, n_feat):
    """Calculate MultiIndex row data for model evaluation."""
    def score(y, p):
        r2 = r2_score(y, p)
        # Prevent division by zero if n_feat is too high relative to sample size
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / max((len(y) - n_feat - 1), 1)
        
        # Calculate MAPE with a small epsilon to avoid division by zero
        mape = np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-10))) * 100
        
        return {
            'RMSE (MW)': round(np.sqrt(mean_squared_error(y, p)), 6),
            'MAE (MW)': round(mean_absolute_error(y, p), 6),
            'MAPE (%)': round(mape, 6),
            'Adj R2': round(adj_r2, 6)
        }
    
    tr, vl, ts = score(y_train, p_train), score(y_val, p_val), score(y_test, p_test)
    
    return {
        ('RMSE (MW)', 'Train'): tr['RMSE (MW)'], ('RMSE (MW)', 'Val'): vl['RMSE (MW)'], ('RMSE (MW)', 'Test'): ts['RMSE (MW)'],
        ('MAE (MW)', 'Train'): tr['MAE (MW)'],   ('MAE (MW)', 'Val'): vl['MAE (MW)'],   ('MAE (MW)', 'Test'): ts['MAE (MW)'],
        ('MAPE (%)', 'Train'): tr['MAPE (%)'],   ('MAPE (%)', 'Val'): vl['MAPE (%)'],   ('MAPE (%)', 'Test'): ts['MAPE (%)'],
        ('Adj R2', 'Train'): tr['Adj R2'],       ('Adj R2', 'Val'): vl['Adj R2'],       ('Adj R2', 'Test'): ts['Adj R2']
    }

n_feat = X_train.shape[1] 

# Data aggregation - Added n_feat to each call
data = {
    'Linear Regression': get_metrics_row(y_train, lin_train_preds, y_val, lin_val_preds, y_test, lin_test_preds, n_feat),
    'XGBoost': get_metrics_row(y_train, xgb_train_preds, y_val, xgb_val_preds, y_test, xgb_test_preds, n_feat),
    'CatBoost': get_metrics_row(y_train, cat_train_preds, y_val, cat_val_preds, y_test, cat_test_preds, n_feat)
}

# DataFrame construction
metrics_df = pd.DataFrame.from_dict(data, orient='index')
metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)

# Styling for display
styled_df = metrics_df.style.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', "#0077FF"), ('color', 'white')]},
    {'selector': 'td', 'props': [('text-align', 'center')]}
]).set_properties(**{'text-align': 'center'})

display(styled_df)

# %%
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Define parameter grid for Linear Model (Ridge)
# alpha is the regularization strength
linear_param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'fit_intercept': [True, False]
}

# Create GridSearchCV for Linear Model
linear_grid_search = GridSearchCV(
    Ridge(),
    linear_param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the grid search
linear_grid_search.fit(X_train, y_train)

# Get best model and parameters
best_linear_model = linear_grid_search.best_estimator_
print(f"Best Linear parameters: {linear_grid_search.best_params_}")
print(f"Best CV score (neg_mse): {linear_grid_search.best_score_}")

# Make predictions with best model
linear_grid_train_preds = best_linear_model.predict(X_train)
linear_grid_val_preds = best_linear_model.predict(X_val)
linear_grid_test_preds = best_linear_model.predict(X_test)

# %%
from sklearn.model_selection import GridSearchCV

# Define parameter grid for CatBoost
param_grid = {
    'depth': [x for x in range(4, 9)],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [500, 1000, 1500]
}

# Create GridSearchCV for CatBoost
cat_grid_search = GridSearchCV(
    CatBoostRegressor(loss_function='RMSE', random_state=42, verbose=False),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the grid search
cat_grid_search.fit(X_train, y_train)

# Get best model and parameters
best_cat_model = cat_grid_search.best_estimator_
print(f"Best CatBoost parameters: {cat_grid_search.best_params_}")
print(f"Best CV score (neg_mse): {cat_grid_search.best_score_}")

# Make predictions with best model
cat_grid_train_preds = best_cat_model.predict(X_train)
cat_grid_val_preds = best_cat_model.predict(X_val)
cat_grid_test_preds = best_cat_model.predict(X_test)

from sklearn.model_selection import GridSearchCV

# Define parameter grid for XGBoost
xgb_param_grid = {
    'max_depth': [x for x in range(4, 9)],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500]
}

# Create GridSearchCV for XGBoost
xgb_grid_search = GridSearchCV(
    xgb_model.__class__(random_state=42),
    xgb_param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the grid search
xgb_grid_search.fit(X_train, y_train)

# Get best model and parameters
best_xgb_model = xgb_grid_search.best_estimator_
print(f"Best XGBoost parameters: {xgb_grid_search.best_params_}")
print(f"Best CV score (neg_mse): {xgb_grid_search.best_score_}")

# Make predictions with best model
xgb_grid_train_preds = best_xgb_model.predict(X_train)
xgb_grid_val_preds = best_xgb_model.predict(X_val)
xgb_grid_test_preds = best_xgb_model.predict(X_test)

import altair as alt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming X_train is your feature matrix
n_feat = X_train.shape[1]

# Data aggregation - Added n_feat to each call
data = {
    "Linear Regression": get_metrics_row(
        y_train,
        linear_grid_train_preds,
        y_val,
        linear_grid_val_preds,
        y_test,
        linear_grid_test_preds,
        n_feat,
    ),
    "XGBoost": get_metrics_row(
        y_train,
        xgb_grid_train_preds,
        y_val,
        xgb_grid_val_preds,
        y_test,
        xgb_grid_test_preds,
        n_feat,
    ),
    "CatBoost": get_metrics_row(
        y_train,
        cat_grid_train_preds,
        y_val,
        cat_grid_val_preds,
        y_test,
        cat_grid_test_preds,
        n_feat,
    ),
}

# DataFrame construction
metrics_df = pd.DataFrame.from_dict(data, orient="index")
metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)

# Styling for display
styled_df = metrics_df.style.set_table_styles(
    [
        {
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("background-color", "#0077FF"),
                ("color", "white"),
            ],
        },
        {"selector": "td", "props": [("text-align", "center")]},
    ]
).set_properties(**{"text-align": "center"})

display(styled_df)

## Regression plots
alt.data_transformers.enable("vegafusion") # for large plots

plot_data = pd.DataFrame({
    'Actual': y_test.values,
    'Linear Regression': linear_grid_test_preds,
    'XGBoost': xgb_grid_test_preds,
    'CatBoost': cat_grid_test_preds
}).reset_index()

melted_df = plot_data.melt(id_vars=['index', 'Actual'], 
                           var_name='Model', 
                           value_name='Prediction')

actual_line = alt.Chart(melted_df).mark_line(color='black', strokeWidth=2, opacity=0.5).encode(
    x=alt.X('index:Q', title='Sample Index'),
    y=alt.Y('Actual:Q', title='MW'),
)

model_order = ['Linear Regression', 'XGBoost', 'CatBoost']

pred_line = alt.Chart(melted_df).mark_line(strokeDash=[5, 5]).encode(
    x='index:Q',
    y='Prediction:Q',
    color=alt.Color('Model:N', 
                    scale=alt.Scale(scheme='tableau10'),
                    sort=model_order) # disable sorting
)

final_chart = alt.layer(actual_line, pred_line).properties(
    width=700,
    height=200
).facet(
    row=alt.Row('Model:N', sort=model_order) # disable sorting
).resolve_scale(
    y='shared'
)

display(final_chart)


import altair as alt
import pandas as pd
import numpy as np

def plot_feature_importance(model, model_name, columns, save_path=None):
    g_color = '#058af7'
    
    # Extract importance based on model type
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        x_title = 'Contribution Score (Abs Coeff)'
    elif hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
        x_title = 'Contribution Score'
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        x_title = 'Contribution Score'
    else:
        raise ValueError("Model type not supported for importance extraction.")

    df = pd.DataFrame({
        'Feature': columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    chart = alt.Chart(df).mark_bar(color=g_color).encode(
        x=alt.X('Importance:Q', title=x_title),
        y=alt.Y('Feature:N', sort='-x', title='Predictor Variable'),
        tooltip=['Feature', 'Importance']
    ).properties(
        title=f'Key Drivers of Energy Demand ({model_name})',
        width=600,
        height=400
    )
    
    if save_path:
        chart.save(save_path)
    
    return chart

## Ridge (Linear Regression)
lin_importance_chart = plot_feature_importance(best_linear_model, 'Ridge Linear', X_train.columns, './figs/lin_imp_chart.png')

## CatBoost
cat_importance_chart = plot_feature_importance(best_cat_model, 'CatBoost', X_train.columns, './figs/xgb_imp_chart.png')

## XGBoost
xgb_importance_chart = plot_feature_importance(best_xgb_model, 'XGBoost', X_train.columns, './figs/cat_imp_chart.png')

display(lin_importance_chart)
display(cat_importance_chart)
display(xgb_importance_chart)


with pd.option_context('display.max_columns', None):
    display(X_train.head())

import pandas as pd
import numpy as np
import altair as alt
import scipy.stats as stats

def plot_qq_grid(models_dict):
    """
    models_dict: {
        'Linear': {'Train': y_p_train, 'Val': y_p_val, 'Test': y_p_test},
        'XGBoost': { ... },
        'CatBoost': { ... }
    }
    """
    all_data = []
    
    # Nested loop to build the 'Tidy' dataframe
    for model_name, splits in models_dict.items():
        for split_name, y_pred in splits.items():
            # Match the correct true values
            y_true = {'Train': y_train, 'Val': y_val, 'Test': y_test}[split_name]
            
            # Calculate Standardized Residuals
            res = (y_true - y_pred).values
            res_std = (res - np.mean(res)) / np.std(res)
            res_sorted = np.sort(res_std)
            
            # Theoretical Normal Quantiles
            theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(res_sorted)))
            
            tmp_df = pd.DataFrame({
                'Theoretical': theoretical,
                'Sample': res_sorted,
                'Model': model_name,
                'Split': split_name
            })
            
            # Downsample for browser performance
            if len(tmp_df) > 3000:
                tmp_df = tmp_df.sample(3000, random_state=42).sort_values('Theoretical')
            
            all_data.append(tmp_df)

    df = pd.concat(all_data)

    # Plot points
    points = alt.Chart(df).mark_circle(size=20, opacity=0.4).encode(
        x=alt.X('Theoretical:Q', title='Theoretical'),
        y=alt.Y('Sample:Q', title='Residuals'),
        color=alt.Color('Split:N', legend=None)
    ).properties(width=200, height=200)

    # The Reference Line (Standard Normal y=x)
    line_df = pd.DataFrame({'x': [-3, 3], 'y': [-3, 3]})
    line = alt.Chart(line_df).mark_line(color='red', strokeDash=[4,4]).encode(
        x='x:Q', y='y:Q'
    )

    # Combine and facet into 3x3
    grid = alt.layer(points, line, data=df).facet(
        column=alt.Column('Split:N', sort=['Train', 'Val', 'Test']),
        row=alt.Row('Model:N')
    ).resolve_scale(
        x='shared', y='shared'
    )

    return grid

results_grid = {
    'Linear': {'Train': best_linear_model.predict(X_train), 'Val': best_linear_model.predict(X_val), 'Test': best_linear_model.predict(X_test)},
    'XGBoost': {'Train': best_xgb_model.predict(X_train), 'Val': best_xgb_model.predict(X_val), 'Test': best_xgb_model.predict(X_test)},
    'CatBoost': {'Train': best_cat_model.predict(X_train), 'Val': best_cat_model.predict(X_val), 'Test': best_cat_model.predict(X_test)}
}
plot_qq_grid(results_grid)