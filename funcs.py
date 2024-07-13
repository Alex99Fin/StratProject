import pandas as pd
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression, Lasso
import numpy as np



def aligns_dfs(df): 
    all_dates = []
    for col in df.columns:
        if col.startswith('Index'):
            dates = pd.to_datetime(df[col])
            all_dates.append(dates)

    min_nan_dates = None
    min_nan_count = float('inf')

    for r in range(len(all_dates) // 2, 0, -1):
        for combo in combinations(all_dates, r):
            combined_dates = pd.concat(combo, axis=1)
            nan_count = combined_dates.isnull().sum().sum()
            if nan_count < min_nan_count:
                min_nan_count = nan_count
                min_nan_dates = combo

    df_aligned = pd.DataFrame(index=min_nan_dates[0])

    for dates_tuple in min_nan_dates:
        for i, (index_col, data_col) in enumerate(zip(df.columns[::2], df.columns[1::2])):
            if index_col.startswith('Index'):
                dates = pd.to_datetime(df[index_col])
                values = df[data_col]

                # Riallinea i dati utilizzando le date del set ottimale
                aligned_values = []
                for date in dates_tuple:
                    mask = dates <= date
                    if mask.any():
                        aligned_values.append(values[mask].iloc[-1])
                    else:
                        aligned_values.append(None)

                df_aligned[data_col] = aligned_values
                
    for col in df_aligned.columns:
        df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')
    df_aligned.ffill(inplace=True)
    return df_aligned


def precompute_exponential_weights(n, halflife):
    lamd = (1 / 2) ** (1 / halflife)
    weights = np.array([lamd ** k for k in range(n)])[::-1]
    return weights/np.sum(weights)


def make_ridge_on_pca_factors(df_spread_diff_maturities, df_pca, alpha, window_size, halflife):
    identity_matrix = np.eye(df_pca.shape[1])
    rolling_results = {}
    
    W = np.sqrt(precompute_exponential_weights(n=window_size, halflife=halflife))
    
    for column in df_spread_diff_maturities.columns:
        y = df_spread_diff_maturities[column]
        
        results_list = []
        index_list = []  
        
        for start in range(len(y) - window_size + 1):
            end = start + window_size

            y_window = y.iloc[start:end] * W
            X_window = df_pca.iloc[start:end].multiply(W, axis=0)

            model = Ridge(alpha=alpha)
            model.fit(X_window, y_window)
            
            y_pred = model.predict(X_window)
            r_squared = r2_score(y_window, y_pred)
            residuals = y_window - y_pred
            mse = np.mean(residuals**2)  
            
            n = len(y_window)  
            p = X_window.shape[1]  
            
            XTX = X_window.T @ X_window
            XTX_reg = XTX + alpha * identity_matrix
            XTX_reg_inv = np.linalg.inv(XTX_reg)
            sandwich = XTX_reg_inv @ XTX @ XTX_reg_inv.T  
            se = np.sqrt(np.diagonal(n / (n - p) * mse * sandwich))

            results = {
                'Intercept': model.intercept_,
                'Intercept_t':  model.intercept_/se[0],
                'R_squared': r_squared
            }
            
            for i in range(p):
                results[f'PC_{i+1}_coef'] = model.coef_[i]
                results[f'PC_{i+1}_t'] = model.coef_[i] / se[i]

            results_list.append(results)
            index_list.append(y_window.index[-1])
        rolling_results[column] = pd.DataFrame(results_list, index=index_list)
    
    return rolling_results

def make_lin_reg_on_pca_factors(df_spread_diff_maturities, df_pca, window_size, halflife):
    rolling_results = {}
    
    W = np.sqrt(precompute_exponential_weights(n=window_size, halflife=halflife))
    
    for column in df_spread_diff_maturities.columns:
        y = df_spread_diff_maturities[column]
        
        results_list = []
        index_list = []
        
        for start in range(len(y) - window_size + 1):
            end = start + window_size

            y_window = y.iloc[start:end] * W
            X_window = df_pca.iloc[start:end]
            
            X_window_scaled = X_window.multiply(W, axis=0)
            
            model = LinearRegression()
            model.fit(X_window_scaled, y_window)
            
            y_pred = model.predict(X_window_scaled)
            r_squared = r2_score(y_window, y_pred)
            residuals = y_window - y_pred
            mse = np.mean(residuals**2)
            
            n = len(y_window)
            p = X_window_scaled.shape[1]
            
            XTX = np.dot(X_window.T, X_window)
            se = np.sqrt(np.diagonal(n/(n-p) * (mse) * XTX))  
        
            results = {
                'Intercept': model.intercept_,               
                'Intercept_t': model.intercept_ / se[0],
                'R_squared': r_squared
            }
            
            for i in range(p):
                results[f'PC_{i+1}_coef'] = model.coef_[i]
                results[f'PC_{i+1}_SE'] = se[i]
                results[f'PC_{i+1}_t'] = model.coef_[i] / se[i]

            results_list.append(results)
            index_list.append(y_window.index[-1])
        
        rolling_results[column] = pd.DataFrame(results_list, index=index_list)
    
    return rolling_results