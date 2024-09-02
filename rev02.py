# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:44:49 2024

@author: ieron
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, t, probplot
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
 
# Downloading and Synchronizing Data
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.dropna()
    return data

tickers = ['AAPL', 'GOOGL', 'SPY', 'TLT', 'GLD']  # Example tickers for stocks/indices and bonds
data = download_data(tickers, '2014-01-01', '2024-01-01')

# Calculate daily returns
returns = data.pct_change().dropna()

# Rescale returns for GARCH modeling
returns_rescaled = returns * 100

# Variance-Covariance Method
def var_cov_method(returns, alpha=0.975, time_horizon=1):
    portfolio_return = returns.sum(axis=1)
    mean_return = portfolio_return.mean()
    std_return = portfolio_return.std()
    var = norm.ppf(alpha) * std_return * np.sqrt(time_horizon)
    return var, portfolio_return

# Variance-Covariance with Student-t Distribution
def var_student_t(returns, alpha=0.975, dfs=[3, 4, 5, 6]):
    portfolio_return = returns.sum(axis=1)
    std_return = portfolio_return.std()
    vars = {df: t.ppf(alpha, df) * std_return for df in dfs}
    return vars, portfolio_return

# Historical Simulation Method
def historical_simulation(returns, alpha=0.975, years=5):
    portfolio_return = returns.sum(axis=1)
    sorted_returns = np.sort(portfolio_return[-(years*252):])
    index = int((1 - alpha) * len(sorted_returns))
    var = -sorted_returns[index]
    return var, portfolio_return

# Constant Conditional Correlation (CCC) with GARCH(1,1)
def ccc_garch(returns, alpha=0.975):
    garch_models = [arch_model(returns_rescaled[col], vol='Garch', p=1, q=1).fit(disp='off') for col in returns]
    garch_vols = np.array([model.conditional_volatility for model in garch_models]).T / 100  # Scale back
    
    portfolio_returns = returns.sum(axis=1)
    vol_portfolio = np.sqrt(np.sum(garch_vols**2, axis=1))
    
    var = -np.percentile(portfolio_returns / vol_portfolio, (1 - alpha) * 100) * np.mean(vol_portfolio)
    return var, portfolio_returns

# Filtered Historical Simulation with EWMA
def filtered_historical_simulation(returns, alpha=0.975, lambda_=0.94):
    portfolio_return = returns.sum(axis=1)
    ewma_vol = portfolio_return.ewm(span=(2/(1-lambda_)-1)).std()
    scaled_returns = portfolio_return / ewma_vol
    
    sorted_returns = np.sort(scaled_returns)
    index = int((1 - alpha) * len(sorted_returns))
    var = -sorted_returns[index] * np.mean(ewma_vol)
    return var, portfolio_return

# Expected Shortfall Calculation
def expected_shortfall(portfolio_return, var_value):
    es = portfolio_return[portfolio_return < -var_value].mean()
    return es

# Backtesting VaR models
def backtest_var(var_value, portfolio_return, alpha):
    violations = (portfolio_return < -var_value).sum()
    expected_violations = len(portfolio_return) * (1 - alpha)
    return violations, expected_violations

# Dynamic VaR Calculation using Moving Window Approach
def dynamic_var(returns, window=250, alpha=0.975):
    rolling_var = returns.rolling(window).apply(lambda x: -np.percentile(x, (1-alpha)*100))
    return rolling_var

# Empirical 5- and 10-days VaRs using Historical Simulation
def empirical_var_non_overlapping(returns, alpha, days):
    non_overlapping_returns = returns.iloc[::days].dropna()
    portfolio_return = non_overlapping_returns.sum(axis=1)
    sorted_returns = np.sort(portfolio_return)
    index = int((1 - alpha) * len(sorted_returns))
    var = -sorted_returns[index]
    return var

# Stress Testing
def stress_testing(data, shocks):
    stressed_values = {shock: data * (1 + shock) for shock in shocks}
    return stressed_values

# Define the alpha levels
alpha_975 = 0.975
alpha_990 = 0.99

# Variance-Covariance Method
var_975, portfolio_return = var_cov_method(returns, alpha=alpha_975)
var_990, _ = var_cov_method(returns, alpha=alpha_990)
es_975 = expected_shortfall(portfolio_return, var_975)
es_990 = expected_shortfall(portfolio_return, var_990)
violations_975, expected_violations_975 = backtest_var(var_975, portfolio_return, alpha=alpha_975)
violations_990, expected_violations_990 = backtest_var(var_990, portfolio_return, alpha=alpha_990)

# Variance-Covariance with Student-t Distribution
vars_student_t, student_t_portfolio_return = var_student_t(returns, alpha=alpha_975, dfs=[3, 4, 5, 6])
es_student_t = {df: expected_shortfall(student_t_portfolio_return, var) for df, var in vars_student_t.items()}
violations_student_t = {df: backtest_var(var, student_t_portfolio_return, alpha=alpha_975) for df, var in vars_student_t.items()}
expected_violations_student_t = {df: len(student_t_portfolio_return) * (1 - alpha_975) for df in vars_student_t.keys()}

# Historical Simulation Method
historical_var_975_5y, historical_portfolio_return = historical_simulation(returns, alpha=alpha_975, years=5)
historical_var_990_5y, _ = historical_simulation(returns, alpha=alpha_990, years=5)
historical_var_975_10y, _ = historical_simulation(returns, alpha=alpha_975, years=10)
historical_var_990_10y, _ = historical_simulation(returns, alpha=alpha_990, years=10)
es_historical_975_5y = expected_shortfall(historical_portfolio_return, historical_var_975_5y)
es_historical_990_5y = expected_shortfall(historical_portfolio_return, historical_var_990_5y)
es_historical_975_10y = expected_shortfall(historical_portfolio_return, historical_var_975_10y)
es_historical_990_10y = expected_shortfall(historical_portfolio_return, historical_var_990_10y)
violations_historical_975_5y, expected_violations_historical_975_5y = backtest_var(historical_var_975_5y, historical_portfolio_return, alpha=alpha_975)
violations_historical_990_5y, expected_violations_historical_990_5y = backtest_var(historical_var_990_5y, historical_portfolio_return, alpha=alpha_990)
violations_historical_975_10y, expected_violations_historical_975_10y = backtest_var(historical_var_975_10y, historical_portfolio_return, alpha=alpha_975)
violations_historical_990_10y, expected_violations_historical_990_10y = backtest_var(historical_var_990_10y, historical_portfolio_return, alpha=alpha_990)

# Constant Conditional Correlation (CCC) with GARCH(1,1)
ccc_garch_var_975, ccc_garch_portfolio_return = ccc_garch(returns, alpha=alpha_975)
ccc_garch_var_990, _ = ccc_garch(returns, alpha=alpha_990)
es_ccc_garch_975 = expected_shortfall(ccc_garch_portfolio_return, ccc_garch_var_975)
es_ccc_garch_990 = expected_shortfall(ccc_garch_portfolio_return, ccc_garch_var_990)
violations_ccc_garch_975, expected_violations_ccc_garch_975 = backtest_var(ccc_garch_var_975, ccc_garch_portfolio_return, alpha=alpha_975)
violations_ccc_garch_990, expected_violations_ccc_garch_990 = backtest_var(ccc_garch_var_990, ccc_garch_portfolio_return, alpha=alpha_990)

# Filtered Historical Simulation with EWMA
fhs_var_975, fhs_portfolio_return = filtered_historical_simulation(returns, alpha=alpha_975)
fhs_var_990, _ = filtered_historical_simulation(returns, alpha=alpha_990)
es_fhs_975 = expected_shortfall(fhs_portfolio_return, fhs_var_975)
es_fhs_990 = expected_shortfall(fhs_portfolio_return, fhs_var_990)
violations_fhs_975, expected_violations_fhs_975 = backtest_var(fhs_var_975, fhs_portfolio_return, alpha=alpha_975)
violations_fhs_990, expected_violations_fhs_990 = backtest_var(fhs_var_990, fhs_portfolio_return, alpha=alpha_990)

# Ensure data and returns are synchronized
returns = returns.dropna()
data = data.loc[returns.index]

# Empirical 5- and 10-days VaRs
empirical_var_5d_975 = empirical_var_non_overlapping(returns, alpha=alpha_975, days=5)
empirical_var_10d_975 = empirical_var_non_overlapping(returns, alpha=alpha_975, days=10)

# Dynamic VaR
dynamic_var_975 = dynamic_var(returns.sum(axis=1), window=250, alpha=alpha_975)
dynamic_var_990 = dynamic_var(returns.sum(axis=1), window=250, alpha=alpha_990)

# Stress Testing
shocks = [0.2, -0.2, 0.4, -0.4]
stressed_values = stress_testing(data, shocks)

# Results Summary
summary = {
    'Method': ['Variance-Covariance', 'Student-t (df=3)', 'Student-t (df=4)', 'Student-t (df=5)', 'Student-t (df=6)', 
               'Historical Simulation 5y', 'Historical Simulation 10y', 'CCC-GARCH', 'Filtered HS EWMA'],
    'VaR 97.5%': [var_975, vars_student_t[3], vars_student_t[4], vars_student_t[5], vars_student_t[6], 
                  historical_var_975_5y, historical_var_975_10y, ccc_garch_var_975, fhs_var_975],
    'VaR 99%': [var_990, vars_student_t[3], vars_student_t[4], vars_student_t[5], vars_student_t[6], 
                historical_var_990_5y, historical_var_990_10y, ccc_garch_var_990, fhs_var_990],
    'ES 97.5%': [es_975, es_student_t[3], es_student_t[4], es_student_t[5], es_student_t[6], 
                 es_historical_975_5y, es_historical_975_10y, es_ccc_garch_975, es_fhs_975],
    'ES 99%': [es_990, es_student_t[3], es_student_t[4], es_student_t[5], es_student_t[6], 
               es_historical_990_5y, es_historical_990_10y, es_ccc_garch_990, es_fhs_990],
    '5-day VaR 97.5%': [empirical_var_5d_975, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '10-day VaR 97.5%': [empirical_var_10d_975, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    'Actual Violations 97.5%': [violations_975, violations_student_t[3][0], violations_student_t[4][0], violations_student_t[5][0], violations_student_t[6][0], 
                                violations_historical_975_5y, violations_historical_975_10y, violations_ccc_garch_975, violations_fhs_975],
    'Expected Violations 97.5%': [expected_violations_975, expected_violations_student_t[3], expected_violations_student_t[4], expected_violations_student_t[5], expected_violations_student_t[6], 
                                  expected_violations_historical_975_5y, expected_violations_historical_975_10y, expected_violations_ccc_garch_975, expected_violations_fhs_975],
    'Actual Violations 99%': [violations_990, violations_student_t[3][0], violations_student_t[4][0], violations_student_t[5][0], violations_student_t[6][0], 
                              violations_historical_990_5y, violations_historical_990_10y, violations_ccc_garch_990, violations_fhs_990],
    'Expected Violations 99%': [expected_violations_990, expected_violations_student_t[3], expected_violations_student_t[4], expected_violations_student_t[5], expected_violations_student_t[6], 
                                expected_violations_historical_990_5y, expected_violations_historical_990_10y, expected_violations_ccc_garch_990, expected_violations_fhs_990]
}

summary_df = pd.DataFrame(summary)
print(summary_df)

# Saving the summary to a CSV file
summary_df.to_csv('VaR_ES_Summary.csv', index=False)

# Plotting VaR violations for each method over time
def plot_var_violations_over_time(portfolio_return, var_975, var_990, method_name):
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_return.index, portfolio_return, label='Portfolio Returns')
    plt.axhline(y=-var_975, color='r', linestyle='-', label=f'VaR 97.5% - {method_name}')
    plt.axhline(y=-var_990, color='g', linestyle='-', label=f'VaR 99% - {method_name}')
    plt.fill_between(portfolio_return.index, -var_975, -var_990, color='gray', alpha=0.2)
    plt.legend()
    plt.title(f'VaR Violations Over Time - {method_name}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.savefig(f'VaR_Violations_Over_Time_{method_name}.png')
    plt.show()

# Plot for Variance-Covariance Method
plot_var_violations_over_time(portfolio_return, var_975, var_990, 'Variance-Covariance')

# Plot for Student-t Method
for df in [3, 4, 5, 6]:
    plot_var_violations_over_time(student_t_portfolio_return, vars_student_t[df], vars_student_t[df], f'Student-t (df={df})')

# Plot for Historical Simulation Method
plot_var_violations_over_time(historical_portfolio_return, historical_var_975_5y, historical_var_990_5y, 'Historical Simulation 5y')
plot_var_violations_over_time(historical_portfolio_return, historical_var_975_10y, historical_var_990_10y, 'Historical Simulation 10y')

# Plot for CCC-GARCH Method
plot_var_violations_over_time(ccc_garch_portfolio_return, ccc_garch_var_975, ccc_garch_var_990, 'CCC-GARCH')

# Plot for Filtered Historical Simulation with EWMA Method
plot_var_violations_over_time(fhs_portfolio_return, fhs_var_975, fhs_var_990, 'Filtered HS EWMA')

# Detailed backtesting results for each method
def backtesting_details(method_name, portfolio_return, var_975, var_990):
    violations_975, expected_violations_975 = backtest_var(var_975, portfolio_return, alpha_975)
    violations_990, expected_violations_990 = backtest_var(var_990, portfolio_return, alpha_990)
    es_975 = expected_shortfall(portfolio_return, var_975)
    es_990 = expected_shortfall(portfolio_return, var_990)
    
    print(f"Backtesting Results for {method_name}:")
    print(f"  Actual Violations 97.5%: {violations_975}")
    print(f"  Expected Violations 97.5%: {expected_violations_975}")
    print(f"  Actual Violations 99%: {violations_990}")
    print(f"  Expected Violations 99%: {expected_violations_990}")
    print(f"  Expected Shortfall 97.5%: {es_975}")
    print(f"  Expected Shortfall 99%: {es_990}")
    print("\n")

# Display detailed backtesting results for each method
backtesting_details('Variance-Covariance', portfolio_return, var_975, var_990)
for df in [3, 4, 5, 6]:
    backtesting_details(f'Student-t (df={df})', student_t_portfolio_return, vars_student_t[df], vars_student_t[df])
backtesting_details('Historical Simulation 5y', historical_portfolio_return, historical_var_975_5y, historical_var_990_5y)
backtesting_details('Historical Simulation 10y', historical_portfolio_return, historical_var_975_10y, historical_var_990_10y)
backtesting_details('CCC-GARCH', ccc_garch_portfolio_return, ccc_garch_var_975, ccc_garch_var_990)
backtesting_details('Filtered HS EWMA', fhs_portfolio_return, fhs_var_975, fhs_var_990)

# Plotting annual VaR violations for each method
def plot_annual_var_violations(annual_summary, method_name):
    plt.figure(figsize=(14, 7))
    plt.plot(annual_summary['Year'], annual_summary['Actual Violations 97.5%'], marker='o', label='Actual Violations 97.5%')
    plt.plot(annual_summary['Year'], annual_summary['Expected Violations 97.5%'], marker='x', label='Expected Violations 97.5%')
    plt.legend()
    plt.title(f'Annual VaR Violations (97.5%) - {method_name}')
    plt.xlabel('Year')
    plt.ylabel('Number of Violations')
    plt.savefig(f'Annual_VaR_Violations_97.5_{method_name}.png')
    plt.show()

# Create annual summary for each method
def create_annual_summary(portfolio_return, var_975, var_990, method_name):
    years = portfolio_return.index.year.unique()
    annual_summary = []

    for year in years:
        yearly_returns = portfolio_return[portfolio_return.index.year == year]
        yearly_var_975 = -np.percentile(yearly_returns, (1-alpha_975)*100)
        yearly_es_975 = yearly_returns[yearly_returns < yearly_var_975].mean()
        violations_975 = (yearly_returns < -var_975).sum()
        expected_violations_975 = len(yearly_returns) * (1 - alpha_975)
        
        annual_summary.append({
            'Year': year,
            'Actual Violations 97.5%': violations_975,
            'Expected Violations 97.5%': expected_violations_975,
            'VaR 97.5%': var_975,
            'ES 97.5%': yearly_es_975
        })
        
    annual_summary_df = pd.DataFrame(annual_summary)
    annual_summary_df.to_csv(f'Annual_Backtesting_Summary_{method_name}.csv', index=False)
   
    print(annual_summary_df)

    plot_annual_var_violations(annual_summary_df, method_name)

# Create and plot annual summaries for each method
create_annual_summary(portfolio_return, var_975, var_990, 'Variance-Covariance')
for df in [3, 4, 5, 6]:
    create_annual_summary(student_t_portfolio_return, vars_student_t[df], vars_student_t[df], f'Student-t (df={df})')
create_annual_summary(historical_portfolio_return, historical_var_975_5y, historical_var_990_5y, 'Historical Simulation 5y')
create_annual_summary(historical_portfolio_return, historical_var_975_10y, historical_var_990_10y, 'Historical Simulation 10y')
create_annual_summary(ccc_garch_portfolio_return, ccc_garch_var_975, ccc_garch_var_990, 'CCC-GARCH')
create_annual_summary(fhs_portfolio_return, fhs_var_975, fhs_var_990, 'Filtered HS EWMA')

# Comparing empirical 5- and 10-day VaRs to one-day VaR with the square root of time rule
sqrt_time_var_5d = var_975 * np.sqrt(5)
sqrt_time_var_10d = var_975 * np.sqrt(10)

print(f"Empirical 5-day VaR (97.5%): {empirical_var_5d_975}")
print(f"Square root of time 5-day VaR (97.5%): {sqrt_time_var_5d}")
print(f"Empirical 10-day VaR (97.5%): {empirical_var_10d_975}")
print(f"Square root of time 10-day VaR (97.5%): {sqrt_time_var_10d}")

# Plotting empirical vs. square root of time VaRs
def plot_empirical_vs_sqrt_time(empirical_var, sqrt_time_var, days, alpha):
    plt.figure(figsize=(12, 6))
    plt.bar(['Empirical VaR', 'Sqrt Time VaR'], [-empirical_var, -sqrt_time_var], color=['blue', 'orange'])
    plt.title(f'Comparison of {days}-Day VaR (Alpha={alpha*100}%)')
    plt.ylabel('VaR')
    plt.savefig(f'Comparison_Empirical_vs_Sqrt_Time_{days}d_VaR.png')
    plt.show()

plot_empirical_vs_sqrt_time(empirical_var_5d_975, sqrt_time_var_5d, 5, alpha_975)
plot_empirical_vs_sqrt_time(empirical_var_10d_975, sqrt_time_var_10d, 10, alpha_975)

###############################################################################
###############################################################################
###############################################################################




























