import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Homework 3 - Efficient Frontier
# Data: DJIA20212025_Dow30CRSP_e3.csv
# Notation follows the lecture note exactly:
# A = 1' Sigma^{-1} mu = mu' Sigma^{-1} 1
# B = mu' Sigma^{-1} mu
# C = 1' Sigma^{-1} 1
# D = B*C - A^2
# ============================================================

# Number of time periods: Jan 2021 to Dec 2025 = 60 months
NT = 5 * 12

# Risk-free rate (%)
RF_ANNUAL = 3.74
RF_MONTHLY = RF_ANNUAL / 12.0

# ============================================================
# (1) Read data and estimate expected returns and covariance
# ============================================================

# Read CSV file
df = pd.read_csv("DJIA20212025_Dow30CRSP_e3.csv")

# Use column name directly
monthly_returns_raw = df["MthRet"].to_numpy()

# Number of securities inferred from data
NS = len(monthly_returns_raw) // NT

# Match MATLAB reshape behavior:
# MATLAB reshape(..., NT, NS) fills by columns, so use order='F'
ret_Monthly = monthly_returns_raw.reshape((NT, NS), order="F")

# Convert decimal returns to percentage returns
ret_Percentage_Monthly = 100 * ret_Monthly

# Expected monthly returns (arithmetic mean)
mu = np.mean(ret_Percentage_Monthly, axis=0).reshape(-1, 1)

# Covariance matrix of monthly returns
sigma = np.cov(ret_Percentage_Monthly, rowvar=False)

# ============================================================
# (2a) Efficient frontier without riskless asset
# ============================================================

ones = np.ones((NS, 1))
sigma_inv = np.linalg.inv(sigma)

# Lecture-note notation
A = float(ones.T @ sigma_inv @ mu)   # 1' Sigma^{-1} mu
B = float(mu.T @ sigma_inv @ mu)     # mu' Sigma^{-1} mu
C = float(ones.T @ sigma_inv @ ones) # 1' Sigma^{-1} 1
D = B * C - A**2

# Generate target returns for the frontier
target_returns = np.linspace(mu.min(), mu.max(), 200)

frontier_mu = []
frontier_std = []

for Rp in target_returns:
    # From lecture note:
    # lambda = (C*Rp - A) / D
    # gamma  = (B - A*Rp) / D
    # w_p = lambda * Sigma^{-1} mu + gamma * Sigma^{-1} 1
    lam = (C * Rp - A) / D
    gam = (B - A * Rp) / D
    w = lam * (sigma_inv @ mu) + gam * (sigma_inv @ ones) # (1)

    port_mu = float(w.T @ mu)
    port_var = float(w.T @ sigma @ w)
    port_std = np.sqrt(port_var)

    frontier_mu.append(port_mu)
    frontier_std.append(port_std)

frontier_mu = np.array(frontier_mu)
frontier_std = np.array(frontier_std)

# ============================================================
# (3) Minimum-Variance Portfolio (MVP)
# ============================================================

# From lecture note:
# w_MVP = Sigma^{-1} 1 / C
w_mvp = (sigma_inv @ ones) / C
mu_mvp = float(w_mvp.T @ mu)
var_mvp = float(w_mvp.T @ sigma @ w_mvp)
std_mvp = np.sqrt(var_mvp)

# ============================================================
# (4) Tangency portfolio with riskless asset
# ============================================================

# From lecture note:
# w_m = Sigma^{-1}(mu - Rf*1) / (A - C*Rf)
excess_mu = mu - RF_MONTHLY * ones
w_tan = (sigma_inv @ excess_mu) / (A - C * RF_MONTHLY)

mu_tan = float(w_tan.T @ mu)
var_tan = float(w_tan.T @ sigma @ w_tan)
std_tan = np.sqrt(var_tan)

sharpe_tan = (mu_tan - RF_MONTHLY) / std_tan

# ============================================================
# (2b) Capital Allocation Line (CAL)
# ============================================================

cal_std = np.linspace(0, max(frontier_std) * 1.2, 200)
cal_mu = RF_MONTHLY + sharpe_tan * cal_std

# ============================================================
# Print results
# ============================================================

np.set_printoptions(precision=6, suppress=True)

print("========== (1) Expected Returns mu ==========")
print(mu)

print("\n========== (1) Covariance Matrix sigma ==========")
print(sigma)

print("\n========== Lecture-note constants ==========")
print("A = 1' Sigma^{-1} mu =", A)
print("B = mu' Sigma^{-1} mu =", B)
print("C = 1' Sigma^{-1} 1 =", C)
print("D = B*C - A^2 =", D)

print("\n========== (3) Minimum-Variance Portfolio (MVP) ==========")
print("Expected return (monthly, %):", mu_mvp)
print("Standard deviation (monthly, %):", std_mvp)
print("Weights:")
print(w_mvp.reshape(-1, 1))

print("\n========== (4) Tangency Portfolio ==========")
print("Risk-free rate (monthly, %):", RF_MONTHLY)
print("Expected return (monthly, %):", mu_tan)
print("Standard deviation (monthly, %):", std_tan)
print("Sharpe ratio:", sharpe_tan)
print("Weights:")
print(w_tan.reshape(-1, 1))

# ============================================================
# Save portfolio weights
# ============================================================

asset_labels = [f"Asset_{i+1}" for i in range(NS)]

result_df = pd.DataFrame({
    "Asset": asset_labels,
    "Expected_Return_%": mu.flatten(),
    "MVP_Weight": w_mvp.flatten(),
    "Tangency_Weight": w_tan.flatten()
})

result_df.to_csv("portfolio_results.csv", index=False)

# ============================================================
# Plot all results in one figure
# ============================================================

plt.figure(figsize=(10, 7))

# Efficient frontier without riskless asset
plt.plot(frontier_std, frontier_mu, label="Efficient Frontier (No Riskless Asset)")

# Capital Allocation Line
plt.plot(cal_std, cal_mu, label="Capital Allocation Line (With Riskless Asset)")

# MVP
plt.plot(std_mvp, mu_mvp, 'o', markersize=8, label="MVP")

# Tangency portfolio
plt.plot(std_tan, mu_tan, 'o', markersize=8, label="Tangency Portfolio")

# Risk-free asset
plt.plot(0, RF_MONTHLY, 'o', markersize=8, label="Risk-Free Asset")

plt.xlabel("Portfolio Standard Deviation (%)")
plt.ylabel("Expected Portfolio Return (%)")
plt.title("Efficient Frontier, CAL, MVP, and Tangency Portfolio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("efficient_frontier_plot.png", dpi=300)
plt.show()
