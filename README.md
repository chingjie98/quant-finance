# 📈 Quant Finance

👋 **Hello and welcome to my Quant Finance project!**  
This repository contains implementations of key quantitative finance models covering portfolio theory, pricing models, risk management, and fixed income.

## 🧠 Implemented Models

### 📊 Portfolio Theory
- **Markowitz Model** – Mean-variance portfolio optimization
- **CAPM (Capital Asset Pricing Model)** – Pricing assets based on systematic risk

### 💰 Option Pricing
- **Black-Scholes Model** – Analytical solution for European options
- **Black-Scholes with Monte Carlo Simulations** – Simulated paths for estimating option prices

### ⚠️ Risk Management
- **Value-at-Risk (VaR)** – Quantifying potential portfolio losses at a given confidence level
- **VaR with Monte Carlo Simulations** – Simulated VaR for non-linear portfolios

### 💵 Fixed Income
- **Vasicek Interest Rate Model** – Stochastic model for interest rate evolution
- **Bond Pricing using Vasicek Model** – Estimating bond prices based on simulated interest rate paths

## 📦 Requirements

This project uses Python and the following key libraries:

- `numpy` – for numerical operations and simulations  
- `pandas` – for data handling and time series analysis  
- `matplotlib` – for plotting results  
- `scipy` – for optimization and statistical functions  
- `yfinance` *(optional)* – for downloading financial data  
- `scikit-learn` – for regression analysis (e.g., CAPM beta estimation)

### 🔧 Installation

You can install the required packages using:

```bash
pip install numpy pandas matplotlib scipy yfinance scikit-learn
```
or
```bash
pip install -r requirements.txt
```
