"""
HW4 Event Study Solution
Market model: R_i,t = alpha_i + beta_i R_m,t + epsilon_i,t
Abnormal return: AR_i,t = R_i,t - (alpha_i + beta_i R_m,t)
CAR: cumulative sum of AR over the event window
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

DATA_DIR = Path('/mnt/data')
OUT_DIR = DATA_DIR / 'hw4_event_study_outputs'
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    'NFLX': DATA_DIR / 'HW_Data_2026_NFLXWBDPSKY(Netflix Stock Price History).csv',
    'WBD': DATA_DIR / 'HW_Data_2026_NFLXWBDPSKY(Warner Bros Discovery Stock Pri).csv',
    'PSKY': DATA_DIR / 'HW_Data_2026_NFLXWBDPSKY(Paramount Skydance Stock Price ).csv',
    'MKT': DATA_DIR / 'HW_Data_2026_NFLXWBDPSKY(S&P 500 Historical Data).csv',
}

STOCKS = ['NFLX', 'WBD', 'PSKY']
EST_START = '2024-11-04'
EST_END = '2025-11-05'
EVENT_DAYS = ['2025-12-05', '2026-02-27']


def load_price_return(path: Path, name: str) -> pd.DataFrame:
    """Load one price file and compute daily close-to-close simple return."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False).astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    df[name] = df['Price'].pct_change()
    return df[['Date', name]]


def event_window(df: pd.DataFrame, d_day: str, left: int = 10, right: int = 10) -> pd.DataFrame:
    """Return trading-day event window from left to right around D-day."""
    d_day = pd.Timestamp(d_day)
    if d_day not in set(df['Date']):
        raise ValueError(f'{d_day.date()} is not a trading day in the merged data.')
    idx = df.index[df['Date'].eq(d_day)][0]
    win = df.iloc[idx + left: idx + right + 1].copy()
    win['t'] = np.arange(left, right + 1)
    return win


def main():
    # 1. Merge returns
    merged = None
    for name, path in FILES.items():
        one = load_price_return(path, name)
        merged = one if merged is None else merged.merge(one, on='Date', how='inner')
    merged = merged.dropna().sort_values('Date').reset_index(drop=True)
    merged.to_csv(OUT_DIR / 'merged_daily_returns.csv', index=False)

    # 2. Estimate market model over the estimation period
    est = merged[(merged['Date'] >= EST_START) & (merged['Date'] <= EST_END)].copy()
    params = {}
    reg_rows = []
    for stock in STOCKS:
        X = sm.add_constant(est['MKT'])
        model = sm.OLS(est[stock], X).fit()
        alpha = model.params['const']
        beta = model.params['MKT']
        params[stock] = {'alpha': alpha, 'beta': beta}
        reg_rows.append({
            'stock': stock,
            'alpha': alpha,
            'alpha_t': model.tvalues['const'],
            'alpha_p_value': model.pvalues['const'],
            'beta': beta,
            'beta_t': model.tvalues['MKT'],
            'beta_p_value': model.pvalues['MKT'],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'n_obs': int(model.nobs),
        })
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT_DIR / '01_market_model_regression_results.csv', index=False)

    # 3. Event studies
    for d in EVENT_DAYS:
        win = event_window(merged, d, -10, 10)
        out = win[['Date', 't', 'MKT']].copy()
        for stock in STOCKS:
            alpha = params[stock]['alpha']
            beta = params[stock]['beta']
            out[f'{stock}_return'] = win[stock]
            out[f'{stock}_expected_return'] = alpha + beta * win['MKT']
            out[f'{stock}_AR'] = out[f'{stock}_return'] - out[f'{stock}_expected_return']
            out[f'{stock}_CAR'] = out[f'{stock}_AR'].cumsum()
        tag = pd.Timestamp(d).strftime('%Y%m%d')
        out.to_csv(OUT_DIR / f'02_event_window_AR_CAR_{tag}.csv', index=False)

        car_1 = out[(out['t'] >= -1) & (out['t'] <= 1)]
        car_1_summary = pd.DataFrame({
            'stock': STOCKS,
            'CAR_-1_to_+1': [car_1[f'{s}_AR'].sum() for s in STOCKS],
        })
        car_1_summary.to_csv(OUT_DIR / f'03_CAR_minus1_plus1_{tag}.csv', index=False)

        # CAR chart
        plt.figure(figsize=(9, 5))
        for stock in STOCKS:
            plt.plot(out['t'], out[f'{stock}_CAR'], marker='o', label=stock)
        plt.axvline(0, linestyle='--', linewidth=1)
        plt.axhline(0, linewidth=1)
        plt.title(f'CARs from t=-10 to t=+10, D-day = {d}')
        plt.xlabel('Trading days relative to D-day')
        plt.ylabel('CAR')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'04_CAR_chart_{tag}.png', dpi=200)
        plt.close()

    print('Done. Results saved in:', OUT_DIR)
    print('\nRegression results:')
    print(reg.round(6).to_string(index=False))
    for d in EVENT_DAYS:
        tag = pd.Timestamp(d).strftime('%Y%m%d')
        print(f'\nCAR(-1,+1), D-day={d}')
        print(pd.read_csv(OUT_DIR / f'03_CAR_minus1_plus1_{tag}.csv').round(6).to_string(index=False))


if __name__ == '__main__':
    main()
