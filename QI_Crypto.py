# =============================================================================
# Alpha Strategy Code Using Binance Data + Enhanced Signal Logic
# =============================================================================

# Install Required Libraries (uncomment if needed)
# !pip install pandas matplotlib statsmodels scikit-learn joblib numba openpyxl requests

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for faster plotting
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import warnings
import datetime
import os
from joblib import Parallel, delayed
import multiprocessing
import requests
from time import sleep

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIGURATION: Binance API + Directory/Paths
# --------------------------------------------------------------------------
BINANCE_BASE_URL = "https://api.binance.com"
TOP_SYMBOL_LIMIT = 100  # number of symbols to analyse
QUOTE_ASSET = "USDT"
REQUEST_TIMEOUT = 10
REQUEST_SLEEP_SECONDS = 0.1
KLINE_INTERVAL = "1d"

# Output paths
today_date = datetime.datetime.today().strftime('%Y-%m-%d')
OUTPUT_FILE_PATH = os.path.join(os.getcwd(), f'analysis_results_{today_date}.xlsx')
PLOT_DIR = os.path.join(os.getcwd(), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Function to load top Binance symbols
# --------------------------------------------------------------------------
def fetch_top_binance_symbols(limit=TOP_SYMBOL_LIMIT, quote_asset=QUOTE_ASSET):
    """
    Fetch the most active Binance spot symbols quoted in the desired asset.

    Parameters:
    - limit (int): Number of symbols to return.
    - quote_asset (str): Quote asset to filter on (e.g., "USDT").

    Returns:
    - List[str]: List of Binance symbols (e.g., ['BTCUSDT', ...]).
    - Dict[str, str]: Dictionary mapping symbols to simple descriptions.
    """
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to retrieve Binance symbols: {exc}") from exc

    tickers_data = response.json()
    leveraged_suffixes = ("UP", "DOWN", "BULL", "BEAR")
    filtered_symbols = []

    for item in tickers_data:
        symbol = item.get("symbol", "")
        if not symbol.endswith(quote_asset):
            continue

        base_symbol = symbol[: -len(quote_asset)]
        if base_symbol.endswith(leveraged_suffixes):
            continue  # skip leveraged tokens

        try:
            volume = float(item.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            volume = 0.0

        filtered_symbols.append((symbol, volume))

    if not filtered_symbols:
        raise RuntimeError("No Binance symbols matched the requested filters.")

    filtered_symbols.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [symbol for symbol, _ in filtered_symbols[:limit]]
    descriptions = {symbol: f"{symbol} spot pair" for symbol in top_symbols}

    return top_symbols, descriptions


# --------------------------------------------------------------------------
# Fetch Data via Binance API
# --------------------------------------------------------------------------
def fetch_symbol_daily_closes(symbol, session, interval=KLINE_INTERVAL):
    """
    Fetch full-history daily close data for a single Binance symbol.

    Parameters:
    - symbol (str): Binance trading pair symbol (e.g., 'BTCUSDT').
    - session (requests.Session): Reusable HTTP session.
    - interval (str): Kline interval (default '1d').

    Returns:
    - pd.Series: Series indexed by datetime with float close prices.
    """
    endpoint = f"{BINANCE_BASE_URL}/api/v3/klines"
    limit = 1000
    start_time = 0
    all_rows = []

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time
        }

        try:
            response = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching klines for {symbol}: {exc}")
            break

        rows = response.json()
        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < limit:
            break

        last_open_time = rows[-1][0]
        start_time = last_open_time + 1
        sleep(REQUEST_SLEEP_SECONDS)

    if not all_rows:
        return pd.Series(dtype=float)

    kline_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=kline_columns)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("Date", inplace=True)

    closes = df["close"].astype(float)
    closes = closes[~closes.index.duplicated(keep="last")]

    return closes


def fetch_data_from_binance(tickers, interval=KLINE_INTERVAL):
    """
    Fetch historical close data for a list of Binance symbols.

    Parameters:
    - tickers (List[str]): Binance trading symbols.
    - interval (str): Kline interval (default '1d').

    Returns:
    - pd.DataFrame: DataFrame indexed by date where columns correspond to symbols.
    """
    if not tickers:
        return pd.DataFrame()

    combined_data = {}
    session = requests.Session()

    try:
        for symbol in tickers:
            print(f"Fetching data from Binance for {symbol}...")
            closes = fetch_symbol_daily_closes(symbol, session, interval=interval)
            if closes.empty:
                print(f"No data returned for {symbol}. Skipping.")
                continue
            combined_data[symbol] = closes
            sleep(REQUEST_SLEEP_SECONDS)
    finally:
        session.close()

    if not combined_data:
        return pd.DataFrame()

    adj_close_df = pd.DataFrame(combined_data)
    adj_close_df.sort_index(inplace=True)
    adj_close_df = adj_close_df[~adj_close_df.index.duplicated(keep="last")]
    adj_close_df.dropna(how='all', inplace=True)

    return adj_close_df


# --------------------------------------------------------------------------
# Load Tickers and Descriptions
# --------------------------------------------------------------------------
try:
    TICKERS, TICKER_DESCRIPTIONS = fetch_top_binance_symbols()
    print(f"Loaded {len(TICKERS)} Binance symbols quoted in {QUOTE_ASSET}.")
except Exception as e:
    print(f"Error loading Binance tickers: {e}")
    TICKERS = []
    TICKER_DESCRIPTIONS = {}


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def preprocess_ticker_data(series):
    """
    Preprocesses the data for a single ticker by forward-filling,
    ensuring a business-day frequency, and dropping NaN.
    """
    series = series.dropna()
    series = series.asfreq('B')
    series.fillna(method='ffill', inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index)
    return series


def log_transform(series):
    """Apply natural log transform, avoiding log(0) errors."""
    series = series.replace(0, 1e-9)
    return np.log(series)


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
def decompose_time_series(series, model='additive', period=252):
    """
    Decomposes the time series into trend, seasonal, and residual components.
    """
    print(f"Performing '{model}' decomposition (period={period})...")
    return seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')


def plot_decomposition(result, title='Time Series Decomposition', ticker='', save_dir=PLOT_DIR):
    """Plot the decomposition results."""
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"{title} - {ticker}", fontsize=16)

    plt.subplot(411)
    plt.plot(result.observed, color='blue', label='Observed')
    plt.legend(loc='upper left')

    plt.subplot(412)
    plt.plot(result.trend, color='orange', label='Trend')
    plt.legend(loc='upper left')

    plt.subplot(413)
    plt.plot(result.seasonal, color='green', label='Seasonal')
    plt.legend(loc='upper left')

    plt.subplot(414)
    plt.plot(result.resid, color='red', label='Residual')
    plt.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"{ticker}_decomposition.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Trend Slope Calculation (Vectorized Rolling Regression)
# --------------------------------------------------------------------------
def calculate_trend_slope_vectorized(trend_series, window=252):
    """
    Calculates rolling linear regression slope on the trend component
    using a vectorized approach.
    """
    y = trend_series.dropna().values
    n = len(y)
    if n < window:
        return pd.Series(dtype=float)

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        rolling_windows = sliding_window_view(y, window_shape=window)
    except AttributeError:
        # fallback for older numpy versions
        shape = (n - window + 1, window)
        strides = (y.strides[0], y.strides[0])
        rolling_windows = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

    X = np.arange(window)
    X_mean = X.mean()
    X_var = ((X - X_mean) ** 2).sum()

    Y_mean = rolling_windows.mean(axis=1)
    cov = (rolling_windows - Y_mean[:, np.newaxis]) @ (X - X_mean)

    slope = cov / X_var
    idx = trend_series.dropna().index[window-1:]
    return pd.Series(data=slope, index=idx)


# --------------------------------------------------------------------------
# Residual Z-Score
# --------------------------------------------------------------------------
def calculate_residual_zscore(residual_series, window=252):
    """
    Calculates rolling Z-score of the residual component.
    """
    rolling_mean = residual_series.rolling(window=window, min_periods=window).mean()
    rolling_std = residual_series.rolling(window=window, min_periods=window).std()
    z_score = (residual_series - rolling_mean) / rolling_std
    return z_score


# --------------------------------------------------------------------------
# Trend Strength Logic
# --------------------------------------------------------------------------
def get_trend_strength(slope):
    """
    Determines a multiplier based on absolute slope.
    """
    abs_slope = abs(slope)
    if abs_slope < 0.01:
        return 1.0  # Weak trend
    elif 0.01 <= abs_slope < 0.05:
        return 1.5  # Moderate trend
    else:
        return 2.0  # Strong trend


# --------------------------------------------------------------------------
# Backtest Strategy with Variable Position Sizing
# --------------------------------------------------------------------------
def backtest_strategy(analysis_df, ticker='', description=''):
    """
    Enhanced backtest:
    - Buys incrementally when Z-score < -2
    - Sells incrementally when Z-score > 2
    - Position sizing is influenced by Z-score magnitude * trend strength.
    - max_position = 3.0 (example) to limit total position.

    Returns:
    - performance_metrics: dict
    - last_trade_signal_details: (signal_type, signal_date, signal_strength)
    """
    analysis_df['Position'] = 0.0
    last_trade_signal_type = 'N/A'
    last_signal_date = 'N/A'
    last_signal_strength = 'N/A'

    position = 0.0
    max_position = 3.0  # maximum position size
    for idx, row in analysis_df.iterrows():
        current_slope = row['Trend_Slope'] if not pd.isna(row['Trend_Slope']) else 0.0
        trend_strength = get_trend_strength(current_slope)

        z_score = row['Residual_Z_Score']
        signal_strength = 0.0
        signal_type = None

        if z_score < -2:
            # Potential Buy
            signal_strength = abs(z_score) * trend_strength
            # Cap at max position
            if position + signal_strength > max_position:
                signal_strength = max_position - position

            if signal_strength > 0:
                signal_type = 'Buy'
                position += signal_strength
                last_trade_signal_type = signal_type
                last_signal_date = idx.strftime('%Y-%m-%d')
                last_signal_strength = round(signal_strength, 2)
                print(f"[{ticker}] Buy on {last_signal_date} (Strength: {last_signal_strength})")

        elif z_score > 2:
            # Potential Sell
            signal_strength = z_score * trend_strength
            # Cap so we don't go negative
            if position - signal_strength < 0:
                signal_strength = position

            if signal_strength > 0:
                signal_type = 'Sell'
                position -= signal_strength
                last_trade_signal_type = signal_type
                last_signal_date = idx.strftime('%Y-%m-%d')
                last_signal_strength = round(signal_strength, 2)
                print(f"[{ticker}] Sell on {last_signal_date} (Strength: {last_signal_strength})")

        analysis_df.at[idx, 'Position'] = position

    # --------------------------------------------------------------------------
    # Modify Position_Shifted to apply signals on the same day
    # --------------------------------------------------------------------------
    # Remove the shift to apply the position on the same day
    analysis_df['Position_Shifted'] = analysis_df['Position']  # Changed from shift(1)

    # Compute returns
    analysis_df['Log_Return'] = analysis_df['Adj_Close'].diff()
    analysis_df['Strategy_Return'] = analysis_df['Position_Shifted'] * analysis_df['Log_Return']
    analysis_df['Strategy_Return'].fillna(0.0, inplace=True)

    # Cumulative returns
    analysis_df['Cumulative_Strategy_Return'] = analysis_df['Strategy_Return'].cumsum().apply(np.exp)
    analysis_df['Cumulative_Buy_and_Hold_Return'] = analysis_df['Log_Return'].cumsum().apply(np.exp)

    # Performance stats
    strategy_total_return = analysis_df['Cumulative_Strategy_Return'].iloc[-1] - 1
    buy_hold_total_return = analysis_df['Cumulative_Buy_and_Hold_Return'].iloc[-1] - 1
    strategy_years = (analysis_df.index[-1] - analysis_df.index[0]).days / 365.25

    if strategy_years <= 0:
        strategy_annual_return = 0.0
        buy_hold_annual_return = 0.0
    else:
        strategy_annual_return = (analysis_df['Cumulative_Strategy_Return'].iloc[-1])**(1/strategy_years) - 1
        buy_hold_annual_return = (analysis_df['Cumulative_Buy_and_Hold_Return'].iloc[-1])**(1/strategy_years) - 1

    strategy_volatility = analysis_df['Strategy_Return'].std() * np.sqrt(252)
    buy_hold_volatility = analysis_df['Log_Return'].std() * np.sqrt(252)

    strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility != 0 else 0.0
    buy_hold_sharpe = buy_hold_annual_return / buy_hold_volatility if buy_hold_volatility != 0 else 0.0

    # Drawdown
    analysis_df['Strategy_Cumulative'] = analysis_df['Strategy_Return'].cumsum().apply(np.exp)
    analysis_df['Strategy_Cumulative_Max'] = analysis_df['Strategy_Cumulative'].cummax()
    analysis_df['Strategy_Drawdown'] = analysis_df['Strategy_Cumulative'] / analysis_df['Strategy_Cumulative_Max'] - 1
    strategy_max_drawdown = analysis_df['Strategy_Drawdown'].min()

    analysis_df['Buy_Hold_Cumulative'] = analysis_df['Log_Return'].cumsum().apply(np.exp)
    analysis_df['Buy_Hold_Cumulative_Max'] = analysis_df['Buy_Hold_Cumulative'].cummax()
    analysis_df['Buy_Hold_Drawdown'] = analysis_df['Buy_Hold_Cumulative'] / analysis_df['Buy_Hold_Cumulative_Max'] - 1
    buy_hold_max_drawdown = analysis_df['Buy_Hold_Drawdown'].min()

    performance_metrics = {
        'Strategy Annual Return': strategy_annual_return,
        'Buy and Hold Annual Return': buy_hold_annual_return,
        'Strategy Sharpe': strategy_sharpe,
        'Buy and Hold Sharpe': buy_hold_sharpe,
        'Strategy Maximum Drawdown': strategy_max_drawdown,
        'Buy and Hold Maximum Drawdown': buy_hold_max_drawdown
    }
    return performance_metrics, (last_trade_signal_type, last_signal_date, last_signal_strength)


# --------------------------------------------------------------------------
# Analyze Momentum
# --------------------------------------------------------------------------
def analyze_momentum(decompose_result, original_series, ticker='', description=''):
    """
    Analyzes momentum based on the trend and residual components,
    plus backtests the strategy.

    Returns:
      - trend_status
      - residual_status
      - performance_metrics
      - last_trade_signal_details
    """
    trend = decompose_result.trend
    residual = decompose_result.resid

    # Trend slope
    slope_series = calculate_trend_slope_vectorized(trend, window=252)
    # Residual z-score
    z_score = calculate_residual_zscore(residual, window=252).rename('Residual_Z_Score')

    # Combine
    analysis_df = pd.concat([
        original_series.rename('Adj_Close'),
        trend.rename('Trend'),
        slope_series.rename('Trend_Slope'),
        residual.rename('Residual'),
        z_score
    ], axis=1)

    # Plot Trend Slope
    plt.figure(figsize=(14, 6))
    plt.plot(analysis_df.index, analysis_df['Trend_Slope'], color='purple', label='Trend Slope (1Yr Rolling)')
    plt.title(f'Trend Slope - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Slope')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{ticker}_trend_slope.png"), dpi=150)
    plt.close()

    # Plot Residual Z-Score
    plt.figure(figsize=(14, 6))
    plt.plot(analysis_df.index, analysis_df['Residual_Z_Score'], color='brown', label='Residual Z-Score (1Yr Rolling)')
    plt.axhline(2, color='red', linestyle='--', label='±2')
    plt.axhline(-2, color='green', linestyle='--')
    plt.axhline(3, color='darkred', linestyle='--', label='±3')
    plt.axhline(-3, color='darkgreen', linestyle='--')
    plt.axhline(4, color='maroon', linestyle='--', label='±4')
    plt.axhline(-4, color='darkolivegreen', linestyle='--')
    plt.title(f'Residual Z-Score - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Z-Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{ticker}_residual_zscore.png"), dpi=150)
    plt.close()

    # Interpretation
    trend_status = "Insufficient Data"
    residual_status = "Insufficient Data"
    if not slope_series.dropna().empty and not z_score.dropna().empty:
        latest_slope = slope_series.dropna().iloc[-1]
        latest_z = z_score.dropna().iloc[-1]

        # Trend Status
        if latest_slope > 0:
            trend_status = "Upward Momentum"
        elif latest_slope < 0:
            trend_status = "Downward Momentum"
        else:
            trend_status = "No Clear Momentum"

        # Residual Status with multi-level thresholds
        if latest_z > 4:
            residual_status = "Highly Overbought Condition"
        elif latest_z > 3:
            residual_status = "Moderately Overbought Condition"
        elif latest_z > 2:
            residual_status = "Slightly Overbought Condition"
        elif latest_z < -4:
            residual_status = "Highly Oversold Condition"
        elif latest_z < -3:
            residual_status = "Moderately Oversold Condition"
        elif latest_z < -2:
            residual_status = "Slightly Oversold Condition"
        else:
            residual_status = "Residuals Within Normal Range"

    # Backtest
    performance_metrics, last_trade_signal_details = backtest_strategy(analysis_df, ticker, description)

    return trend_status, residual_status, performance_metrics, last_trade_signal_details


# --------------------------------------------------------------------------
# Process a Single Ticker
# --------------------------------------------------------------------------
def process_ticker(ticker, description, adj_close_data):
    """
    Runs the entire pipeline for a single ticker:
      - Preprocess
      - Log transform
      - Decompose
      - Analyze momentum
      - Backtest
    """
    result = {
        'Ticker': ticker,
        'Description': description,
        'Trend Status': 'N/A',
        'Residual Status': 'N/A',
        'Strategy Annual Return': np.nan,
        'Buy and Hold Annual Return': np.nan,
        'Strategy Sharpe': np.nan,
        'Buy and Hold Sharpe': np.nan,
        'Strategy Maximum Drawdown': np.nan,
        'Buy and Hold Maximum Drawdown': np.nan,
        'Last Trade Signal': 'N/A',
        'Last Signal Date': 'N/A',
        'Last Signal Strength': 'N/A'
    }

    if ticker not in adj_close_data.columns:
        return result

    ticker_series = adj_close_data[ticker].dropna()
    if ticker_series.empty:
        return result

    # 1) Preprocess
    preprocessed_series = preprocess_ticker_data(ticker_series)
    # 2) Log transform
    log_series = log_transform(preprocessed_series)

    # 3) Decompose
    try:
        additive_result = decompose_time_series(log_series, model='additive', period=252)
    except Exception:
        return result  # not enough data or decomposition error

    # 4) Plot Decomposition
    plot_decomposition(
        additive_result,
        title='Additive Decomposition of Log-Transformed Data',
        ticker=ticker
    )

    # 5) Analyze Momentum & Backtest
    trend_status, residual_status, performance_metrics, last_trade_signal_details = \
        analyze_momentum(additive_result, log_series, ticker, description)

    # Update result
    result.update({
        'Trend Status': trend_status,
        'Residual Status': residual_status,
        'Strategy Annual Return': performance_metrics['Strategy Annual Return'],
        'Buy and Hold Annual Return': performance_metrics['Buy and Hold Annual Return'],
        'Strategy Sharpe': performance_metrics['Strategy Sharpe'],
        'Buy and Hold Sharpe': performance_metrics['Buy and Hold Sharpe'],
        'Strategy Maximum Drawdown': performance_metrics['Strategy Maximum Drawdown'],
        'Buy and Hold Maximum Drawdown': performance_metrics['Buy and Hold Maximum Drawdown']
    })

    last_trade_signal_type, last_signal_date, last_signal_strength = last_trade_signal_details
    if last_trade_signal_type in ['Buy', 'Sell']:
        result['Last Trade Signal'] = last_trade_signal_type
        result['Last Signal Date'] = last_signal_date
        result['Last Signal Strength'] = last_signal_strength

    return result


# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------
def main():
    if not TICKERS:
        print("No tickers found. Exiting.")
        return

    # 1) Fetch data from Binance
    print("\nFetching data from Binance...")
    adj_close_data = fetch_data_from_binance(
        tickers=TICKERS,
        interval=KLINE_INTERVAL
    )

    # Filter only those tickers that actually returned data
    common_tickers = [t for t in TICKERS if t in adj_close_data.columns]
    if not common_tickers:
        print("No matching tickers found in Binance data. Exiting.")
        return

    adj_close_data = adj_close_data[common_tickers]

    # 2) Parallel processing
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing.")
    results = Parallel(n_jobs=num_cores)(
        delayed(process_ticker)(
            ticker,
            TICKER_DESCRIPTIONS.get(ticker, 'No Description Available'),
            adj_close_data
        )
        for ticker in common_tickers
    )

    # 3) Prepare and save results
    if results:
        results_df = pd.DataFrame(results)
        columns_order = [
            'Ticker',
            'Description',
            'Trend Status',
            'Residual Status',
            'Strategy Annual Return',
            'Buy and Hold Annual Return',
            'Strategy Sharpe',
            'Buy and Hold Sharpe',
            'Strategy Maximum Drawdown',
            'Buy and Hold Maximum Drawdown',
            'Last Trade Signal',
            'Last Signal Date',
            'Last Signal Strength'
        ]
        results_df = results_df[columns_order]
        results_df.to_excel(OUTPUT_FILE_PATH, index=False)
        print(f"\nAnalysis results saved to '{OUTPUT_FILE_PATH}'.")
    else:
        print("No results generated. Exiting.")

    print("\nAll tickers processed successfully.")


# Run Main
if __name__ == "__main__":
    main()
