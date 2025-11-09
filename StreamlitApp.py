import streamlit as st, sys, traceback, os
st.set_page_config(page_title="Volume Profile", layout="wide")
st.write("✅ App booted to top of script")
st.sidebar.write("✅ Sidebar is alive")

# If anything fails later, show the traceback in the UI
def show_ex():
    st.error("Unhandled exception:")
    st.code("".join(traceback.format_exception(*sys.exc_info())))

# Quick env info (helps spot wrong interpreter / paths)
with st.expander("Debug info"):
    st.write({"python": sys.executable, "cwd": os.getcwd()})
    


import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
class VolumeProfileAnalyzer:
    def __init__(self, exchange_name='binance', symbol='BTC/USDT', timeframe='1h'):
        self.exchange = getattr(ccxt, exchange_name)()
        self.symbol = symbol
        self.timeframe = timeframe

    def fetch_data(self, start_date, end_date):
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        ohlcv = self.exchange.fetch_ohlcv(
            symbol=self.symbol,
            timeframe=self.timeframe,
            since=start_timestamp,
            limit=1000
        )

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def calculate_volume_profile(self, df, num_bins, value_area_pct=0.7, method='time', window_size=None):
        if method == 'fixed' and window_size:
            df = df.tail(window_size)

        price_range = np.linspace(df['low'].min(), df['high'].max(), num_bins)
        volumes = np.zeros(num_bins - 1)

        for idx, row in df.iterrows():
            price_position = np.searchsorted(price_range, [row['low'], row['high']])
            volume_per_bar = row['volume'] / (price_position[1] - price_position[0])
            volumes[price_position[0]:price_position[1]] += volume_per_bar

        # Calculate POC and Value Area
        total_volume = np.sum(volumes)
        poc_idx = np.argmax(volumes)
        poc_price = (price_range[poc_idx] + price_range[poc_idx + 1]) / 2

        # Calculate Value Area
        value_area_volume = total_volume * value_area_pct
        cumulative_volume = 0
        value_area_indices = [poc_idx]
        left_idx = right_idx = poc_idx

        while cumulative_volume < value_area_volume and (left_idx > 0 or right_idx < len(volumes) - 1):
            left_vol = volumes[left_idx - 1] if left_idx > 0 else 0
            right_vol = volumes[right_idx + 1] if right_idx < len(volumes) - 1 else 0

            if left_vol > right_vol and left_idx > 0:
                left_idx -= 1
                cumulative_volume += left_vol
                value_area_indices.append(left_idx)
            elif right_idx < len(volumes) - 1:
                right_idx += 1
                cumulative_volume += right_vol
                value_area_indices.append(right_idx)

        vah = price_range[max(value_area_indices)]
        val = price_range[min(value_area_indices)]

        return {
            'price_levels': price_range,
            'volumes': volumes,
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'value_area_indices': value_area_indices
        }

def plot_volume_profile(data, profile_data, chart_title):
    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.7, 0.3],
                        shared_yaxes=True,
                        subplot_titles=('Price Action', 'Volume Profile'))

    # Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Volume Profile
    fig.add_trace(
        go.Bar(
            x=profile_data['volumes'],
            y=profile_data['price_levels'][:-1],
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(0,0,255,0.3)',
            width=(profile_data['price_levels'][1] - profile_data['price_levels'][0])
        ),
        row=1, col=2
    )

    # Add POC and Value Area lines
    for price, name, color in [
        (profile_data['poc'], 'POC', 'red'),
        (profile_data['val'], 'VAL', 'green'),
        (profile_data['vah'], 'VAH', 'green')
    ]:
        fig.add_hline(
            y=price,
            line_dash="dash",
            line_color=color,
            annotation_text=name,
            annotation_position="right"
        )

    fig.update_layout(
        height=800,
        title=chart_title,
        showlegend=True,
        hovermode='y'
    )

    return fig

def main():
    st.title('Volume Profile Analysis')

    # Sidebar Configuration
    st.sidebar.header('Settings')

    exchange = st.sidebar.selectbox(
        'Select Exchange',
        ['binance', 'kraken', 'coinbase']
    )

    symbol = st.sidebar.text_input(
        'Trading Pair',
        value='BTC/USDT'
    )

    timeframe = st.sidebar.selectbox(
        'Timeframe',
        ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        index=4
    )

    analysis_method = st.sidebar.radio(
        'Analysis Method',
        ['Time-based', 'Fixed Window']
    )

    if analysis_method == 'Fixed Window':
        window_size = st.sidebar.slider(
            'Window Size (candles)',
            min_value=50,
            max_value=1000,
            value=200
        )
    else:
        window_size = None

    # Volume Profile Parameters
    num_bins = st.sidebar.slider(
        'Number of Price Bins',
        min_value=50,
        max_value=200,
        value=100
    )

    value_area_pct = st.sidebar.slider(
        'Value Area Percentage',
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05
    )

    # Date Range Selection
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=30)

    start_date = st.sidebar.date_input('Start Date', value=start_date)
    end_date = st.sidebar.date_input('End Date', value=end_date)

    if st.sidebar.button('Calculate Volume Profile'):
        try:
            analyzer = VolumeProfileAnalyzer(exchange, symbol, timeframe)

            with st.spinner('Fetching data...'):
                data = analyzer.fetch_data(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

            with st.spinner('Calculating Volume Profile...'):
                profile_data = analyzer.calculate_volume_profile(
                    data,
                    num_bins,
                    value_area_pct,
                    'fixed' if analysis_method == 'Fixed Window' else 'time',
                    window_size
                )

            # Display Results
            st.subheader('Volume Profile Analysis Results')

            fig = plot_volume_profile(
                data,
                profile_data,
                f'Volume Profile - {symbol} ({timeframe})'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Point of Control', f'{profile_data["poc"]:.2f}')
            with col2:
                st.metric('Value Area High', f'{profile_data["vah"]:.2f}')
            with col3:
                st.metric('Value Area Low', f'{profile_data["val"]:.2f}')

            # Additional Statistics
            st.subheader('Profile Statistics')
            stats_df = pd.DataFrame({
                'Metric': ['Total Volume', 'Volume within VA', 'Price Range'],
                'Value': [
                    f'{np.sum(profile_data["volumes"]):.2f}',
                    f'{np.sum(profile_data["volumes"][profile_data["value_area_indices"]]):.2f}',
                    f'{data["high"].max() - data["low"].min():.2f}'
                ]
            })
            st.table(stats_df)

            # Download Option
            csv_data = pd.DataFrame({
                'Price Levels': profile_data['price_levels'][:-1],
                'Volume': profile_data['volumes']
            })

            csv = csv_data.to_csv().encode('utf-8')
            st.download_button(
                label="Download Profile Data",
                data=csv,
                file_name=f'volume_profile_{symbol}_{timeframe}.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f'An error occurred: {str(e)}')