# app.py
# PRIMA DI ESEGUIRE QUESTO SCRIPT, ASSICURATI DI AVER INSTALLATO LE DIPENDENZE!
# Apri il tuo terminale o prompt dei comandi e digita:
# pip install streamlit yfinance pandas matplotlib mplfinance tabulate numpy ta

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
# from tabulate import tabulate # Potrebbe non essere necessario con Streamlit
import numpy as np
import ta # Technical Analysis library
from datetime import datetime, timezone # Importa timezone per fromisoformat
import json

# --- IMPOSTAZIONE PAGINA (DEVE ESSERE IL PRIMO COMANDO STREAMLIT) ---
st.set_page_config(layout="wide", page_title="Bonvo Stock Analyzer Pro")

# --- CONFIGURAZIONI GLOBALI DELL'APP ( tramite Streamlit Sidebar) ---
st.sidebar.header("Configurazioni Indicatori")
SMA_SHORT_WINDOW = st.sidebar.slider("SMA Finestra Corta", 5, 50, 20, key="sma_short")
SMA_LONG_WINDOW = st.sidebar.slider("SMA Finestra Lunga", 20, 200, 50, key="sma_long")
RSI_WINDOW = st.sidebar.slider("RSI Finestra", 7, 30, 14, key="rsi_window")
MACD_FAST = st.sidebar.slider("MACD Veloce", 5, 20, 12, key="macd_fast")
MACD_SLOW = st.sidebar.slider("MACD Lento", 20, 50, 26, key="macd_slow")
MACD_SIGNAL = st.sidebar.slider("MACD Segnale", 5, 20, 9, key="macd_signal")
BBANDS_WINDOW = st.sidebar.slider("Bollinger Bands Finestra", 10, 50, 20, key="bb_window")
BBANDS_STD_DEV = st.sidebar.slider("Bollinger Bands Dev.Std.", 1.0, 3.0, 2.0, 0.1, key="bb_std")
ATR_WINDOW = st.sidebar.slider("ATR Finestra", 7, 30, 14, key="atr_window")

st.sidebar.header("Configurazioni Visualizzazione")
NEWS_ITEMS_TO_SHOW = st.sidebar.slider("Numero News da Mostrare", 3, 15, 7, key="news_items")
CHART_PERIOD = st.sidebar.selectbox("Periodo Dati Grafico", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=2, key="chart_period")
CHART_INTERVAL = st.sidebar.selectbox("Intervallo Dati Grafico", ["1d", "5d", "1wk", "1mo"], index=0, key="chart_interval")


# --- FUNZIONI DI ANALISI ---

@st.cache_resource(ttl=300)
def fetch_stock_data(ticker_symbol, period="1y", interval="1d"):
    """Fetches historical stock data and company info."""
    ticker = yf.Ticker(ticker_symbol)
    try:
        info = ticker.info
        if not info or 'longName' not in info or info.get('longName') is None:
            st.error(f"Could not retrieve detailed info for {ticker_symbol}. It might be delisted, a very new listing, or an invalid ticker.")
            return None, None, None
    except Exception as e:
        st.error(f"Error fetching company info for {ticker_symbol}: {e}")
        return None, None, None

    data = ticker.history(period=period, interval=interval)
    if data.empty:
        st.warning(f"Could not fetch historical price data for {ticker_symbol} for the selected period/interval.")
        return ticker, info, None

    data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True, errors='ignore')
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Historical data for {ticker_symbol} is missing one or more required columns: open, high, low, close, volume.")
        return ticker, info, None

    return ticker, info, data

def calculate_technical_indicators(data):
    """Calculates various technical indicators using the 'ta' library."""
    if data is None or data.empty:
        return None
    data_ti = data.copy()

    data_ti[f'SMA_{SMA_SHORT_WINDOW}'] = ta.trend.sma_indicator(data_ti['close'], window=SMA_SHORT_WINDOW, fillna=True)
    data_ti[f'SMA_{SMA_LONG_WINDOW}'] = ta.trend.sma_indicator(data_ti['close'], window=SMA_LONG_WINDOW, fillna=True)
    data_ti['RSI'] = ta.momentum.rsi(data_ti['close'], window=RSI_WINDOW, fillna=True)
    macd_indicator = ta.trend.MACD(data_ti['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL, fillna=True)
    data_ti['MACD'] = macd_indicator.macd()
    data_ti['MACD_Signal'] = macd_indicator.macd_signal()
    data_ti['MACD_Hist'] = macd_indicator.macd_diff()
    bb_indicator = ta.volatility.BollingerBands(data_ti['close'], window=BBANDS_WINDOW, window_dev=BBANDS_STD_DEV, fillna=True)
    data_ti['BB_High'] = bb_indicator.bollinger_hband()
    data_ti['BB_Low'] = bb_indicator.bollinger_lband()
    data_ti['BB_Mid'] = bb_indicator.bollinger_mavg()
    data_ti['ATR'] = ta.volatility.average_true_range(data_ti['high'], data_ti['low'], data_ti['close'], window=ATR_WINDOW, fillna=True)
    return data_ti


def display_company_overview(info):
    st.subheader("--- COMPANY OVERVIEW ---")
    if info is None:
        st.warning("Company information not available.")
        return
    overview_data = {
        "Company Name": info.get('longName', 'N/A'),
        "Ticker": info.get('symbol', 'N/A'),
        "Sector": info.get('sector', 'N/A'),
        "Industry": info.get('industry', 'N/A'),
        "Country": info.get('country', 'N/A'),
        "Website": info.get('website', 'N/A'),
    }
    cols = st.columns(2)
    for i, (key, value) in enumerate(overview_data.items()):
        cols[i % 2].markdown(f"**{key}:** {value}")

    if 'longBusinessSummary' in info and info['longBusinessSummary']:
        with st.expander("Business Summary"):
            st.write(info['longBusinessSummary'])
    else:
        st.info("Business summary not available.")

def display_financial_snapshot(info, data_hist):
    st.subheader("--- FINANCIAL SNAPSHOT ---")
    if info is None:
        st.warning("Financial snapshot information not available (company info missing).")
        return

    latest_close = data_hist['close'].iloc[-1] if data_hist is not None and not data_hist.empty else info.get('previousClose', 'N/A')
    latest_volume = data_hist['volume'].iloc[-1] if data_hist is not None and not data_hist.empty else info.get('volume', 0)

    metrics_list = [
        ("Current Price", f"${latest_close:,.2f}" if isinstance(latest_close, (int, float)) else 'N/A'),
        ("Previous Close", f"${info.get('previousClose', 0):,.2f}" if info.get('previousClose') else 'N/A'),
        ("Day's Range", f"${info.get('dayLow', 0):,.2f} - ${info.get('dayHigh', 0):,.2f}" if info.get('dayLow') and info.get('dayHigh') else 'N/A'),
        ("52 Week Range", f"${info.get('fiftyTwoWeekLow', 0):,.2f} - ${info.get('fiftyTwoWeekHigh', 0):,.2f}" if info.get('fiftyTwoWeekLow') and info.get('fiftyTwoWeekHigh') else 'N/A'),
        ("Volume", f"{latest_volume:,.0f}" if isinstance(latest_volume, (int, float)) else 'N/A'),
        ("Avg. Volume (10d)", f"{info.get('averageVolume10days', 0):,.0f}" if info.get('averageVolume10days') else 'N/A'),
        ("Market Cap", f"${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else 'N/A'),
        ("P/E Ratio (TTM)", f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A'),
        ("Forward P/E", f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else 'N/A'),
        ("EPS (TTM)", f"${info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else 'N/A'),
        ("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A'),
        ("Beta (5Y Monthly)", f"{info.get('beta', 0):.2f}" if info.get('beta') else 'N/A'),
    ]
    if 'earningsTimestamp' in info and info['earningsTimestamp']:
        try:
            metrics_list.append(("Next Earnings Date", datetime.fromtimestamp(info['earningsTimestamp']).strftime('%Y-%m-%d')))
        except:
             metrics_list.append(("Next Earnings Date", "Error formatting date"))
    else:
        metrics_list.append(("Next Earnings Date", "N/A"))

    half_len = (len(metrics_list) + 1) // 2
    col1_data = metrics_list[:half_len]
    col2_data = metrics_list[half_len:]

    cols = st.columns(2)
    with cols[0]:
        for metric, value in col1_data:
            st.markdown(f"**{metric}:** {value}")
    with cols[1]:
        for metric, value in col2_data:
            st.markdown(f"**{metric}:** {value}")

def display_technical_analysis(data):
    if data is None or data.empty:
        st.warning("No data available for technical analysis.")
        return

    st.subheader("--- TECHNICAL ANALYSIS ---")
    latest = data.iloc[-1]

    with st.expander("**Simple Moving Averages (SMAs)**", expanded=True):
        st.markdown(f"  - *What*: Average closing price over a specific period. Used to identify trend direction.")
        sma_short_val = latest.get(f'SMA_{SMA_SHORT_WINDOW}', float('nan'))
        sma_long_val = latest.get(f'SMA_{SMA_LONG_WINDOW}', float('nan'))
        st.write(f"  - {SMA_SHORT_WINDOW}-day SMA: {sma_short_val:.2f}")
        st.write(f"  - {SMA_LONG_WINDOW}-day SMA: {sma_long_val:.2f}")
        if not pd.isna(sma_short_val) and not pd.isna(sma_long_val):
            if sma_short_val > sma_long_val:
                st.success("  - Interpretation: Short-term SMA is above long-term SMA, generally a bullish signal (Golden Cross if recent).")
            else:
                st.error("  - Interpretation: Short-term SMA is below long-term SMA, generally a bearish signal (Death Cross if recent).")

    with st.expander("**Relative Strength Index (RSI)**", expanded=True):
        st.markdown(f"  - *What*: Momentum oscillator measuring speed and change of price movements (0-100).")
        rsi_val = latest.get('RSI', float('nan'))
        st.write(f"  - Current RSI ({RSI_WINDOW}-day): {rsi_val:.2f}")
        if not pd.isna(rsi_val):
            if rsi_val > 70:
                st.error("  - Interpretation: Overbought (RSI > 70). Potential for a price pullback.")
            elif rsi_val < 30:
                st.success("  - Interpretation: Oversold (RSI < 30). Potential for a price bounce.")
            else:
                st.info("  - Interpretation: Neutral (RSI between 30 and 70).")

    with st.expander("**Moving Average Convergence Divergence (MACD)**", expanded=True):
        st.markdown(f"  - *What*: Trend-following momentum indicator showing relationship between two EMAs.")
        macd_line_val = latest.get('MACD', float('nan'))
        signal_line_val = latest.get('MACD_Signal', float('nan'))
        hist_val = latest.get('MACD_Hist', float('nan'))
        st.write(f"  - MACD Line: {macd_line_val:.2f}")
        st.write(f"  - Signal Line: {signal_line_val:.2f}")
        st.write(f"  - Histogram: {hist_val:.2f}")
        if not pd.isna(macd_line_val) and not pd.isna(signal_line_val):
            if macd_line_val > signal_line_val and hist_val > 0:
                st.success("  - Interpretation: MACD line is above Signal line (bullish crossover / positive histogram). Suggests upward momentum.")
            elif macd_line_val < signal_line_val and hist_val < 0:
                st.error("  - Interpretation: MACD line is below Signal line (bearish crossover / negative histogram). Suggests downward momentum.")
            else:
                st.info("  - Interpretation: MACD and Signal lines are very close or crossing.")

    with st.expander("**Bollinger Bands (BBands)**", expanded=True):
        st.markdown(f"  - *What*: Volatility bands placed above and below a moving average. Price is high at upper band, low at lower band.")
        bb_high_val = latest.get('BB_High', float('nan'))
        bb_mid_val = latest.get('BB_Mid', float('nan'))
        bb_low_val = latest.get('BB_Low', float('nan'))
        st.write(f"  - Upper Band: {bb_high_val:.2f}")
        st.write(f"  - Middle Band (SMA {BBANDS_WINDOW}): {bb_mid_val:.2f}")
        st.write(f"  - Lower Band: {bb_low_val:.2f}")
        if not pd.isna(latest['close']) and not pd.isna(bb_high_val) and not pd.isna(bb_low_val) and not pd.isna(bb_mid_val):
            if latest['close'] > bb_high_val:
                st.warning("  - Interpretation: Price is above the Upper Band. Potentially overextended, could revert to mean.")
            elif latest['close'] < bb_low_val:
                st.warning("  - Interpretation: Price is below the Lower Band. Potentially oversold, could revert to mean.")
            else:
                st.info("  - Interpretation: Price is within the bands.")
            band_width = (bb_high_val - bb_low_val) / bb_mid_val * 100 if bb_mid_val != 0 else float('nan')
            st.write(f"  - Bandwidth: {band_width:.2f}%. Narrow bands may precede volatility, wide bands indicate it.")

    with st.expander("**Average True Range (ATR)**", expanded=True):
        st.markdown(f"  - *What*: Measures market volatility by decomposing the entire range of an asset price for that period.")
        atr_val = latest.get('ATR', float('nan'))
        st.write(f"  - Current ATR ({ATR_WINDOW}-day): {atr_val:.2f} (price units)")
        if not pd.isna(atr_val) and not pd.isna(latest['close']) and latest['close'] != 0 :
            atr_percent = (atr_val / latest['close']) * 100
            st.write(f"  - ATR as % of Price: {atr_percent:.2f}%")
            st.info(f"  - Interpretation: Higher ATR indicates higher volatility, lower ATR indicates lower volatility. Useful for setting stop-losses or profit targets.")

    with st.expander("**Volume Analysis**", expanded=True):
        avg_vol_20d = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else float('nan')
        latest_vol = latest.get('volume', float('nan'))
        st.write(f"  - Latest Volume: {latest_vol:,.0f}")
        if not pd.isna(avg_vol_20d):
            st.write(f"  - 20-day Avg Volume: {avg_vol_20d:,.0f}")
            if not pd.isna(latest_vol) and latest_vol > avg_vol_20d * 1.5:
                st.info("  - Interpretation: Volume is significantly higher than average, suggesting strong conviction behind the recent price move.")
            elif not pd.isna(latest_vol) and latest_vol < avg_vol_20d * 0.75:
                 st.info("  - Interpretation: Volume is significantly lower than average, suggesting weak conviction or consolidation.")
            else:
                st.info("  - Interpretation: Volume is around its recent average or data is insufficient.")
        else:
            st.info("  - Not enough data for 20-day average volume.")


def display_latest_news_and_catalysts(ticker_object, company_info):
    st.subheader("--- LATEST NEWS & POTENTIAL CATALYSTS ---")
    if ticker_object is None or company_info is None:
        st.warning("News information not available (ticker object or company info missing).")
        return

    st.markdown("""
    A 'catalyst' is an event or piece of news that can significantly move a stock's price.
    Look for news related to: Earnings reports, new product launches/updates, FDA approvals (for biotech/pharma),
    major contract wins, mergers & acquisitions (M&A), management changes, regulatory changes, industry trends,
    analyst upgrades/downgrades, or significant economic news impacting the company/sector.
    """)

    try:
        news_items = ticker_object.news
        st.markdown("**Recent News Headlines (Source: Yahoo Finance):**")
        if news_items and isinstance(news_items, list) and len(news_items) > 0:
            news_data_for_df = []
            processed_count = 0
            effectively_empty_articles_shown = 0

            for article_idx, article_wrapper in enumerate(news_items):
                if processed_count >= NEWS_ITEMS_TO_SHOW:
                    break

                if not isinstance(article_wrapper, dict) or 'content' not in article_wrapper or not isinstance(article_wrapper['content'], dict):
                    continue

                article_content = article_wrapper['content']
                title_raw = article_content.get('title')
                publisher_raw = article_content.get('provider', {}).get('displayName')
                link_raw = article_content.get('clickThroughUrl', {}).get('url') or article_content.get('canonicalUrl', {}).get('url')
                provider_publish_time_raw_str = article_content.get('pubDate')

                title_disp = title_raw if title_raw else 'N/A'
                publisher_disp = publisher_raw if publisher_raw else 'N/A'
                link_disp = link_raw if link_raw else 'N/A'

                publish_time_str = "N/A"
                if provider_publish_time_raw_str:
                    try:
                        dt_object = datetime.fromisoformat(provider_publish_time_raw_str.replace('Z', '+00:00'))
                        publish_time_str = dt_object.strftime('%Y-%m-%d %H:%M')
                    except (ValueError, TypeError):
                        publish_time_str = "Invalid Date"

                if title_disp == 'N/A' and publisher_disp == 'N/A' and link_disp == 'N/A' and publish_time_str == 'N/A':
                    effectively_empty_articles_shown +=1

                max_title_len = 100
                if isinstance(title_disp, str) and len(title_disp) > max_title_len:
                    title_disp = title_disp[:max_title_len-3] + "..."

                link_md = f"[Link]({link_disp})" if link_disp != 'N/A' and link_disp is not None else "N/A"
                news_data_for_df.append({'Date': publish_time_str, 'Publisher': publisher_disp, 'Title': title_disp, 'Link': link_md})
                processed_count += 1

            if news_data_for_df:
                news_df = pd.DataFrame(news_data_for_df)
                st.markdown(news_df.to_markdown(index=False), unsafe_allow_html=True)

                if effectively_empty_articles_shown == processed_count and processed_count > 0:
                    st.caption(f"Note: All {processed_count} news items displayed appeared to lack specific data (title, publisher, etc.) from the source despite items being present.")
                elif effectively_empty_articles_shown > 0 :
                     st.caption(f"Note: {effectively_empty_articles_shown} of the {processed_count} news items displayed appeared to lack specific data.")
            else:
                st.info(f"No news articles were processed for display for {company_info.get('symbol', 'this ticker')}.")

        elif isinstance(news_items, list) and len(news_items) == 0:
             st.info(f"No recent news headlines were returned by Yahoo Finance for {company_info.get('symbol', 'this ticker')} (the news list was empty).")
        else:
            st.warning(f"Could not retrieve news for {company_info.get('symbol', 'this ticker')}. The news feed might be unavailable or the data format was unexpected.")

    except Exception as e:
        st.error(f"An unexpected error occurred while fetching or processing news: {e}")

    st.markdown("""
    **Note:** The news above is provided as-is. Critical thinking is required to assess its potential impact.
    Always verify information from multiple reputable sources.
    """)


def plot_stock_data_enhanced(data, ticker_symbol, info):
    """Plots stock data with more indicators using mplfinance."""
    if info is None:
        st.warning("Cannot plot stock data: company info missing.")
        return
    min_data_needed = max(SMA_LONG_WINDOW, BBANDS_WINDOW, MACD_SLOW, RSI_WINDOW, ATR_WINDOW, 20)
    if data is None or data.empty or len(data) < min_data_needed:
        st.warning(f"Not enough data to plot all indicators (need at least {min_data_needed} data points, got {len(data) if data is not None else 0}). Try a longer period.")
        return

    st.subheader(f"Stock Chart for {info.get('longName', ticker_symbol)}")

    ap = []
    if 'BB_High' in data.columns and 'BB_Low' in data.columns and 'BB_Mid' in data.columns:
        ap.append(mpf.make_addplot(data['BB_High'], linestyle='dashdot', width=0.7, panel=0, color='lightgray', ylabel='Price ($)'))
        ap.append(mpf.make_addplot(data['BB_Low'], linestyle='dashdot', width=0.7, panel=0, color='lightgray'))
        ap.append(mpf.make_addplot(data['BB_Mid'], panel=0, color='gray', linestyle='dotted', width=0.7))

    if f'SMA_{SMA_SHORT_WINDOW}' in data.columns:
        ap.append(mpf.make_addplot(data[f'SMA_{SMA_SHORT_WINDOW}'], panel=0, color='blue', width=0.9)) # RIMOSSO legend
    if f'SMA_{SMA_LONG_WINDOW}' in data.columns:
        ap.append(mpf.make_addplot(data[f'SMA_{SMA_LONG_WINDOW}'], panel=0, color='orange', width=0.9)) # RIMOSSO legend

    if 'RSI' in data.columns:
        ap.append(mpf.make_addplot(data['RSI'], panel=2, color='purple', ylabel='RSI', secondary_y=False)) # RIMOSSO legend
        ap.append(mpf.make_addplot(pd.Series(70, index=data.index), panel=2, color='red', linestyle='--', secondary_y=False, width=0.7))
        ap.append(mpf.make_addplot(pd.Series(30, index=data.index), panel=2, color='green', linestyle='--', secondary_y=False, width=0.7))

    if 'MACD' in data.columns and 'MACD_Signal' in data.columns and 'MACD_Hist' in data.columns:
        ap.append(mpf.make_addplot(data['MACD'], panel=3, color='green', ylabel='MACD', secondary_y=False)) # RIMOSSO legend
        ap.append(mpf.make_addplot(data['MACD_Signal'], panel=3, color='red', secondary_y=False)) # RIMOSSO legend
        macd_hist_positive = data['MACD_Hist'].copy(); macd_hist_positive[macd_hist_positive < 0] = 0
        macd_hist_negative = data['MACD_Hist'].copy(); macd_hist_negative[macd_hist_negative > 0] = 0
        ap.append(mpf.make_addplot(macd_hist_positive, type='bar', panel=3, color='darkgreen', alpha=0.7, secondary_y='auto')) # RIMOSSO legend
        ap.append(mpf.make_addplot(macd_hist_negative, type='bar', panel=3, color='darkred', alpha=0.7, secondary_y='auto')) # RIMOSSO legend

    fig, axes = mpf.plot(data,
                         type='candle',
                         style='yahoo',
                         title=f"\n{info.get('longName', ticker_symbol)} ({ticker_symbol}) Stock Analysis",
                         volume=True,
                         ylabel_lower='Volume',
                         addplot=ap,
                         panel_ratios=(6,1,2,2),
                         figratio=(12,7),
                         mav=(),
                         main_panel = 0,
                         volume_panel = 1,
                         returnfig=True,
                         show_nontrading=False,
                         tight_layout=True
                        )
    st.pyplot(fig)
    plt.close(fig)
    
    st.caption(f"""
        Legenda Grafico:
        - Candele: Prezzo (Apertura, Massimo, Minimo, Chiusura)
        - Linea Blu (pannello prezzo): SMA {SMA_SHORT_WINDOW} giorni
        - Linea Arancione (pannello prezzo): SMA {SMA_LONG_WINDOW} giorni
        - Grafico Viola (pannello 2): RSI ({RSI_WINDOW} giorni)
        - Linee Verde/Rossa (pannello 3): MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) / Segnale MACD
        - Barre Verde/Rossa scuro (pannello 3): Istogramma MACD
    """)


def generate_concluding_thoughts(data, info):
    if info is None:
        st.warning("Cannot generate concluding thoughts: company info missing.")
        return
    if data is None or data.empty:
        st.info("No data available for concluding thoughts.")
        return

    st.subheader("--- CONCLUDING THOUGHTS & CONSIDERATIONS (NOT FINANCIAL ADVICE) ---")
    st.markdown("""
    The following observations are based on the technical indicators and a brief review of recent news headlines.
    A holistic view requires deeper fundamental analysis and news interpretation.
    **This is NOT financial advice.**
    """)

    latest = data.iloc[-1]
    bullish_points = []
    bearish_points = []
    neutral_points = []

    price = latest.get('close', float('nan'))
    sma_short = latest.get(f'SMA_{SMA_SHORT_WINDOW}', float('nan'))
    sma_long = latest.get(f'SMA_{SMA_LONG_WINDOW}', float('nan'))
    if not pd.isna(price) and not pd.isna(sma_short) and not pd.isna(sma_long):
        if price > sma_short and sma_short > sma_long:
            bullish_points.append(f"Price (${price:.2f}) is above SMA{SMA_SHORT_WINDOW} (${sma_short:.2f}), which is above SMA{SMA_LONG_WINDOW} (${sma_long:.2f}) (Strong Bullish Trend Structure).")
        elif price < sma_short and sma_short < sma_long:
            bearish_points.append(f"Price (${price:.2f}) is below SMA{SMA_SHORT_WINDOW} (${sma_short:.2f}), which is below SMA{SMA_LONG_WINDOW} (${sma_long:.2f}) (Strong Bearish Trend Structure).")
        elif price > sma_short and price > sma_long :
             bullish_points.append(f"Price (${price:.2f}) is above both SMA{SMA_SHORT_WINDOW} (${sma_short:.2f}) and SMA{SMA_LONG_WINDOW} (${sma_long:.2f}).")
        elif price < sma_short and price < sma_long :
             bearish_points.append(f"Price (${price:.2f}) is below both SMA{SMA_SHORT_WINDOW} (${sma_short:.2f}) and SMA{SMA_LONG_WINDOW} (${sma_long:.2f}).")
        else:
            neutral_points.append(f"Price (${price:.2f}) is mixed relative to its SMAs (SMA{SMA_SHORT_WINDOW}: ${sma_short:.2f}, SMA{SMA_LONG_WINDOW}: ${sma_long:.2f}).")

    rsi = latest.get('RSI', float('nan'))
    if not pd.isna(rsi):
        if rsi > 70:
            bearish_points.append(f"RSI ({rsi:.2f}) is in overbought territory (>70), suggesting potential for a pullback.")
        elif rsi < 30:
            bullish_points.append(f"RSI ({rsi:.2f}) is in oversold territory (<30), suggesting potential for a bounce.")
        else:
            neutral_points.append(f"RSI ({rsi:.2f}) is neutral (30-70).")

    macd_val = latest.get('MACD', float('nan'))
    signal_val = latest.get('MACD_Signal', float('nan'))
    hist_val = latest.get('MACD_Hist', float('nan'))
    if not pd.isna(macd_val) and not pd.isna(signal_val) and not pd.isna(hist_val):
        if macd_val > signal_val and hist_val > 0:
            bullish_points.append(f"MACD ({macd_val:.2f}) is above its signal line ({signal_val:.2f}) with positive histogram ({hist_val:.2f}) (Bullish Momentum).")
        elif macd_val < signal_val and hist_val < 0:
            bearish_points.append(f"MACD ({macd_val:.2f}) is below its signal line ({signal_val:.2f}) with negative histogram ({hist_val:.2f}) (Bearish Momentum).")
        else:
            neutral_points.append("MACD is neutral or crossing its signal line.")

    bb_high = latest.get('BB_High', float('nan'))
    bb_low = latest.get('BB_Low', float('nan'))
    if not pd.isna(price) and not pd.isna(bb_high) and not pd.isna(bb_low):
        if price > bb_high:
            bearish_points.append(f"Price (${price:.2f}) is above the upper Bollinger Band (${bb_high:.2f}), suggesting it might be overextended short-term.")
        elif price < bb_low:
            bullish_points.append(f"Price (${price:.2f}) is below the lower Bollinger Band (${bb_low:.2f}), suggesting it might be oversold short-term.")

    latest_vol = latest.get('volume', float('nan'))
    avg_vol_20d = data['volume'].rolling(window=20).mean().iloc[-1] if len(data) >=20 else float('nan')
    if not pd.isna(latest_vol) and not pd.isna(avg_vol_20d) and avg_vol_20d > 0:
        if latest_vol > avg_vol_20d * 1.5:
            neutral_points.append(f"Volume ({latest_vol:,.0f}) is significantly above its 20-day average ({avg_vol_20d:,.0f}), confirming current price action strength.")
        elif latest_vol < avg_vol_20d * 0.75:
            neutral_points.append(f"Volume ({latest_vol:,.0f}) is below its 20-day average ({avg_vol_20d:,.0f}), suggesting weaker conviction.")

    st.markdown("**Potential Bullish Signals Observed (from Technicals):**")
    if bullish_points:
        for point in bullish_points: st.success(f"  - {point}")
    else:
        st.info("  - None significant at this time based on current settings.")

    st.markdown("\n**Potential Bearish Signals Observed (from Technicals):**")
    if bearish_points:
        for point in bearish_points: st.error(f"  - {point}")
    else:
        st.info("  - None significant at this time based on current settings.")

    st.markdown("\n**Neutral/Mixed Signals or Items to Monitor (from Technicals):**")
    if neutral_points:
        for point in neutral_points: st.warning(f"  - {point}")
    else:
        st.info("  - None significant at this time based on current settings.")

    st.markdown("\n**Context from News & Catalysts:**")
    st.markdown(f"""
      - Review the 'LATEST NEWS & POTENTIAL CATALYSTS' section for {info.get('longName', 'the company')}.
      - Are there any recent announcements (earnings, products, partnerships, regulatory news, etc.) that could explain recent price action or act as future catalysts?
      - Is the sentiment in the news generally positive, negative, or mixed?
      - How might broader market news or sector-specific developments be affecting the stock?
    """)

    st.markdown("\n**Overall Recommendation Approach (Not a Specific Buy/Sell Call):**")
    st.markdown("""
      1. **Synthesize Technicals & News:** Combine the technical signals with your interpretation of the news.
         - Strong bullish technicals + positive catalyst news = Higher conviction for potential upside.
         - Strong bearish technicals + negative news = Higher conviction for potential downside.
         - Mixed signals or conflicting news require more caution and deeper investigation.
      2. **Consider Fundamentals (Beyond this tool's scope):** Analyze the company's financial health, valuation (P/E, P/S, etc.), competitive advantages, and growth prospects. This tool provides some snapshot metrics, but full fundamental analysis is a separate, crucial step.
      3. **Assess Risk Tolerance & Investment Horizon:** How much risk are you willing to take? Is this a short-term trade or a long-term investment? Answers will influence decision-making.
      4. **Market Conditions:** Is the overall market bullish, bearish, or sideways? This macro environment affects all stocks.
    """)

    st.markdown("\n**Example Thought Process (Illustrative - you must do your own):**")
    st.markdown(f"""
      - *If* technicals are mostly bullish (e.g., price above SMAs, RSI rising, MACD bullish) *AND* recent news includes a positive earnings surprise or a new product launch for {info.get('longName', 'the company')}, this *might suggest* a favorable outlook. Further research into the specifics of the news and the company's fundamentals would be essential.
      - *Conversely, if* technicals are bearish (e.g., death cross, RSI falling, MACD bearish) *AND* news highlights increased competition or a regulatory setback, this *might suggest* caution.
    """)

    st.markdown("\n**Disclaimer:**")
    st.markdown("""
      - This automated analysis is for educational and informational purposes ONLY. It is NOT financial advice.
      - Identifying true catalysts and their impact is complex and requires experience.
      - ALWAYS conduct your own thorough research (DYOR) and consult with a qualified financial advisor before making any investment decisions.
    """)


# --- STRUTTURA PRINCIPALE DELL'APP STREAMLIT ---
def run_app():
    st.title("📈 Bonvo Stock Analyzer Pro 📊")
    st.markdown("Un'applicazione per l'analisi tecnica e fondamentale di base dei titoli azionari.")

    ticker_symbol_input = st.text_input("Inserisci il simbolo del ticker (es. AAPL, MSFT, NVDA):", "AAPL", key="ticker_input").upper()

    if ticker_symbol_input:
        analyze_button = st.button(f"Analizza {ticker_symbol_input}", key="analyze_btn", type="primary")

        if analyze_button:
            with st.spinner(f"Sto analizzando {ticker_symbol_input}... Potrebbe richiedere qualche secondo."):
                ticker_obj, company_info, historical_data = fetch_stock_data(ticker_symbol_input, period=CHART_PERIOD, interval=CHART_INTERVAL)

                if ticker_obj is not None and company_info is not None and historical_data is not None and not historical_data.empty:
                    data_with_indicators = calculate_technical_indicators(historical_data)

                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Grafico",
                        "🏢 Info Azienda",
                        "📈 Analisi Tecnica",
                        "📰 News & Catalizzatori",
                        "💡 Considerazioni Finali"
                    ])

                    with tab1:
                        plot_stock_data_enhanced(data_with_indicators, ticker_symbol_input, company_info)

                    with tab2:
                        display_company_overview(company_info)
                        st.divider()
                        display_financial_snapshot(company_info, data_with_indicators if data_with_indicators is not None else historical_data)

                    with tab3:
                        display_technical_analysis(data_with_indicators)

                    with tab4:
                        display_latest_news_and_catalysts(ticker_obj, company_info)

                    with tab5:
                        generate_concluding_thoughts(data_with_indicators, company_info)

                    st.success(f"Analisi per {ticker_symbol_input} completata!")
                else:
                    st.error(f"Impossibile recuperare dati sufficienti o completi per {ticker_symbol_input} per eseguire l'analisi. Controlla il ticker o prova un periodo/intervallo diverso.")
    else:
        st.info("Inserisci un simbolo ticker per iniziare l'analisi.")

    st.sidebar.markdown("---")
    st.sidebar.info("Questa app è solo a scopo educativo e informativo. Non costituisce consulenza finanziaria.")
    st.sidebar.markdown("Creato con Streamlit & yfinance.")

if __name__ == "__main__":
    run_app()
