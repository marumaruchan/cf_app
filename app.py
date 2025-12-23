import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import streamlit as st


# ========= パス設定 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "selected_advanced_vwap_indicators_model.txt")
STOCK_MASTER_PATH = os.path.join(BASE_DIR, "stock_all.xls")


# ========= 194特徴量 Predictor（Flask 版をそのまま流用） =========
class HighSpeedComplete194Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.selected_194_features = []
        self._init_features()

        self.request_count = 0
        self.last_request_time = time.time()
        self.request_lock = threading.Lock()

    def _init_features(self):
        base_features = [
            'Price_Change_10d', 'RSI_9', 'CMF', 'ADX', 'SMA_100', 'High_Low_Ratio',
            'ATR', 'Stoch_D', 'Volume_SMA_10', 'Volatility_10d', 'Volatility_20d',
            'BB_width', 'MACD_histogram', 'ADX_pos', 'Price_Change_20d', 'Volume_RSI',
            'SMA_50', 'MACD', 'Stoch_K', 'Range_Position_20d', 'BB_upper', 'SMA_5',
            'Range_Position_10d', 'EMA_10', 'CCI', 'SMA_10', 'Days_Above_SMA_50',
            'Price_Volume_Trend', 'Range_Position_5d', 'Price_Change_5d', 'Volatility_5d',
            'RSI_21', 'PSAR', 'BB_Convergence_Indicator',
            'VWAP_20d_40d_Diff', 'VWAP_20d_40d_Slope_Difference', 'RSI_VWAP_20d_Sync_Indicator',
            'RSI_VWAP_40d_Sync_Indicator', 'Price_Position_in_VWAP_40d_Bands',
            'consecutive_touch_VWAP_40d_upper_1sigma', 'VWAP_20d_Divergence_Rate',
            'consecutive_touch_VWAP_20d_upper_1sigma', 'consecutive_touch_VWAP_20d_lower_1sigma',
            'consecutive_touch_VWAP_40d_lower_1sigma', 'Price_Position_in_VWAP_20d_Bands',
            'consecutive_touch_VWAP_20d_lower_2sigma', 'VWAP_20d_BB_Dual_Lower_Break',
            'consecutive_touch_VWAP_40d_lower_2sigma', 'touch_VWAP_40d_lower_2sigma',
            'VWAP_Lower_Touch_Composite', 'VWAP_20d_lower_2sigma', 'VWAP_40d_upper_2sigma',
            'VWAP_40d_lower_1sigma', 'VWAP_20d',
            'True_Range_Ratio_x_ADX_neg', 'True_Range_Ratio_x_BB_width', 'SMA_200_x_Volume_SMA_20',
            'OBV_x_SMA_200', 'True_Range_Ratio_x_SMA_200', 'AD_Line_x_MACD_signal',
            'AD_Line_x_SMA_200', 'MACD_signal_x_Volume_SMA_20', 'True_Range_Ratio_x_Volume_SMA_20',
            'AD_Line_x_Volume_SMA_20', 'OBV_x_MACD_signal', 'AD_Line_x_OBV', 'SMA_200_x_BB_width',
            'Volume_SMA_20_x_ADX_neg', 'SMA_200_x_MACD_signal', 'Volume_SMA_20_x_BB_width',
            'True_Range_Ratio_x_OBV', 'MACD_signal_x_BB_width', 'OBV_x_BB_width', 'Price_Volume_Sync',
            'AD_Line_x_ADX_neg', 'MACD_signal_x_ADX_neg', 'ADX_neg_x_BB_width', 'OBV_x_Volume_SMA_20',
            'SMA_200_x_ADX_neg', 'AD_Line_x_BB_width', 'True_Range_Ratio_x_MACD_signal',
            'poly_SMA_200_SMA_100', 'poly_OBV_MACD_signal', 'poly_True_Range_Ratio_OBV',
            'poly_AD_Line_SMA_100', 'poly_OBV_SMA_200', 'poly_AD_Line_OBV', 'poly_OBV_Volume_SMA_20',
            'poly_BB_width_SMA_100', 'poly_True_Range_Ratio_AD_Line', 'poly_AD_Line_BB_width',
            'poly_OBV_ADX', 'poly_MACD_signal_ADX', 'poly_Volume_SMA_20_Volume_SMA_10',
            'poly_AD_Line_Volatility_20d', 'poly_SMA_200_MACD_signal', 'poly_AD_Line_MACD_signal',
            'poly_True_Range_Ratio_SMA_200', 'poly_MACD_signal_SMA_100', 'poly_OBV_BB_width',
            'poly_True_Range_Ratio_Volatility_20d', 'poly_AD_Line_CMF', 'poly_BB_width_ADX',
            'poly_SMA_200_ADX', 'poly_True_Range_Ratio_MACD_signal', 'poly_True_Range_Ratio_SMA_100',
            'poly_Volatility_20d_SMA_100', 'poly_ADX_Volatility_20d', 'poly_AD_Line_ADX',
            'poly_True_Range_Ratio_ADX', 'poly_MACD_signal_Volume_SMA_10', 'poly_MACD_signal_Volatility_20d',
            'poly_MACD_signal_CMF', 'poly_SMA_200_Volume_SMA_20', 'poly_MACD_signal_BB_width',
            'poly_AD_Line_RSI_9', 'poly_Volume_SMA_20_Volatility_20d', 'poly_Volatility_20d_ADX_pos',
            'poly_ADX_neg_Volatility_20d', 'poly_ADX_CMF', 'poly_AD_Line_ADX_pos',
            'poly_SMA_200_Volatility_20d', 'poly_BB_width_CMF', 'poly_AD_Line_Volume_SMA_20',
            'poly_True_Range_Ratio_CMF', 'poly_OBV_Volatility_20d', 'poly_CMF_ADX_pos',
            'Market_Relative_Strength', 'AD_Line_div_Volume_SMA_20', 'OBV_div_Volume_SMA_20',
            'AD_Line_div_OBV', 'SMA_200_div_BB_width', 'AD_Line_div_SMA_200', 'AD_Line_minus_OBV',
            'SMA_200_div_Volume_SMA_20', 'Composite_Trend_Strength', 'MACD_signal_div_BB_width',
            'AD_Line_div_BB_width', 'AD_Line_div_MACD_signal', 'Multi_Period_RSI_Diff',
            'OBV_div_SMA_200', 'Volume_SMA_20_div_BB_width', 'OBV_div_ADX_neg',
            'OBV_minus_Volume_SMA_20', 'Volume_SMA_20_div_ADX_neg', 'OBV_div_MACD_signal'
        ]

        additional_features = [
            'SMA_200', 'Volume_SMA_20', 'RSI_14', 'MACD_signal', 'OBV', 'AD_Line',
            'True_Range_Ratio', 'BB_lower', 'BB_percent', 'Price_Change_1d', 'Price_Change_3d',
            'Volume_Change', 'EMA_20', 'Williams_R', 'Ultimate_Oscillator', 'ROC_10', 'ROC_20',
            'MFI', 'Plus_DI', 'Minus_DI', 'VWAP_40d', 'VWAP_20d_upper_1sigma', 'VWAP_20d_upper_2sigma',
            'VWAP_40d_upper_1sigma', 'VWAP_40d_lower_2sigma', 'touch_VWAP_20d_upper_1sigma',
            'touch_VWAP_20d_lower_1sigma', 'touch_VWAP_20d_lower_2sigma', 'touch_VWAP_40d_upper_1sigma',
            'touch_VWAP_40d_lower_1sigma', 'Vol_Adjusted_Momentum', 'RSI_Momentum_Normalized',
            'Volume_Weighted_Return', 'VWAP_Upper_Touch_Composite', 'Max_Consecutive_VWAP_Touch',
            'Total_Consecutive_VWAP_Touch'
        ]

        all_features = base_features + additional_features
        self.selected_194_features = all_features[:194]

    def load_model(self):
        self.model = lgb.Booster(model_file=self.model_path)

    def safe_api_call(self):
        with self.request_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < 0.02:
                time.sleep(0.05 - elapsed)
            self.last_request_time = time.time()
            self.request_count += 1
            if self.request_count % 100 == 0:
                time.sleep(2)

    @staticmethod
    def calculate_rsi_fast(price_series, period=14):
        delta = price_series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return (100 - (100 / (1 + rs))).fillna(50)

    def calculate_additional_indicators_fast(self, df):
        low_14 = df["Low"].rolling(14).min()
        high_14 = df["High"].rolling(14).max()
        df["Stoch_K"] = (100 * (df["Close"] - low_14) / (high_14 - low_14 + 1e-8)).fillna(50)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean().fillna(50)

        price_change = df["Close"].diff().fillna(0)
        volume_direction = np.sign(price_change)
        df["OBV"] = (df["Volume"] * volume_direction).cumsum()

        clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-8)
        df["AD_Line"] = (clv * df["Volume"]).cumsum().fillna(0)
        df["CMF"] = clv.rolling(20).mean().fillna(0)

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df["True_Range_Ratio"] = (tr / (df["Close"] + 1e-8)).fillna(0)
        df["ATR"] = tr.rolling(14).mean().fillna(0)

        high_diff = df["High"].diff()
        low_diff = df["Low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0).abs()
        plus_di = (100 * plus_dm.rolling(14).mean() / (tr.rolling(14).mean() + 1e-8)).fillna(0)
        minus_di = (100 * minus_dm.rolling(14).mean() / (tr.rolling(14).mean() + 1e-8)).fillna(0)
        df["ADX"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)).fillna(0)
        df["ADX_pos"] = plus_di
        df["ADX_neg"] = minus_di
        df["Plus_DI"] = plus_di
        df["Minus_DI"] = minus_di

        df["High_Low_Ratio"] = ((df["High"] - df["Low"]) / df["Close"]).fillna(0)
        df["Volume_RSI"] = self.calculate_rsi_fast(df["Volume"], 14)
        df["CCI"] = ((df["Close"] - df["Close"].rolling(20).mean()) /
                     (df["Close"].rolling(20).std() + 1e-8)).fillna(0)
        df["Williams_R"] = (-100 * (high_14 - df["Close"]) / (high_14 - low_14 + 1e-8)).fillna(-50)
        df["Ultimate_Oscillator"] = df["Close"].pct_change().rolling(14).mean().fillna(0) * 100
        df["ROC_10"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100).fillna(0)
        df["ROC_20"] = ((df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100).fillna(0)
        df["MFI"] = df["CMF"] * 100

        for period in [5, 10, 20]:
            low_min = df["Low"].rolling(period).min()
            high_max = df["High"].rolling(period).max()
            df[f"Range_Position_{period}d"] = ((df["Close"] - low_min) / (high_max - low_min + 1e-8)).fillna(0.5)

        df["Days_Above_SMA_50"] = (df["Close"] > df["SMA_50"]).rolling(20).sum().fillna(0)
        df["Price_Volume_Trend"] = ((df["Close"].diff() / df["Close"].shift(1)) * df["Volume"]).cumsum().fillna(0)
        df["PSAR"] = df["Close"].rolling(10).min().fillna(df["Close"])
        return df

    def calculate_vwap_features_fast(self, df):
        for period in [20, 40]:
            tp = (df["High"] + df["Low"] + df["Close"]) / 3
            pv = (tp * df["Volume"]).rolling(period).sum()
            vol_sum = df["Volume"].rolling(period).sum()
            df[f"VWAP_{period}d"] = (pv / (vol_sum + 1e-8)).fillna(df["Close"])

            diff = df["Close"] - df[f"VWAP_{period}d"]
            std = diff.rolling(period).std().fillna(df["Close"].rolling(period).std())
            df[f"VWAP_{period}d_upper_1sigma"] = df[f"VWAP_{period}d"] + std
            df[f"VWAP_{period}d_lower_1sigma"] = df[f"VWAP_{period}d"] - std
            df[f"VWAP_{period}d_upper_2sigma"] = df[f"VWAP_{period}d"] + 2 * std
            df[f"VWAP_{period}d_lower_2sigma"] = df[f"VWAP_{period}d"] - 2 * std

            df[f"touch_VWAP_{period}d_upper_1sigma"] = (df["High"] >= df[f"VWAP_{period}d_upper_1sigma"]).astype(int)
            df[f"touch_VWAP_{period}d_lower_1sigma"] = (df["Low"] <= df[f"VWAP_{period}d_lower_1sigma"]).astype(int)
            df[f"touch_VWAP_{period}d_lower_2sigma"] = (df["Low"] <= df[f"VWAP_{period}d_lower_2sigma"]).astype(int)

            for col in [
                f"touch_VWAP_{period}d_upper_1sigma",
                f"touch_VWAP_{period}d_lower_1sigma",
                f"touch_VWAP_{period}d_lower_2sigma",
            ]:
                seq = []
                ser = df[col].tail(50)
                cnt = 0
                for v in ser:
                    if v == 1:
                        cnt += 1
                    else:
                        cnt = 0
                    seq.append(cnt)
                full = [0] * (len(df) - len(seq)) + seq
                df[f"consecutive_{col}"] = full

        if "VWAP_20d" in df.columns and "VWAP_40d" in df.columns:
            df["VWAP_20d_40d_Diff"] = df["VWAP_20d"] - df["VWAP_40d"]
            df["VWAP_20d_Divergence_Rate"] = (df["Close"] - df["VWAP_20d"]) / (df["VWAP_20d"] + 1e-8)
            df["VWAP_20d_40d_Slope_Difference"] = df["VWAP_20d"].diff(5) - df["VWAP_40d"].diff(5)

            if "RSI_9" in df.columns:
                rsi_m = df["RSI_9"] - 50
                div20 = (df["Close"] - df["VWAP_20d"]) / (df["VWAP_20d"] + 1e-8)
                div40 = (df["Close"] - df["VWAP_40d"]) / (df["VWAP_40d"] + 1e-8)
                df["RSI_VWAP_20d_Sync_Indicator"] = rsi_m * div20 * 100
                df["RSI_VWAP_40d_Sync_Indicator"] = rsi_m * div40 * 100

            for period in [20, 40]:
                std_val = df[f"VWAP_{period}d_upper_1sigma"] - df[f"VWAP_{period}d"]
                pos = (df["Close"] - df[f"VWAP_{period}d"]) / (2 * std_val + 1e-8)
                df[f"Price_Position_in_VWAP_{period}d_Bands"] = np.clip(pos, -1, 1)

        touch_cols = [c for c in df.columns if "touch_VWAP" in c and "lower" in c]
        if touch_cols:
            df["VWAP_Lower_Touch_Composite"] = df[touch_cols].sum(axis=1)

        cons_cols = [c for c in df.columns if "consecutive_touch_VWAP" in c]
        if cons_cols:
            df["Max_Consecutive_VWAP_Touch"] = df[cons_cols].max(axis=1)
            df["Total_Consecutive_VWAP_Touch"] = df[cons_cols].sum(axis=1)

        if "BB_lower" in df.columns and "touch_VWAP_20d_lower_1sigma" in df.columns:
            df["VWAP_20d_BB_Dual_Lower_Break"] = (
                (df["touch_VWAP_20d_lower_1sigma"] == 1) & (df["Close"] < df["BB_lower"])
            ).astype(int)

        return df

    def generate_derived_features_fast(self, df):
        if "Price_Change_10d" in df.columns:
            df["Market_Relative_Strength"] = df["Price_Change_10d"]
        if "ADX" in df.columns:
            df["Composite_Trend_Strength"] = df["ADX"] / 100
        if "RSI_9" in df.columns and "RSI_14" in df.columns:
            df["Multi_Period_RSI_Diff"] = df["RSI_9"] - df["RSI_14"]
        if "BB_percent" in df.columns and "BB_width" in df.columns:
            df["BB_Convergence_Indicator"] = df["BB_percent"] / (df["BB_width"] + 1e-8)
        if "Price_Change_5d" in df.columns and "OBV" in df.columns:
            df["Price_Volume_Sync"] = df["Price_Change_5d"] * np.sign(df["OBV"])
        if "Price_Change_5d" in df.columns and "Volatility_10d" in df.columns:
            df["Vol_Adjusted_Momentum"] = df["Price_Change_5d"] / (df["Volatility_10d"] + 1e-8)
        if "RSI_9" in df.columns:
            df["RSI_Momentum_Normalized"] = (df["RSI_9"] - 50) / 50
        if "Price_Change_1d" in df.columns and "Volume_Change" in df.columns:
            df["Volume_Weighted_Return"] = df["Price_Change_1d"] * np.log(df["Volume_Change"].abs() + 1)

        combos = [
            ("True_Range_Ratio", "ADX_neg"),
            ("SMA_200", "Volume_SMA_20"),
            ("OBV", "SMA_200"),
            ("AD_Line", "MACD_signal"),
            ("OBV", "MACD_signal"),
            ("AD_Line", "OBV"),
            ("True_Range_Ratio", "BB_width"),
            ("AD_Line", "Volume_SMA_20"),
        ]
        for f1, f2 in combos:
            if f1 in df.columns and f2 in df.columns:
                df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
                df[f"{f1}_div_{f2}"] = df[f1] / (df[f2].abs() + 1e-8)
                df[f"{f1}_minus_{f2}"] = df[f1] - df[f2]

        polys = [
            ("SMA_200", "SMA_100"),
            ("OBV", "MACD_signal"),
            ("AD_Line", "OBV"),
            ("True_Range_Ratio", "OBV"),
            ("OBV", "Volume_SMA_20"),
        ]
        for f1, f2 in polys:
            if f1 in df.columns and f2 in df.columns:
                df[f"poly_{f1}_{f2}"] = df[f1] * df[f2]
        return df

    def calculate_all_features_fast(self, df):
        for period in [1, 3, 5, 10, 20]:
            df[f"Price_Change_{period}d"] = df["Close"].pct_change(period).fillna(0)
        df["Volume_Change"] = df["Volume"].pct_change().fillna(0)
        for period in [5, 10, 20]:
            df[f"Volatility_{period}d"] = df["Close"].pct_change().rolling(period).std().fillna(0)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"SMA_{period}"] = df["Close"].rolling(period).mean().fillna(df["Close"])
        df["EMA_10"] = df["Close"].ewm(span=10).mean().fillna(df["Close"])
        df["EMA_20"] = df["Close"].ewm(span=20).mean().fillna(df["Close"])
        for period in [10, 20]:
            df[f"Volume_SMA_{period}"] = df["Volume"].rolling(period).mean().fillna(df["Volume"])
        for period in [9, 14, 21]:
            df[f"RSI_{period}"] = self.calculate_rsi_fast(df["Close"], period)
        exp1 = df["Close"].ewm(span=12).mean()
        exp2 = df["Close"].ewm(span=26).mean()
        df["MACD"] = (exp1 - exp2).fillna(0)
        df["MACD_signal"] = df["MACD"].ewm(span=9).mean().fillna(0)
        df["MACD_histogram"] = (df["MACD"] - df["MACD_signal"]).fillna(0)
        sma_20 = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["BB_upper"] = (sma_20 + 2 * std).fillna(df["Close"])
        df["BB_lower"] = (sma_20 - 2 * std).fillna(df["Close"])
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]).fillna(0)
        df["BB_percent"] = ((df["Close"] - df["BB_lower"]) / (df["BB_width"] + 1e-8)).fillna(0.5)

        df = self.calculate_additional_indicators_fast(df)
        df = self.calculate_vwap_features_fast(df)
        df = self.generate_derived_features_fast(df)
        return df

    def get_fast_prediction(self, code: str):
        symbol = f"{code}.T"
        self.safe_api_call()

        end = datetime.now()
        start = end - timedelta(days=300)
        df = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        if len(df) < 100:
            return None

        df = df.tail(200).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        latest_date = df.index[-1]
        now = datetime.now()
        days_since = (now - latest_date).days
        if days_since > 7:
            return None

        df = self.calculate_all_features_fast(df)
        latest = df.iloc[-1]
        vec = []
        for f in self.selected_194_features:
            if f in df.columns and not pd.isna(latest[f]):
                vec.append(float(latest[f]))
            else:
                vec.append(0.0)
        vec = np.array(vec[:194], dtype=np.float64)
        if vec.shape[0] < 194:
            vec = np.concatenate([vec, np.zeros(194 - vec.shape[0])])
        vec = np.nan_to_num(vec, 0)
        vec = np.clip(vec, -1e6, 1e6)

        proba = float(self.model.predict([vec])[0])
        return {
            "code": code,
            "prediction": proba,
            "latest_price": float(df["Close"].iloc[-1]),
            "latest_date": latest_date,
            "days_old": int(days_since),
            "rsi_9": float(df["RSI_9"].iloc[-1]) if "RSI_9" in df.columns else None,
            "vwap_20d": float(df["VWAP_20d"].iloc[-1]) if "VWAP_20d" in df.columns else None,
            "adx": float(df["ADX"].iloc[-1]) if "ADX" in df.columns else None,
        }


@st.cache_resource
def get_predictor():
    pred = HighSpeedComplete194Predictor(MODEL_PATH)
    pred.load_model()
    return pred


@st.cache_data
def load_stock_master():
    # engine='openpyxl' を追加（.xls でも強制的に openpyxl で読む）
    df = pd.read_excel(STOCK_MASTER_PATH, engine='openpyxl')
    df = df.dropna(subset=["コード"]).copy()
    df["コード"] = df["コード"].astype(str).str.zfill(4)
    df["銘柄名"] = df["銘柄名"].astype(str)
    return df


@st.cache_data
def fetch_price_history(code: str, period: str, interval: str):
    if period == "3ヶ月":
        days = 90
    elif period == "6ヶ月":
        days = 180
    elif period == "1年":
        days = 365
    elif period == "2年":
        days = 730
    else:
        days = 365 * 5
    end = datetime.now()
    start = end - timedelta(days=days)
    yf_interval = {"日足": "1d", "週足": "1wk", "月足": "1mo"}[interval]
    df = yf.Ticker(f"{code}.T").history(start=start, end=end, interval=yf_interval)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ========= Streamlit UI =========
def main():
    st.set_page_config(
        page_title="194特徴量AIチャート",
        layout="wide",
    )

    st.title("📈 194特徴量AIチャート（Streamlit版）")

    df_master = load_stock_master()
    predictor = get_predictor()

    # --- 左サイドバー ---
    with st.sidebar:
        st.header("銘柄選択")

        search = st.text_input("銘柄名 / コードで検索")
        df_show = df_master
        if search:
            s = search.strip()
            df_show = df_master[
                df_master["コード"].str.contains(s)
                | df_master["銘柄名"].str.contains(s)
            ]

        codes = df_show["コード"].tolist()
        labels = [f"{c} | {n}" for c, n in zip(df_show["コード"], df_show["銘柄名"])]
        selected_codes = st.multiselect(
            "表示する銘柄（複数選択可）",
            options=codes,
            format_func=lambda c: f"{c} | {df_master.loc[df_master['コード']==c,'銘柄名'].iloc[0]}",
        )

        st.markdown("---")
        st.subheader("チャート設定")
        cols = st.radio("列数", [1, 2, 3, 4], index=1, horizontal=True)
        interval = st.selectbox("足", ["日足", "週足", "月足"], index=0)
        period = st.selectbox("期間", ["3ヶ月", "6ヶ月", "1年", "2年", "5年"], index=2)

        st.markdown("---")
        st.subheader("AI分析")
        run_ai = st.button("🤖 選択中銘柄をAI分析")

    st.caption(f"現在の選択銘柄数: {len(selected_codes)}")

    # --- AI分析 ---
    ai_results = None
    if run_ai and selected_codes:
        st.info("AI分析を実行中…少し待ってね")
        rows = []
        for code in selected_codes:
            with st.spinner(f"{code} を解析中…"):
                pred = predictor.get_fast_prediction(code)
            if pred is not None:
                name = df_master.loc[df_master["コード"] == code, "銘柄名"].iloc[0]
                rows.append(
                    {
                        "コード": code,
                        "銘柄名": name,
                        "予測確率": pred["prediction"],
                        "最新株価": pred["latest_price"],
                        "RSI_9": pred["rsi_9"],
                        "VWAP_20d": pred["vwap_20d"],
                        "ADX": pred["adx"],
                        "データ日付": pred["latest_date"].strftime("%Y-%m-%d"),
                        "何日前": pred["days_old"],
                    }
                )
        if rows:
            ai_results = pd.DataFrame(rows).sort_values("予測確率", ascending=False)
            st.subheader("AI分析結果")
            st.dataframe(
                ai_results.style.format(
                    {
                        "予測確率": "{:.3f}",
                        "最新株価": "{:.0f}",
                        "RSI_9": "{:.1f}",
                        "VWAP_20d": "{:.0f}",
                        "ADX": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

    # --- チャート表示 ---
    if not selected_codes:
        st.warning("左のサイドバーで銘柄を選択するとチャートが表示されるよ")
        return

    st.subheader("チャート表示")

    n = len(selected_codes)
    rows = (n + cols - 1) // cols
    idx = 0
    for _ in range(rows):
        cols_container = st.columns(cols)
        for c in range(cols):
            if idx >= n:
                break
            code = selected_codes[idx]
            name = df_master.loc[df_master["コード"] == code, "銘柄名"].iloc[0]

            with cols_container[c]:
                st.markdown(f"**{code} {name}**")
                df_price = fetch_price_history(code, period, interval)
                if df_price is None or df_price.empty:
                    st.info("データ取得できず")
                else:
                    # 出来高が見切れないように subplot で縦を確保
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.03,
                    )

                    fig.add_trace(
                        go.Candlestick(
                            x=df_price.index,
                            open=df_price["Open"],
                            high=df_price["High"],
                            low=df_price["Low"],
                            close=df_price["Close"],
                            name="Price",
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Bar(
                            x=df_price.index,
                            y=df_price["Volume"],
                            name="Volume",
                            marker_color="lightgray",
                        ),
                        row=2,
                        col=1,
                    )

                    fig.update_layout(
                        height=400,
                        margin=dict(l=40, r=10, t=20, b=40),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if ai_results is not None and code in ai_results["コード"].values:
                        row = ai_results[ai_results["コード"] == code].iloc[0]
                        st.caption(
                            f"AI予測: {row['予測確率']:.3f} / RSI9: {row['RSI_9']:.1f} / ADX: {row['ADX']:.1f}"
                        )
            idx += 1


if __name__ == "__main__":
    main()
