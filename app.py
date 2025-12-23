import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ========= パス設定 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "selected_advanced_vwap_indicators_model.txt")
STOCK_MASTER_PATH = os.path.join(BASE_DIR, "stock_all.xlsx")  # ← xlsx に修正！

# ========= グローバルレート制限用 =========
_request_lock = threading.Lock()
_last_request_time = 0
_request_count = 0


def rate_limited_sleep():
    """yfinance API呼び出し前に必ず呼ぶ"""
    global _last_request_time, _request_count
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_time
        
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        
        _last_request_time = time.time()
        _request_count += 1
        
        if _request_count % 50 == 0:
            time.sleep(5)


# ========= 194特徴量 Predictor =========
class HighSpeedComplete194Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.selected_194_features = []
        self._init_features()

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
        df["CCI"] = ((df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std() + 1e-8)).fillna(0)
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
            ("True_Range_Ratio", "ADX_neg"), ("SMA_200", "Volume_SMA_20"),
            ("OBV", "SMA_200"), ("AD_Line", "MACD_signal"),
            ("OBV", "MACD_signal"), ("AD_Line", "OBV"),
            ("True_Range_Ratio", "BB_width"), ("AD_Line", "Volume_SMA_20"),
        ]
        for f1, f2 in combos:
            if f1 in df.columns and f2 in df.columns:
                df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
                df[f"{f1}_div_{f2}"] = df[f1] / (df[f2].abs() + 1e-8)
                df[f"{f1}_minus_{f2}"] = df[f1] - df[f2]

        polys = [
            ("SMA_200", "SMA_100"), ("OBV", "MACD_signal"),
            ("AD_Line", "OBV"), ("True_Range_Ratio", "OBV"),
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
        rate_limited_sleep()

        end = datetime.now()
        start = end - timedelta(days=300)
        
        try:
            df = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        except Exception:
            return None

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
            "rsi_14": float(df["RSI_14"].iloc[-1]) if "RSI_14" in df.columns else None,
            "vwap_20d": float(df["VWAP_20d"].iloc[-1]) if "VWAP_20d" in df.columns else None,
            "adx": float(df["ADX"].iloc[-1]) if "ADX" in df.columns else None,
            "macd": float(df["MACD"].iloc[-1]) if "MACD" in df.columns else None,
            "bb_percent": float(df["BB_percent"].iloc[-1]) if "BB_percent" in df.columns else None,
        }


@st.cache_resource
def get_predictor():
    pred = HighSpeedComplete194Predictor(MODEL_PATH)
    pred.load_model()
    return pred


@st.cache_data
def load_stock_master():
    df = pd.read_excel(STOCK_MASTER_PATH)
    df = df.dropna(subset=["コード"]).copy()
    df["コード"] = df["コード"].astype(str).str.zfill(4)
    df["銘柄名"] = df["銘柄名"].astype(str)
    return df


@st.cache_data(ttl=3600)
def fetch_price_history(code: str, days: int, interval: str = "1d"):
    """価格履歴取得（キャッシュ付き）"""
    end = datetime.now()
    start = end - timedelta(days=days)
    
    rate_limited_sleep()
    
    try:
        df = yf.Ticker(f"{code}.T").history(start=start, end=end, interval=interval)
    except Exception as e:
        st.error(f"{code} データ取得失敗: {e}")
        return None
    
    if df.empty:
        return None
    
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def create_chart_with_indicators(df, code, name, pred_data=None):
    """Flask版と同じ見た目のチャート生成"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.02,
        subplot_titles=("価格", "出来高", "RSI"),
    )

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1,
    )

    # 移動平均線（SMA20, SMA50）
    if len(df) >= 50:
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(color="orange", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50", line=dict(color="blue", width=1)), row=1, col=1)

    # 出来高
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, showlegend=False),
        row=2, col=1,
    )

    # RSI
    if len(df) >= 14:
        rsi = HighSpeedComplete194Predictor.calculate_rsi_fast(df["Close"], 14)
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="purple", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", opacity=0.5, row=3, col=1)

    fig.update_layout(
        title=dict(
            text=f"【{code}】{name}",
            font=dict(size=20, color="#333"),
            x=0.5,
            xanchor="center",
        ),
        height=700,
        margin=dict(l=50, r=50, t=80, b=40),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
    
    return fig


# ========= Streamlit UI（Flask版と同じ見た目） =========
def main():
    st.set_page_config(
        page_title="194特徴量完全版AI株価予測",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # カスタムCSS（Flask版風）
    st.markdown("""
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card h2 {
            font-size: 3rem;
            margin: 0;
            font-weight: bold;
        }
        .metric-card p {
            font-size: 1rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        .indicator-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f77b4;
        }
        .indicator-card h4 {
            margin: 0 0 0.5rem 0;
            color: #333;
            font-size: 0.9rem;
        }
        .indicator-card .value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

    # タイトル（Flask版と同じ）
    st.markdown('<div class="main-title">📊 194特徴量 完全版AI株価予測チャート</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">LightGBM × VWAP × 194種テクニカル指標</div>', unsafe_allow_html=True)

    df_master = load_stock_master()
    predictor = get_predictor()

    # サイドバー
    with st.sidebar:
        st.header("🔍 銘柄検索")
        search = st.text_input("銘柄名 or コード")
        
        df_show = df_master
        if search:
            s = search.strip()
            df_show = df_master[
                df_master["コード"].str.contains(s, na=False)
                | df_master["銘柄名"].str.contains(s, na=False)
            ]

        if not df_show.empty:
            codes = df_show["コード"].tolist()
            names = df_show["銘柄名"].tolist()
            options = [f"{c} - {n}" for c, n in zip(codes, names)]
            
            selected = st.selectbox("銘柄を選択", options, index=0)
            selected_code = selected.split(" - ")[0]
        else:
            st.warning("該当銘柄なし")
            selected_code = None

        st.markdown("---")
        st.subheader("⚙️ 表示設定")
        period_days = st.slider("表示期間（日）", 30, 730, 365, step=30)
        
        st.markdown("---")
        analyze_btn = st.button("🚀 AI分析実行", use_container_width=True, type="primary")

    if selected_code is None:
        st.info("左のサイドバーで銘柄を選択してください")
        return

    name = df_master.loc[df_master["コード"] == selected_code, "銘柄名"].iloc[0]

    # AI分析実行
    pred_data = None
    if analyze_btn:
        with st.spinner(f"{selected_code} {name} を分析中..."):
            pred_data = predictor.get_fast_prediction(selected_code)

    # 予測結果を大きく表示（Flask版風）
    if pred_data:
        st.markdown("### 🤖 AI予測結果")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            pred_score = pred_data["prediction"]
            score_percent = pred_score * 100
            
            if pred_score >= 0.7:
                color_grad = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
                signal = "🟢 強気"
            elif pred_score >= 0.5:
                color_grad = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                signal = "🟡 中立"
            else:
                color_grad = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
                signal = "🔴 弱気"
            
            st.markdown(f"""
            <div style="background: {color_grad}; padding: 2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
                <h2 style="margin: 0; font-size: 3.5rem; font-weight: bold;">{score_percent:.1f}%</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.95;">{signal}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="indicator-card">
                <h4>最新株価</h4>
                <div class="value">¥{pred_data['latest_price']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="indicator-card">
                <h4>データ日付</h4>
                <div class="value" style="font-size: 1.2rem;">{pred_data['latest_date'].strftime('%m/%d')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # テクニカル指標カード
        st.markdown("### 📈 主要テクニカル指標")
        col1, col2, col3, col4 = st.columns(4)
        
        indicators = [
            ("RSI(9)", pred_data.get("rsi_9"), col1),
            ("RSI(14)", pred_data.get("rsi_14"), col2),
            ("ADX", pred_data.get("adx"), col3),
            ("BB %", pred_data.get("bb_percent", 0) * 100 if pred_data.get("bb_percent") else None, col4),
        ]
        
        for label, value, col in indicators:
            with col:
                if value is not None:
                    st.markdown(f"""
                    <div class="indicator-card">
                        <h4>{label}</h4>
                        <div class="value">{value:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # チャート表示
    st.markdown("---")
    st.markdown("### 📊 価格チャート")
    
    df_price = fetch_price_history(selected_code, period_days)
    
    if df_price is not None and not df_price.empty:
        fig = create_chart_with_indicators(df_price, selected_code, name, pred_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("チャートデータを取得できませんでした")


if __name__ == "__main__":
    main()
