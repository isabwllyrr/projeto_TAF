from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sktime.split import ExpandingWindowSplitter

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional boosting engine
    XGBRegressor = None

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional deep learning engine
    torch = nn = None

try:
    from arch import arch_model
except ImportError:  # pragma: no cover - friendly message in the UI
    arch_model = None


TRADING_DAYS = 252
MARKET_PRESETS = {
    "Brasil": {
        "benchmark": "^BVSP",
        "assets": {
            "Petrobras PN": "PETR4.SA",
            "Vale ON": "VALE3.SA",
            "Itau Unibanco PN": "ITUB4.SA",
            "Banco do Brasil ON": "BBAS3.SA",
            "B3 ON": "B3SA3.SA",
            "Weg ON": "WEGE3.SA",
            "Ambev ON": "ABEV3.SA",
            "Magazine Luiza ON": "MGLU3.SA",
            "Eletrobras ON": "ELET3.SA",
            "Suzano ON": "SUZB3.SA",
        },
        "default": ["Petrobras PN", "Vale ON", "Itau Unibanco PN", "Banco do Brasil ON", "B3 ON", "Weg ON"],
    },
}
YF_CACHE_DIR = Path(".yfinance_cache")
YF_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE_DIR))


@dataclass
class RegressionResult:
    asset: str
    alpha_daily: float
    alpha_annual: float
    beta: float
    risk_premium_annual: float
    r2: float
    p_value_beta: float


@dataclass
class FamaFrenchResult:
    asset: str
    alpha_daily: float
    market_beta: float
    smb_beta: float
    hml_beta: float
    r2: float


@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple[str, ...], benchmark: str, start: str, end: str) -> pd.DataFrame:
    symbols = sorted(set(tickers + (benchmark,)))
    data = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].rename(columns={"Close": symbols[0]})

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


@st.cache_data(show_spinner=False)
def load_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).get_info()
        except Exception:
            info = {}

        if not info:
            continue

        rows.append(
            {
                "Ativo": ticker,
                "Setor": info.get("sector") or "-",
                "Valor de mercado": info.get("marketCap"),
                "P/L": info.get("trailingPE"),
                "P/VP": info.get("priceToBook"),
                "ROE": info.get("returnOnEquity"),
                "Margem liquida": info.get("profitMargins"),
                "Dividend yield": info.get("dividendYield"),
                "Divida/Patrimonio": info.get("debtToEquity"),
            }
        )

    return pd.DataFrame(rows)


def parse_tickers(raw: str) -> tuple[str, ...]:
    tickers = [item.strip().upper() for item in raw.replace(";", ",").split(",")]
    return tuple(dict.fromkeys(t for t in tickers if t))


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


def load_price_csv(uploaded_file: io.BytesIO | None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    prices = pd.read_csv(uploaded_file)
    prices.columns = [str(col).strip() for col in prices.columns]
    date_col = next((col for col in prices.columns if col.lower() in {"date", "data", "dt"}), None)

    if date_col is None:
        raise ValueError("O CSV de precos precisa ter uma coluna date, data ou dt.")

    prices[date_col] = pd.to_datetime(prices[date_col])
    prices = prices.set_index(date_col).sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    prices.columns = [str(col).strip().upper() for col in prices.columns]
    return prices


def capm(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, risk_free_annual: float) -> list[RegressionResult]:
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    market_excess = benchmark_returns - rf_daily
    results: list[RegressionResult] = []

    for asset in asset_returns.columns:
        frame = pd.concat(
            {
                "asset_excess": asset_returns[asset] - rf_daily,
                "market_excess": market_excess,
            },
            axis=1,
        ).dropna()

        if len(frame) < 30:
            continue

        model = sm.OLS(frame["asset_excess"], sm.add_constant(frame["market_excess"])).fit()
        beta = float(model.params["market_excess"])
        expected_market_premium = float(market_excess.mean() * TRADING_DAYS)
        results.append(
            RegressionResult(
                asset=asset,
                alpha_daily=float(model.params["const"]),
                alpha_annual=float(((1 + model.params["const"]) ** TRADING_DAYS) - 1),
                beta=beta,
                risk_premium_annual=beta * expected_market_premium,
                r2=float(model.rsquared),
                p_value_beta=float(model.pvalues["market_excess"]),
            )
        )

    return results


def load_factor_csv(uploaded_file: io.BytesIO | None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    factors = pd.read_csv(uploaded_file)
    factors.columns = [str(col).strip().lower() for col in factors.columns]
    date_col = next((col for col in factors.columns if col in {"date", "data", "dt"}), None)
    required = {"mkt-rf", "smb", "hml"}

    if date_col is None or not required.issubset(set(factors.columns)):
        raise ValueError("O CSV precisa ter as colunas date, mkt-rf, smb e hml.")

    factors[date_col] = pd.to_datetime(factors[date_col])
    factors = factors.set_index(date_col).sort_index()

    for col in ["mkt-rf", "smb", "hml", "rf"]:
        if col in factors.columns:
            factors[col] = pd.to_numeric(factors[col], errors="coerce")
            if factors[col].abs().median() > 1:
                factors[col] = factors[col] / 100

    return factors.dropna(subset=["mkt-rf", "smb", "hml"])


def fama_french(asset_returns: pd.DataFrame, factors: pd.DataFrame, risk_free_annual: float) -> list[FamaFrenchResult]:
    if factors.empty:
        return []

    rf_daily_default = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    results: list[FamaFrenchResult] = []

    for asset in asset_returns.columns:
        frame = pd.concat([asset_returns[asset].rename("asset"), factors], axis=1).dropna()
        if len(frame) < 30:
            continue

        rf = frame["rf"] if "rf" in frame.columns else rf_daily_default
        y = frame["asset"] - rf
        x = sm.add_constant(frame[["mkt-rf", "smb", "hml"]])
        model = sm.OLS(y, x).fit()
        results.append(
            FamaFrenchResult(
                asset=asset,
                alpha_daily=float(model.params["const"]),
                market_beta=float(model.params["mkt-rf"]),
                smb_beta=float(model.params["smb"]),
                hml_beta=float(model.params["hml"]),
                r2=float(model.rsquared),
            )
        )

    return results


def garch_summary(asset_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if arch_model is None:
        return pd.DataFrame()

    for asset in asset_returns.columns:
        series = asset_returns[asset].dropna() * 100
        if len(series) < 100:
            continue
        model = arch_model(series, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        last_vol_daily = float(fit.conditional_volatility.iloc[-1] / 100)
        rows.append(
            {
                "Ativo": asset,
                "Omega": fit.params.get("omega", np.nan),
                "Alpha[1]": fit.params.get("alpha[1]", np.nan),
                "Beta[1]": fit.params.get("beta[1]", np.nan),
                "Persistencia": fit.params.get("alpha[1]", 0) + fit.params.get("beta[1]", 0),
                "Vol. condicional anual": last_vol_daily * np.sqrt(TRADING_DAYS),
                "AIC": fit.aic,
            }
        )

    return pd.DataFrame(rows)


def annualized_performance(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Retorno anual": returns.mean() * TRADING_DAYS,
            "Volatilidade anual": returns.std() * np.sqrt(TRADING_DAYS),
            "Drawdown maximo": returns.apply(max_drawdown),
        }
    ).reset_index(names="Ativo")


def max_drawdown(series: pd.Series) -> float:
    wealth = (1 + series.dropna()).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return float(drawdown.min()) if not drawdown.empty else np.nan


def capm_dataframe(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, risk_free_annual: float) -> pd.DataFrame:
    columns = ["asset", "alpha_daily", "alpha_annual", "beta", "risk_premium_annual", "r2", "p_value_beta"]
    rows = [result.__dict__ for result in capm(asset_returns, benchmark_returns, risk_free_annual)]
    return pd.DataFrame(rows, columns=columns)


def prediction_frame(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, asset: str) -> pd.DataFrame:
    base = pd.concat(
        {
            "retorno": asset_returns[asset],
            "mercado": benchmark_returns,
        },
        axis=1,
    ).dropna()

    frame = pd.DataFrame(index=base.index)
    for lag in [1, 2, 3, 5, 10]:
        frame[f"ret_lag_{lag}"] = base["retorno"].shift(lag)
        frame[f"mkt_lag_{lag}"] = base["mercado"].shift(lag)

    frame["media_5"] = base["retorno"].rolling(5).mean().shift(1)
    frame["vol_5"] = base["retorno"].rolling(5).std().shift(1)
    frame["media_21"] = base["retorno"].rolling(21).mean().shift(1)
    frame["vol_21"] = base["retorno"].rolling(21).std().shift(1)
    frame["target"] = base["retorno"].shift(-1)
    return frame.dropna()


def model_catalog(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "Random Forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Boosting": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=12,
            random_state=random_state,
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
        )
    return models


def backtest_ml_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in frame.columns if col != "target"]
    X = frame[feature_cols]
    y = frame["target"]

    if len(frame) < 160:
        return pd.DataFrame(), pd.DataFrame()

    splitter = ExpandingWindowSplitter(
        initial_window=min(252, max(90, len(frame) // 2)),
        step_length=10,
        fh=[1],
    )
    models = model_catalog()
    rows = []
    predictions = []

    for model_name, model in models.items():
        fold_actual = []
        fold_pred = []
        strategy_returns = []
        bh_returns = []
        dates = []

        for train_idx, test_idx in splitter.split(y):
            if len(test_idx) == 0:
                continue

            test_pos = int(test_idx[0])
            fitted = model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = float(fitted.predict(X.iloc[[test_pos]])[0])
            actual = float(y.iloc[test_pos])
            signal = 1 if pred > 0 else 0

            dates.append(y.index[test_pos])
            fold_pred.append(pred)
            fold_actual.append(actual)
            strategy_returns.append(signal * actual)
            bh_returns.append(actual)

        if not fold_actual:
            continue

        actual_arr = np.array(fold_actual)
        pred_arr = np.array(fold_pred)
        strategy_arr = np.array(strategy_returns)
        bh_arr = np.array(bh_returns)
        rows.append(
            {
                "Modelo": model_name,
                "RMSE": float(np.sqrt(mean_squared_error(actual_arr, pred_arr))),
                "MAE": float(mean_absolute_error(actual_arr, pred_arr)),
                "Acuracia direcional": float((np.sign(pred_arr) == np.sign(actual_arr)).mean()),
                "Retorno estrategia": float(np.prod(1 + strategy_arr) - 1),
                "Buy and hold": float(np.prod(1 + bh_arr) - 1),
                "Operacoes": len(actual_arr),
            }
        )

        predictions.extend(
            {
                "Data": date,
                "Modelo": model_name,
                "Previsto": pred,
                "Realizado": actual,
                "Retorno estrategia": strat,
                "Retorno buy and hold": bh,
            }
            for date, pred, actual, strat, bh in zip(dates, fold_pred, fold_actual, strategy_returns, bh_returns)
        )

    return pd.DataFrame(rows), pd.DataFrame(predictions)


def fit_recurrent_model(frame: pd.DataFrame, model_type: str) -> tuple[float | None, str]:
    if torch is None or nn is None:
        return None, "PyTorch nao esta instalado neste ambiente."

    feature_cols = [col for col in frame.columns if col != "target"]
    if len(frame) < 220:
        return None, "Amostra insuficiente para treinar rede recorrente."

    values = frame[feature_cols + ["target"]].to_numpy(dtype=float)
    X_raw = values[:, :-1]
    y_raw = values[:, -1]
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std == 0] = 1
    X_scaled = (X_raw - mean) / std

    lookback = 12
    X_seq = []
    y_seq = []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback : i])
        y_seq.append(y_raw[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    split = int(len(X_seq) * 0.8)
    if split <= 0 or split >= len(X_seq):
        return None, "Amostra insuficiente para separar treino e teste."

    class RecurrentRegressor(nn.Module):
        def __init__(self, kind: str, n_features: int):
            super().__init__()
            layer_cls = nn.LSTM if kind == "LSTM" else nn.GRU
            self.rnn = layer_cls(input_size=n_features, hidden_size=16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            output, _ = self.rnn(x)
            return self.head(output[:, -1, :]).squeeze(-1)

    torch.manual_seed(42)
    X_train = torch.tensor(X_seq[:split], dtype=torch.float32)
    y_train = torch.tensor(y_seq[:split], dtype=torch.float32)
    X_test = torch.tensor(X_seq[split:], dtype=torch.float32)

    model = RecurrentRegressor(model_type, X_seq.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(35):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()
    rmse = float(np.sqrt(mean_squared_error(y_seq[split:], pred)))
    return rmse, "ok"


def format_percent_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    styled = df.copy()
    for col in columns:
        if col in styled.columns:
            styled[col] = styled[col].map(lambda value: f"{value:.2%}" if pd.notna(value) else "-")
    return styled


def pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else "-"


def compact_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    if abs(value) >= 1_000_000_000_000:
        return f"US$ {value / 1_000_000_000_000:.2f} tri"
    if abs(value) >= 1_000_000_000:
        return f"US$ {value / 1_000_000_000:.2f} bi"
    if abs(value) >= 1_000_000:
        return f"US$ {value / 1_000_000:.2f} mi"
    return f"US$ {value:,.0f}"


def format_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    if "Valor de mercado" in formatted.columns:
        formatted["Valor de mercado"] = formatted["Valor de mercado"].map(compact_money)
    for col in ["ROE", "Margem liquida", "Dividend yield"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(pct)
    for col in ["P/L", "P/VP", "Divida/Patrimonio"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda value: f"{value:.2f}" if pd.notna(value) else "-")
    return formatted


def style_figure(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, Arial", "color": "#172033", "size": 13},
        margin={"l": 20, "r": 20, "t": 24, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12},
        },
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d9dee8")
    fig.update_yaxes(gridcolor="#edf0f5", zeroline=False, linecolor="#d9dee8")
    return fig


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{note}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Projeto TAF - Entrega II", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --app-bg: #eef2f6;
            --panel: #ffffff;
            --ink: #111827;
            --muted: #667085;
            --line: #d9e0ea;
            --accent: #0f9f88;
            --accent-strong: #087968;
            --accent-soft: #e3f7f1;
            --gold: #c7933e;
            --blue: #2f5f8f;
            --sidebar: #111827;
            --sidebar-soft: #1f2937;
            --sidebar-muted: #aab4c2;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f8fafc 0%, var(--app-bg) 42%, #eef2f6 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.35rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid #0b1220;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.15rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--sidebar-muted);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f8fafc;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-top: 0.65rem;
        }

        [data-testid="stSidebar"] label {
            color: #d7dde7;
            font-size: 0.78rem;
            font-weight: 600;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stNumberInput"] input {
            border: 1px solid #334155;
            border-radius: 8px;
            background: #f8fafc;
            color: #0f172a;
            min-height: 40px;
            box-shadow: none;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stDateInput"] input:focus,
        [data-testid="stNumberInput"] input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(15,159,136,0.24), rgba(47,95,143,0.16)),
                #111827;
            border: 1px solid #273447;
            border-radius: 8px;
            padding: 1.65rem 1.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 44px rgba(17, 24, 39, 0.16);
        }

        .eyebrow {
            color: #8be0cf;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .hero h1 {
            color: #ffffff;
            font-size: 2.25rem;
            line-height: 1.1;
            letter-spacing: 0;
            margin: 0.35rem 0;
        }

        .hero p {
            color: #cbd5e1;
            max-width: 820px;
            font-size: 0.98rem;
            margin: 0;
        }

        .module-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.8rem 0 1.1rem;
        }

        .module-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-top: 3px solid var(--accent);
            border-radius: 8px;
            padding: 0.95rem;
            box-shadow: 0 12px 30px rgba(17, 24, 39, 0.055);
        }

        .module-card b {
            color: var(--ink);
            display: block;
            font-size: 0.93rem;
            margin-bottom: 0.22rem;
        }

        .module-card span {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .sidebar-brand {
            background: var(--sidebar-soft);
            border: 1px solid #334155;
            border-left: 3px solid #31c6ac;
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            margin: 0.2rem 0 1.15rem;
        }

        .sidebar-brand strong {
            color: #ffffff;
            display: block;
            font-size: 1.02rem;
            letter-spacing: 0;
            margin-bottom: 0.12rem;
        }

        .sidebar-brand span {
            color: var(--sidebar-muted);
            font-size: 0.76rem;
        }

        .sidebar-section {
            color: #8be0cf;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: .09em;
            margin: 1.2rem 0 0.55rem;
            text-transform: uppercase;
        }

        [data-testid="stFileUploader"] {
            background: #f8fafc;
            border: 1px dashed #64748b;
            border-radius: 8px;
            padding: 0.7rem;
            box-shadow: none;
        }

        [data-testid="stFileUploader"] section {
            padding: 0.35rem 0;
            border: 0;
        }

        [data-testid="stFileUploader"] button {
            border-radius: 8px;
            border-color: #d5dce6;
        }

        [data-testid="stSidebar"] hr {
            margin: 1rem 0;
            border-color: #334155;
        }

        .section-head {
            margin: 1.1rem 0 0.75rem;
        }

        .section-head h2 {
            color: var(--ink);
            font-size: 1.22rem;
            margin: 0;
            letter-spacing: 0;
        }

        .section-head p {
            color: var(--muted);
            font-size: 0.88rem;
            margin: 0.25rem 0 0;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            min-height: 120px;
            box-shadow: 0 14px 34px rgba(17, 24, 39, 0.06);
        }

        .metric-card span {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }

        .metric-card strong {
            color: var(--ink);
            display: block;
            font-size: 1.65rem;
            letter-spacing: 0;
            line-height: 1.1;
        }

        .metric-card small {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            margin-top: 0.55rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            border-bottom: 0;
            padding-top: 0.15rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.35rem;
            box-shadow: 0 10px 28px rgba(17, 24, 39, 0.035);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: var(--muted);
            font-weight: 600;
            padding: 0.65rem 0.85rem;
        }

        .stTabs [aria-selected="true"] {
            background: #111827;
            color: #ffffff;
            border: 1px solid #111827;
        }

        [data-testid="stDataFrame"],
        [data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 14px 34px rgba(17, 24, 39, 0.055);
        }

        [data-testid="stPlotlyChart"] {
            padding: 0.8rem;
        }

        h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .module-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 620px) {
            .module-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 1.65rem;
            }
        }
    </style>

    <section class="hero">
        <div class="eyebrow">Fase II - Gestao Quantitativa Brasil</div>
        <h1>Filtro de Risco, Predicao e Backtesting</h1>
        <p>Analise de ativos brasileiros com filtros econometricos, modelos preditivos e validacao temporal para apoiar a selecao inicial da carteira.</p>
    </section>

    <div class="module-grid">
        <div class="module-card"><b>Fundamentalista</b><span>Qualidade, valor, setor e indicadores contabeis.</span></div>
        <div class="module-card"><b>CAPM</b><span>Beta, alfa e premio de risco contra o benchmark.</span></div>
        <div class="module-card"><b>Fama-French</b><span>Exposicao a mercado, tamanho (SMB) e valor (HML).</span></div>
        <div class="module-card"><b>ARCH/GARCH</b><span>Dinamica da volatilidade e persistencia de risco.</span></div>
        <div class="module-card"><b>Predicao</b><span>Random Forest, Boosting, validacao temporal e backtesting.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <strong>TAF Quant</strong>
            <span>Ativos brasileiros e predicao</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">Universo</div>', unsafe_allow_html=True)
    market = st.selectbox("Mercado", options=list(MARKET_PRESETS.keys()))
    preset = MARKET_PRESETS[market]
    selected_names = st.multiselect(
        "Ativos",
        options=list(preset["assets"].keys()),
        default=preset["default"],
        help="Selecione os ativos que entram no filtro inicial.",
    )
    selected_tickers = tuple(preset["assets"][name] for name in selected_names)
    benchmark = st.selectbox(
        "Benchmark",
        options=[preset["benchmark"], "BOVA11.SA"],
        index=0,
        help="Indice usado como referencia nos modelos de risco.",
    )
    with st.expander("Avancado: tickers manuais"):
        custom_tickers = st.text_input("Tickers extras", value="", placeholder="Ex: RENT3.SA, RADL3.SA")
        custom_benchmark = st.text_input("Benchmark manual", value="", placeholder="Ex: ^BVSP ou BOVA11.SA")
    start = st.date_input("Data inicial", value=pd.Timestamp("2021-01-01"))
    end = st.date_input("Data final", value=pd.Timestamp.today())
    risk_free = st.number_input("Taxa livre de risco", min_value=0.0, max_value=1.0, value=0.105, step=0.005)
    st.markdown('<div class="sidebar-section">Filtros CAPM</div>', unsafe_allow_html=True)
    max_beta_filter = st.slider("Beta maximo", min_value=0.0, max_value=3.0, value=1.4, step=0.1)
    min_alpha_filter = st.slider("Alfa anual minimo", min_value=-0.50, max_value=0.50, value=0.0, step=0.01)
    st.markdown('<div class="sidebar-section">Fontes opcionais</div>', unsafe_allow_html=True)
    price_file = st.file_uploader(
        "Precos reais (CSV)",
        type=["csv"],
        help="Use uma coluna date e uma coluna para cada ticker, incluindo o benchmark.",
    )
    factor_file = st.file_uploader(
        "Fatores Fama-French (CSV)",
        type=["csv"],
        help="Use fatores brasileiros de mercado, SMB, HML e, opcionalmente, RF.",
    )

tickers = tuple(dict.fromkeys(selected_tickers + parse_tickers(custom_tickers)))
if custom_benchmark.strip():
    benchmark = custom_benchmark.strip().upper()

if not tickers:
    st.warning("Informe pelo menos um ticker.")
    st.stop()

try:
    prices = load_price_csv(price_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if prices.empty:
    prices = load_prices(tickers, benchmark, str(start), str(end))

if prices.empty or benchmark not in prices.columns:
    st.error(
        "Nao foi possivel carregar dados reais. Confira a conexao com o Yahoo Finance ou envie um CSV real "
        "com as colunas date, os ativos e o benchmark."
    )
    st.stop()

returns = compute_returns(prices)
asset_returns = returns[[ticker for ticker in tickers if ticker in returns.columns]]
benchmark_returns = returns[benchmark]

if asset_returns.empty:
    st.error("Nenhum ativo informado retornou dados validos.")
    st.stop()

tabs = st.tabs(["Visao geral", "Fundamentalista", "CAPM", "Fama-French", "ARCH/GARCH", "Predicao", "Ranking"])

with tabs[0]:
    section_header("Visao geral", "Evolucao dos precos ajustados, retorno, volatilidade e drawdown.")
    normalized = prices / prices.iloc[0]
    fig = px.line(normalized, labels={"value": "Preco normalizado", "index": "Data", "variable": "Ticker"})
    st.plotly_chart(style_figure(fig), use_container_width=True)

    perf = annualized_performance(asset_returns)
    metric_cols = st.columns(4)
    best_return = perf.loc[perf["Retorno anual"].idxmax()]
    lowest_vol = perf.loc[perf["Volatilidade anual"].idxmin()]
    lowest_drawdown = perf.loc[perf["Drawdown maximo"].idxmax()]
    with metric_cols[0]:
        metric_card("Ativos analisados", str(len(asset_returns.columns)), "Universo selecionado")
    with metric_cols[1]:
        metric_card("Maior retorno", pct(best_return["Retorno anual"]), str(best_return["Ativo"]))
    with metric_cols[2]:
        metric_card("Menor volatilidade", pct(lowest_vol["Volatilidade anual"]), str(lowest_vol["Ativo"]))
    with metric_cols[3]:
        metric_card("Menor drawdown", pct(lowest_drawdown["Drawdown maximo"]), str(lowest_drawdown["Ativo"]))

    st.dataframe(
        format_percent_columns(perf, ["Retorno anual", "Volatilidade anual", "Drawdown maximo"]),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    section_header("Analise fundamentalista", "Indicadores de valor, rentabilidade, porte e estrutura financeira dos ativos.")
    fundamentals = load_fundamentals(tuple(asset_returns.columns))
    if fundamentals.empty:
        st.info(
            "Nao foi possivel carregar indicadores fundamentalistas pelo Yahoo Finance. "
            "Os filtros econometricos continuam disponiveis com os precos reais."
        )
    else:
        st.dataframe(format_fundamentals(fundamentals), use_container_width=True, hide_index=True)

        chart_data = fundamentals.dropna(subset=["P/L", "ROE"], how="all")
        if not chart_data.empty:
            fig = px.scatter(
                chart_data,
                x="P/L",
                y="ROE",
                text="Ativo",
                color="Setor",
                size="Valor de mercado",
                labels={"P/L": "Preco/Lucro", "ROE": "ROE"},
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(style_figure(fig, height=390), use_container_width=True)

with tabs[2]:
    section_header("CAPM", "Estimacao de beta, alfa e premio de risco de cada ativo frente ao benchmark.")
    capm_df = capm_dataframe(asset_returns, benchmark_returns, risk_free)
    if capm_df.empty:
        st.warning("Amostra insuficiente para estimar CAPM. Cada ativo precisa de pelo menos 30 retornos alinhados ao benchmark.")
    else:
        capm_df = capm_df.rename(
            columns={
                "asset": "Ativo",
                "alpha_daily": "Alfa diario",
                "alpha_annual": "Alfa anual",
                "beta": "Beta",
                "risk_premium_annual": "Premio de risco anual",
                "r2": "R2",
                "p_value_beta": "p-valor beta",
            }
        )
        st.dataframe(
            format_percent_columns(capm_df, ["Alfa diario", "Alfa anual", "Premio de risco anual"]),
            use_container_width=True,
            hide_index=True,
        )
        fig = px.bar(capm_df, x="Ativo", y="Beta", color="Beta", color_continuous_scale="Tealgrn")
        st.plotly_chart(style_figure(fig, height=390), use_container_width=True)

with tabs[3]:
    section_header("Modelo de tres fatores", "Decomposicao dos retornos em mercado, tamanho (SMB) e valor (HML).")
    factor_source = "CSV enviado"
    if factor_file is not None:
        try:
            factors = load_factor_csv(factor_file)
        except ValueError as exc:
            st.error(str(exc))
            factors = pd.DataFrame()
    else:
        factors = pd.DataFrame()

    if factors.empty:
        st.info(
            "Para ativos brasileiros, envie um CSV com fatores brasileiros de mercado, SMB e HML. "
            "Isso evita usar fatores dos EUA em uma aplicacao focada no Brasil."
        )
    else:
        st.caption(f"Fonte dos fatores: {factor_source}")
        ff_results = fama_french(asset_returns, factors, risk_free)
        ff_df = pd.DataFrame([result.__dict__ for result in ff_results])
        if ff_df.empty:
            st.warning("Nao houve intersecao suficiente entre retornos e fatores.")
        else:
            ff_df = ff_df.rename(
                columns={
                    "asset": "Ativo",
                    "alpha_daily": "Alfa diario",
                    "market_beta": "Beta mercado",
                    "smb_beta": "SMB",
                    "hml_beta": "HML",
                    "r2": "R2",
                }
            )
            st.dataframe(format_percent_columns(ff_df, ["Alfa diario"]), use_container_width=True, hide_index=True)
            melted = ff_df.melt(id_vars="Ativo", value_vars=["Beta mercado", "SMB", "HML"], var_name="Fator")
            fig = px.bar(
                melted,
                x="Ativo",
                y="value",
                color="Fator",
                barmode="group",
                color_discrete_sequence=["#1f9d8a", "#4f6f9f", "#d98f45"],
            )
            st.plotly_chart(style_figure(fig, height=390), use_container_width=True)

with tabs[4]:
    section_header("ARCH/GARCH", "Modelagem da heterocedasticidade condicional e da persistencia da volatilidade.")
    if arch_model is None:
        st.error("Instale a dependencia `arch` para habilitar esta aba.")
    else:
        garch_df = garch_summary(asset_returns)
        if garch_df.empty:
            st.warning("Amostra insuficiente para estimar GARCH(1,1).")
        else:
            st.dataframe(
                format_percent_columns(garch_df, ["Vol. condicional anual"]),
                use_container_width=True,
                hide_index=True,
            )
            fig = px.bar(
                garch_df,
                x="Ativo",
                y="Persistencia",
                color="Vol. condicional anual",
                color_continuous_scale="Tealgrn",
            )
            st.plotly_chart(style_figure(fig, height=390), use_container_width=True)

with tabs[5]:
    section_header("Predicao e backtesting", "Modelos de machine learning treinados com validacao temporal para prever retornos futuros.")
    prediction_asset = st.selectbox("Ativo para predicao", options=list(asset_returns.columns))
    frame_pred = prediction_frame(asset_returns, benchmark_returns, prediction_asset)

    if frame_pred.empty or len(frame_pred) < 160:
        st.warning("Amostra insuficiente para treinar e validar os modelos preditivos.")
    else:
        metrics_ml, predictions_ml = backtest_ml_models(frame_pred)
        if metrics_ml.empty:
            st.warning("Nao foi possivel gerar validacao temporal para os modelos.")
        else:
            view_metrics = metrics_ml.copy()
            view_metrics["RMSE"] = view_metrics["RMSE"].map(lambda value: f"{value:.4%}")
            view_metrics["MAE"] = view_metrics["MAE"].map(lambda value: f"{value:.4%}")
            view_metrics["Acuracia direcional"] = view_metrics["Acuracia direcional"].map(pct)
            view_metrics["Retorno estrategia"] = view_metrics["Retorno estrategia"].map(pct)
            view_metrics["Buy and hold"] = view_metrics["Buy and hold"].map(pct)
            st.dataframe(view_metrics, use_container_width=True, hide_index=True)

            curve = predictions_ml.copy()
            curve["Estrategia acumulada"] = curve.groupby("Modelo")["Retorno estrategia"].transform(lambda s: (1 + s).cumprod() - 1)
            curve["Buy and hold acumulado"] = curve.groupby("Modelo")["Retorno buy and hold"].transform(lambda s: (1 + s).cumprod() - 1)
            curve_plot = curve.melt(
                id_vars=["Data", "Modelo"],
                value_vars=["Estrategia acumulada", "Buy and hold acumulado"],
                var_name="Serie",
                value_name="Retorno acumulado",
            )
            fig = px.line(curve_plot, x="Data", y="Retorno acumulado", color="Modelo", line_dash="Serie")
            st.plotly_chart(style_figure(fig, height=410), use_container_width=True)

        dl_rows = []
        for model_type in ["GRU", "LSTM"]:
            rmse, status = fit_recurrent_model(frame_pred, model_type)
            dl_rows.append({"Modelo": model_type, "RMSE holdout": rmse, "Status": status})
        dl_df = pd.DataFrame(dl_rows)
        dl_view = dl_df.copy()
        dl_view["RMSE holdout"] = dl_view["RMSE holdout"].map(lambda value: f"{value:.4%}" if pd.notna(value) else "-")
        st.dataframe(dl_view, use_container_width=True, hide_index=True)
        if torch is None:
            st.info("GRU e LSTM estao estruturados no app, mas exigem PyTorch instalado no ambiente.")

with tabs[6]:
    section_header("Filtro consolidado", "Ranking final combinando retorno, volatilidade, drawdown, beta e risco condicional.")
    perf = annualized_performance(asset_returns)
    capm_df = capm_dataframe(asset_returns, benchmark_returns, risk_free)
    garch_df = garch_summary(asset_returns) if arch_model is not None else pd.DataFrame()

    ranking = perf.rename(columns={"Ativo": "asset"}).merge(capm_df, on="asset", how="left")
    if not garch_df.empty:
        ranking = ranking.merge(garch_df[["Ativo", "Vol. condicional anual", "Persistencia"]].rename(columns={"Ativo": "asset"}), on="asset", how="left")
    else:
        ranking["Vol. condicional anual"] = np.nan
        ranking["Persistencia"] = np.nan

    ranking["Score risco"] = (
        ranking["Volatilidade anual"].rank(pct=True)
        + ranking["Drawdown maximo"].abs().rank(pct=True)
        + ranking["beta"].abs().rank(pct=True)
        + ranking["Vol. condicional anual"].rank(pct=True)
    )
    ranking["Score risco"] = ranking["Score risco"].fillna(ranking["Score risco"].median())
    ranking["Passa filtro beta"] = ranking["beta"].le(max_beta_filter)
    ranking["Passa filtro alpha"] = ranking["alpha_annual"].ge(min_alpha_filter)
    ranking["Passa filtros CAPM"] = ranking["Passa filtro beta"] & ranking["Passa filtro alpha"]
    if len(ranking) >= 3 and ranking["Score risco"].nunique() >= 3:
        ranking["Classificacao"] = pd.qcut(
            ranking["Score risco"].rank(method="first"),
            q=3,
            labels=["Menor risco", "Risco medio", "Maior risco"],
        )
    else:
        ranking["Classificacao"] = "Risco medio"

    view = ranking.rename(
        columns={
            "asset": "Ativo",
            "beta": "Beta",
            "alpha_annual": "Alfa anual",
            "risk_premium_annual": "Premio de risco anual",
            "r2": "R2 CAPM",
        }
    )[
        [
            "Ativo",
            "Classificacao",
            "Retorno anual",
            "Volatilidade anual",
            "Drawdown maximo",
            "Beta",
            "Alfa anual",
            "Premio de risco anual",
            "Vol. condicional anual",
            "Persistencia",
            "Passa filtros CAPM",
        ]
    ].sort_values("Classificacao")

    st.dataframe(
        format_percent_columns(
            view,
            ["Retorno anual", "Volatilidade anual", "Drawdown maximo", "Alfa anual", "Premio de risco anual", "Vol. condicional anual"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranking["Volatilidade anual"],
            y=ranking["Retorno anual"],
            mode="markers+text",
            text=ranking["asset"],
            textposition="top center",
            marker={"size": 14, "color": ranking["Score risco"], "colorscale": "RdYlGn_r", "showscale": True},
        )
    )
    fig.update_layout(xaxis_title="Volatilidade anual", yaxis_title="Retorno anual")
    st.plotly_chart(style_figure(fig, height=430), use_container_width=True)
