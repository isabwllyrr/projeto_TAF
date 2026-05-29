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
from pandas_datareader import data as web_data

try:
    from arch import arch_model
except ImportError:  # pragma: no cover - friendly message in the UI
    arch_model = None


TRADING_DAYS = 252
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


@st.cache_data(show_spinner=False)
def load_fama_french_factors(start: str, end: str) -> pd.DataFrame:
    raw = web_data.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start, end=end)[0]
    raw.index = pd.to_datetime(raw.index)
    raw = raw.rename(columns={"Mkt-RF": "mkt-rf", "SMB": "smb", "HML": "hml", "RF": "rf"})
    raw.columns = [str(col).strip().lower() for col in raw.columns]
    for col in ["mkt-rf", "smb", "hml", "rf"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce") / 100
    return raw[["mkt-rf", "smb", "hml", "rf"]].dropna()


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


st.set_page_config(page_title="Filtro de Risco - Fase I", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --app-bg: #f6f7f9;
            --panel: #ffffff;
            --ink: #172033;
            --muted: #687386;
            --line: #e5e9f0;
            --accent: #1f9d8a;
            --accent-soft: #e8f7f4;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #eef1f5;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] label {
            color: #2d3748;
            font-size: 0.82rem;
            font-weight: 600;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stNumberInput"] input {
            border: 1px solid transparent;
            border-radius: 8px;
            background: #ffffff;
            color: var(--ink);
            min-height: 42px;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stDateInput"] input:focus,
        [data-testid="stNumberInput"] input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }

        .hero {
            padding: 0.3rem 0 1.2rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.2rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .hero h1 {
            color: var(--ink);
            font-size: 2.15rem;
            line-height: 1.1;
            letter-spacing: 0;
            margin: 0.35rem 0;
        }

        .hero p {
            color: var(--muted);
            max-width: 760px;
            font-size: 0.98rem;
            margin: 0;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            min-height: 120px;
            box-shadow: 0 10px 28px rgba(23, 32, 51, 0.04);
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
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            color: var(--muted);
            font-weight: 600;
            padding: 0.7rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: var(--ink);
            border: 1px solid var(--line);
            border-bottom-color: #ffffff;
        }

        [data-testid="stDataFrame"],
        [data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 28px rgba(23, 32, 51, 0.03);
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
    </style>

    <section class="hero">
        <div class="eyebrow">Fase I - Gestao Quantitativa</div>
        <h1>Filtro de Risco e Econometria</h1>
        <p>Analise de ativos com CAPM, fatores de Fama-French e volatilidade condicional para apoiar a selecao inicial da carteira.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Parametros")
    raw_tickers = st.text_input("Ativos", value="AAPL, MSFT, NVDA, AMZN, GOOGL, JPM, XOM")
    benchmark = st.text_input("Benchmark", value="^GSPC").strip().upper()
    start = st.date_input("Data inicial", value=pd.Timestamp("2021-01-01"))
    end = st.date_input("Data final", value=pd.Timestamp.today())
    risk_free = st.number_input("Taxa livre de risco anual", min_value=0.0, max_value=1.0, value=0.045, step=0.005)
    st.divider()
    price_file = st.file_uploader(
        "CSV de precos reais opcional",
        type=["csv"],
        help="Use uma coluna date e uma coluna para cada ticker, incluindo o benchmark.",
    )
    factor_file = st.file_uploader(
        "CSV Fama-French opcional",
        type=["csv"],
        help="Opcional. Se nao enviar, o app tenta buscar os fatores reais de Kenneth French.",
    )

tickers = parse_tickers(raw_tickers)

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

tabs = st.tabs(["Visao geral", "Fundamentalista", "CAPM", "Fama-French", "ARCH/GARCH", "Ranking"])

with tabs[0]:
    st.subheader("Precos ajustados")
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
    st.subheader("Analise fundamentalista")
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
    st.subheader("CAPM: beta, alfa e premio de risco")
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
    st.subheader("Modelo de tres fatores de Fama-French")
    factor_source = "CSV enviado"
    if factor_file is not None:
        try:
            factors = load_factor_csv(factor_file)
        except ValueError as exc:
            st.error(str(exc))
            factors = pd.DataFrame()
    else:
        try:
            factors = load_fama_french_factors(str(start), str(end))
            factor_source = "Kenneth French Data Library"
        except Exception:
            factors = pd.DataFrame()

    if factors.empty:
        st.info(
            "Nao foi possivel carregar os fatores reais SMB e HML automaticamente. "
            "Envie um CSV de fatores para estimar Fama-French; isso nao depende de adicionar mais ativos."
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
    st.subheader("ARCH/GARCH: dinamica da volatilidade")
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
    st.subheader("Filtro consolidado")
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
            "Premio de risco anual",
            "Vol. condicional anual",
            "Persistencia",
        ]
    ].sort_values("Classificacao")

    st.dataframe(
        format_percent_columns(
            view,
            ["Retorno anual", "Volatilidade anual", "Drawdown maximo", "Premio de risco anual", "Vol. condicional anual"],
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
