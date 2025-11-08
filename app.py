
import streamlit as st
import pandas as pd
import numpy as np
import io, json, base64
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import datetime as _dt
import requests as _rq

st.set_page_config(page_title="Fronteira Eficiente Pro+ — Diário, Regras por Classe, Exportações", layout="wide")

# =====================
# Optional libs
# =====================
try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import cvxpy as cp
    CVXPY_OK = True
except Exception:
    CVXPY_OK = False

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# =====================
# Header / Badges
# =====================
st.title("Fronteira Eficiente Pro+ — Diário, Regras por Classe, Exportações")
st.caption("Dados reais (BCB + yfinance), janela 1–3 anos, μ personalizável, Σ com shrinkage, limites por classe e por grupo, fronteira eficiente, "
           "regras setoriais (RDMercado/Crédito), exportações Excel/PNG e diagnósticos.")

# =====================
# Utils
# =====================
def _badge(text: str, color: str = "#0ea5e9"):
    st.markdown(f"<span style='background:{color}; color:white; padding:3px 8px; border-radius:12px; font-size:12px'>{text}</span>", unsafe_allow_html=True)

def read_uploaded_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        content = uploaded_file.getvalue()
        if content is None or len(content) == 0:
            st.error("O arquivo enviado está vazio. Envie um CSV válido ou use o modo de dados reais.")
            return None
        return pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError:
        st.error("O arquivo enviado está vazio. Verifique o CSV.")
        return None
    except Exception as e:
        st.error(f"Falha ao ler o CSV: {e}")
        return None

def detect_format(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = [c.lower() for c in df.columns]
    if all(x in cols_lower for x in ["date","ticker","price"]):
        low = {c.lower(): c for c in df.columns}
        piv = df.rename(columns={low["date"]: "date", low["ticker"]: "ticker", low["price"]: "price"})
        piv["date"] = pd.to_datetime(piv["date"])
        wide = piv.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()
        return wide
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df

def get_prices(uploaded_file):
    if uploaded_file is not None:
        dfu = read_uploaded_csv(uploaded_file)
        if dfu is None:
            return None
        return detect_format(dfu)
    if st.session_state.get("__sample_loaded__") and "sample_prices" in st.session_state:
        return st.session_state["sample_prices"]
    return None

def to_returns(df_prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method == "pct":
        rets = df_prices.pct_change().dropna(how="all")
    else:
        rets = np.log(df_prices/df_prices.shift(1)).dropna(how="all")
    return rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")

def annualize(mu: pd.Series, sigma: pd.DataFrame, periods_per_year: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
    mu_ann = mu * periods_per_year
    Sigma_ann = sigma * periods_per_year
    return mu_ann, Sigma_ann

def ewma_mean(rets: pd.DataFrame, lam: float) -> pd.Series:
    weights = np.array([lam**i for i in range(len(rets))])[::-1]
    weights = weights/weights.sum()
    return pd.Series(np.dot(weights, rets.values), index=rets.columns)

def ewma_cov(rets: pd.DataFrame, lam: float) -> pd.DataFrame:
    mu_ew = ewma_mean(rets, lam)
    X = rets - mu_ew
    Sigma = np.zeros((X.shape[1], X.shape[1]))
    w_sum = 0.0
    for t in range(X.shape[0]-1, -1, -1):
        x = X.iloc[t].values.reshape(-1, 1)
        Sigma = lam*Sigma + (1-lam)*(x @ x.T)
        w_sum = lam*w_sum + (1-lam)
    Sigma = Sigma / max(w_sum, 1e-12)
    return pd.DataFrame(Sigma, index=rets.columns, columns=rets.columns)

def ledoit_wolf_cov(rets: pd.DataFrame) -> pd.DataFrame:
    if not SKLEARN_OK:
        st.warning("scikit-learn não disponível. Usando covariância amostral.")
        return rets.cov()
    lw = LedoitWolf().fit(rets.values)
    return pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)

def regularize_spd(Sigma: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    try:
        np.linalg.cholesky(Sigma.values)
        return Sigma
    except np.linalg.LinAlgError:
        pass
    A = Sigma.values.copy()
    boost = eps
    for _ in range(12):
        try:
            np.linalg.cholesky(A + boost*np.eye(A.shape[0]))
            return pd.DataFrame(A + boost*np.eye(A.shape[0]), index=Sigma.index, columns=Sigma.columns)
        except np.linalg.LinAlgError:
            boost *= 10.0
    return pd.DataFrame(A + boost*np.eye(A.shape[0]), index=Sigma.index, columns=Sigma.columns)

def portfolio_metrics(w: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame) -> Dict[str, float]:
    mu_p = float(mu.values @ w)
    sig_p = float(np.sqrt(w @ Sigma.values @ w))
    return {"mu": mu_p, "sigma": sig_p}

# ===== Optimization helpers =====
def optimize_qp(mu: pd.Series, Sigma: pd.DataFrame, target_ret: Optional[float], rf: float,
                lb: np.ndarray, ub: np.ndarray, group_bounds: List[Tuple[List[int], float, float]],
                objective: str) -> Optional[np.ndarray]:
    if not CVXPY_OK:
        return None
    n = len(mu)
    w = cp.Variable(n)
    Sigma_cp = cp.psd_wrap(Sigma.values)
    cons = [cp.sum(w) == 1, w >= lb, w <= ub]
    for idxs, gmin, gmax in group_bounds:
        if idxs:
            cons.append(cp.sum(w[idxs]) >= gmin)
            cons.append(cp.sum(w[idxs]) <= gmax)
    if objective == "Sharpe Máximo":
        if target_ret is None:
            mu_min = float(np.min(mu.values))
            mu_max = float(np.max(mu.values))
            grid = np.linspace(mu_min, mu_max, 25)
            best_w, best_sharpe = None, -1e18
            for t in grid:
                obj = cp.Minimize(cp.quad_form(w, Sigma_cp))
                prob = cp.Problem(obj, cons + [mu.values @ w >= t])
                try:
                    prob.solve(solver=cp.OSQP, verbose=False)
                    if w.value is None:
                        prob.solve(solver=cp.SCS, verbose=False)
                except Exception:
                    continue
                if w.value is None:
                    continue
                wv = np.array(w.value).ravel()
                mu_p = float(mu.values @ wv)
                sig_p = float(np.sqrt(wv @ Sigma.values @ wv))
                if sig_p > 0:
                    sharpe = (mu_p - rf) / sig_p
                    if sharpe > best_sharpe:
                        best_sharpe, best_w = sharpe, wv
            return best_w
        else:
            obj = cp.Minimize(cp.quad_form(w, Sigma_cp))
            cons2 = cons + [mu.values @ w >= target_ret]
            prob = cp.Problem(obj, cons2)
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
                if w.value is None:
                    prob.solve(solver=cp.SCS, verbose=False)
            except Exception:
                return None
            return None if w.value is None else np.array(w.value).ravel()
    else:
        obj = cp.Minimize(cp.quad_form(w, Sigma_cp))
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if w.value is None:
                prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return None
        return None if w.value is None else np.array(w.value).ravel()

def compute_frontier(mu: pd.Series, Sigma: pd.DataFrame, lb: np.ndarray, ub: np.ndarray,
                     group_bounds: List[Tuple[List[int], float, float]], n_points: int = 30) -> List[Dict]:
    if not CVXPY_OK:
        return []
    mu_vals = mu.values
    mu_min = float(np.min(mu_vals))
    mu_max = float(np.max(mu_vals))
    targets = np.linspace(mu_min, mu_max, n_points)
    out = []
    for t in targets:
        w = optimize_qp(mu, Sigma, t, 0.0, lb, ub, group_bounds, "Sharpe Máximo")
        if w is None:
            continue
        m = portfolio_metrics(w, mu, Sigma)
        out.append({"target": float(t), "w": w, **m})
    return out

# =====================
# Providers — BCB (SGS) + yfinance (cached)
# =====================
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_yf_prices(ticker: str, start: str = "2018-01-01", end: Optional[str] = None) -> Optional[pd.Series]:
    if not YF_OK:
        st.error("yfinance não está instalado. Adicione 'yfinance' no requirements.txt.")
        return None
    try:
        if end is None:
            end = _dt.date.today().isoformat()
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            return None
        s = None
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0):
                s = df['Adj Close']
            elif 'Close' in df.columns.get_level_values(0):
                s = df['Close']
            if isinstance(s, pd.DataFrame):
                if s.shape[1] >= 1:
                    s = s.iloc[:, 0]
        else:
            if 'Adj Close' in df.columns:
                s = df['Adj Close']
            elif 'Close' in df.columns:
                s = df['Close']
        if s is None:
            return None
        s = pd.to_numeric(s, errors='coerce').dropna()
        s.index = pd.to_datetime(s.index, errors='coerce')
        s = s[~s.index.isna()]
        s.name = ticker
        return s
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_bcb_series(code: int, start: str = "2018-01-01") -> Optional[pd.Series]:
    try:
        start_br = pd.to_datetime(start).strftime("%d/%m/%Y")
        end_br = _dt.date.today().strftime("%d/%m/%Y")
        base = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        url = f"{base}?formato=json&dataInicial={start_br}&dataFinal={end_br}"
        r = _rq.get(url, timeout=30)
        try:
            r.raise_for_status()
        except Exception:
            url2 = f"{base}/ultimos/5000?formato=json"
            r = _rq.get(url2, timeout=30)
            r.raise_for_status()
        js = r.json()
        df = pd.DataFrame(js)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['data']).set_index('data')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        s = df['valor'].dropna()
        s.name = f"BCB_{code}"
        return s
    except Exception:
        return None

def _rate_series_to_price_index(rate_annual_pct: pd.Series) -> pd.Series:
    if rate_annual_pct.empty:
        return rate_annual_pct
    r_daily = rate_annual_pct / 100.0 / 252.0
    idx = 100.0 * (1.0 + r_daily).cumprod()
    idx.name = rate_annual_pct.name
    return idx

def build_real_dataset(mapping: pd.DataFrame, start: str = "2018-01-01") -> pd.DataFrame:
    cols = []
    failures = []
    for _, row in mapping.iterrows():
        classe = str(row.get('Classe','')).strip()
        fonte  = str(row.get('Fonte','')).strip().lower()
        ident  = str(row.get('Identificador','')).strip()
        if not classe or not fonte or not ident:
            continue
        try:
            if fonte == 'yfinance':
                s = _fetch_yf_prices(ident, start=start)
                if s is None or s.empty:
                    failures.append((classe, fonte, ident, 'Sem dados via yfinance'))
                else:
                    cols.append(s.rename(classe))
            elif fonte == 'bcb':
                try:
                    code_int = int(float(ident))
                except Exception:
                    failures.append((classe, fonte, ident, 'Código BCB inválido'))
                    continue
                s = _fetch_bcb_series(code_int, start=start)
                if s is None or s.empty:
                    failures.append((classe, fonte, ident, 'Sem dados via BCB/SGS'))
                else:
                    s_idx = _rate_series_to_price_index(s).rename(classe)
                    cols.append(s_idx)
            else:
                failures.append((classe, fonte, ident, "Fonte desconhecida. Use 'yfinance' ou 'bcb'."))
        except Exception as e:
            failures.append((classe, fonte, ident, f'Erro inesperado: {e}'))
    fallback_map = {
        "CreditoTrad": ("yfinance", "IRFM11.SA"),
        "CredEstrut":  ("yfinance", "IRFM11.SA"),
        "RDMercado":   ("yfinance", "IRFM11.SA"),
    }
    assembled = set([s.name for s in cols]) if cols else set()
    fallbacks_applied = []
    for (classe, fonte, ident, motivo) in list(failures):
        if classe in fallback_map and classe not in assembled:
            fb_fonte, fb_ident = fallback_map[classe]
            try:
                if fb_fonte == "yfinance":
                    sfb = _fetch_yf_prices(fb_ident, start=start)
                    if sfb is not None and not sfb.empty:
                        cols.append(sfb.rename(classe))
                        assembled.add(classe)
                        fallbacks_applied.append((classe, fb_fonte, fb_ident, f"Fallback aplicado (falha original: {motivo})"))
            except Exception as e:
                failures.append((classe, fb_fonte, fb_ident, f"Fallback falhou: {e}"))
    st.session_state['__build_failures__'] = failures
    st.session_state['__fallbacks_applied__'] = fallbacks_applied
    if not cols:
        return pd.DataFrame()
    panel = pd.concat(cols, axis=1).sort_index().dropna(how='all')
    panel = panel.ffill().dropna(how='any')
    return panel

# =====================
# Class overrides / business rules
# =====================
IDADI_SIGMA = {"1a": 0.0204, "2a": 0.0145, "3a": 0.0176}
IDADI_MU    = {"1a": 0.1335, "2a": 0.1353, "3a": 0.1336}

def _apply_variance_override(Sigma: pd.DataFrame, asset: str, target_sigma: float) -> pd.DataFrame:
    if asset not in Sigma.index:
        return Sigma
    cur_var = float(Sigma.loc[asset, asset])
    if cur_var <= 0:
        return Sigma
    cur_sigma = float(np.sqrt(cur_var))
    if cur_sigma <= 0:
        return Sigma
    scale = target_sigma / cur_sigma
    S = Sigma.copy()
    S.loc[asset, :] *= scale
    S.loc[:, asset] *= scale
    S.loc[asset, asset] = target_sigma**2
    return S

def _safely_get_series(df_prices: pd.DataFrame, col: str, fallback_ticker: Optional[str], start: str) -> Optional[pd.Series]:
    if col in df_prices.columns:
        return df_prices[col].dropna()
    if fallback_ticker:
        s = _fetch_yf_prices(fallback_ticker, start=start)
        return s
    return None

def apply_class_overrides_from_prices(prices: pd.DataFrame, mu_ann: pd.Series, Sigma_ann: pd.DataFrame, win_key: str, start_date: str) -> Tuple[pd.Series, pd.DataFrame]:
    mu = mu_ann.copy()
    S = Sigma_ann.copy()
    cols = list(prices.columns)

    rets_all = to_returns(prices, method="log")
    try:
        s_irfm = _safely_get_series(prices, "RDMercado", "IRFM11.SA", start_date)
        s_selic = _safely_get_series(prices, "SELIC", None, start_date)
        if s_irfm is not None and s_selic is not None:
            r = to_returns(pd.concat([s_irfm.rename("IRFM"), s_selic.rename("SELIC")], axis=1).dropna())
            r_mix = 0.3*r["IRFM"] + 0.7*r["SELIC"]
            mu.loc["RDMercado"] = float(r_mix.mean() * 252)
            for a in cols:
                ra = rets_all[a].tail(len(r_mix))
                cov = float(np.cov(r_mix, ra, ddof=1)[0,1] * 252)
                S.loc["RDMercado", a] = cov
                S.loc[a, "RDMercado"] = cov
            S.loc["RDMercado","RDMercado"] = float(r_mix.var() * 252)
    except Exception as e:
        st.warning(f"Mix RDMercado falhou: {e}")

    if "CreditoTrad" in cols:
        try:
            mu.loc["CreditoTrad"] = IDADI_MU.get(win_key, mu.get("CreditoTrad", 0.0))
            target_sigma = IDADI_SIGMA.get(win_key, float(np.sqrt(S.loc["CreditoTrad","CreditoTrad"])))
            S = _apply_variance_override(S, "CreditoTrad", target_sigma)
        except Exception as e:
            st.warning(f"Override CreditoTrad falhou: {e}")

    if "CredEstrut" in cols:
        try:
            mu_selic = float(mu.get("SELIC", 0.0))
            mu.loc["CredEstrut"] = mu_selic + 0.01
            if "CreditoTrad" in cols:
                sig_ct = float(np.sqrt(S.loc["CreditoTrad","CreditoTrad"]))
                target_sigma = sig_ct + 0.01
            else:
                target_sigma = float(np.sqrt(S.loc["CredEstrut","CredEstrut"])) + 0.01
            S = _apply_variance_override(S, "CredEstrut", target_sigma)
        except Exception as e:
            st.warning(f"Override CredEstrut falhou: {e}")

    S = regularize_spd(S, eps=1e-8)
    return mu, S

# =====================
# Sidebar — Inputs & Config
# =====================
st.sidebar.header("1) Dados")
uploaded = st.sidebar.file_uploader("CSV de preços (formato largo) ou (date,ticker,price). Frequência: diária.", type=["csv"])

st.sidebar.checkbox("Usar dados reais (online)", key="use_real_online", help="Baixa séries de mercado (yfinance/BCB) e monta automaticamente os preços.")

if st.session_state.get("use_real_online", False):
    st.sidebar.caption("Classe → Fonte ('yfinance' ou 'bcb') → Identificador (ticker ou código SGS).")
    default_map = pd.DataFrame({
        "Classe": ["SELIC", "IMA_B5", "RDMercado", "CreditoTrad", "CredEstrut", "AcoesBR", "AcoesEXT", "Multimerc"],
        "Fonte":  ["bcb",   "yfinance","yfinance",  "yfinance",    "yfinance",  "yfinance","yfinance","yfinance"],
        "Identificador": ["1178","IMAB11.SA","IRFM11.SA","IRFM11.SA","IRFM11.SA","BOVA11.SA","^GSPC","BOVA11.SA"]
    })
    if "real_map" not in st.session_state:
        st.session_state["real_map"] = default_map.copy()

    st.session_state["real_map"] = st.sidebar.data_editor(
        st.session_state["real_map"],
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Classe": st.column_config.TextColumn("Classe"),
            "Fonte": st.column_config.TextColumn("Fonte"),
            "Identificador": st.column_config.TextColumn("Identificador"),
        },
        key="real_map_editor"
    )
    start_date_real = st.sidebar.date_input("Início das séries", value=pd.to_datetime("2020-01-01"))
    if st.sidebar.button("Baixar & Montar dados"):
        panel = build_real_dataset(st.session_state["real_map"], start=start_date_real.strftime("%Y-%m-%d"))
        if panel.empty:
            st.sidebar.error("Não foi possível montar a base real. Verifique o mapeamento.")
            fails = st.session_state.get('__build_failures__', [])
            if fails:
                with st.expander('Falhas na montagem — detalhes'):
                    fd = pd.DataFrame(fails, columns=['Classe','Fonte','Identificador','Motivo'])
                    st.dataframe(fd, use_container_width=True)
        else:
            st.session_state["__sample_loaded__"] = True
            st.session_state["sample_prices"] = panel
            st.sidebar.success(f"Base real montada com sucesso. {panel.shape[0]} linhas × {panel.shape[1]} classes.")
            fails = st.session_state.get('__build_failures__', [])
            if fails:
                with st.expander('Algumas séries falharam — detalhes'):
                    fd = pd.DataFrame(fails, columns=['Classe','Fonte','Identificador','Motivo'])
                    st.dataframe(fd, use_container_width=True)
            fbs = st.session_state.get('__fallbacks_applied__', [])
            if fbs:
                with st.expander('Fallbacks aplicados — detalhes'):
                    fbdf = pd.DataFrame(fbs, columns=['Classe','Fonte','Identificador','Observação'])
                    st.dataframe(fbdf, use_container_width=True)

st.sidebar.caption("Cálculos **Diários**, anualização por **252**.")
ret_method = st.sidebar.selectbox("Retorno", ["Log", "Percentual"], index=0)
ret_method = "log" if ret_method == "Log" else "pct"

st.sidebar.header("2) Estimação")
mu_method = st.sidebar.selectbox("Retorno esperado (μ)", ["Média histórica", "EWMA (média)"], index=0)
sigma_method = st.sidebar.selectbox("Covariância (Σ)", ["Amostral", "EWMA (cov)", "Ledoit-Wolf (shrinkage)"], index=2)
lam = st.sidebar.slider("λ (EWMA)", min_value=0.80, max_value=0.99, value=0.94, step=0.01)

st.sidebar.header("3) Otimização")
objective = st.sidebar.selectbox("Objetivo", ["Sharpe Máximo", "Mínima Variância (sem alvo)"], index=0)
rf = st.sidebar.number_input("Taxa livre de risco (anualizada)", min_value=-0.5, max_value=0.5, value=0.0, step=0.001, format="%.3f")

st.sidebar.header("Janela de estimação")
win_label = st.sidebar.selectbox("Período para μ/Σ", ["1 ano (~252 d.u.)", "2 anos (~504 d.u.)", "3 anos (~756 d.u.)"], index=1)
_win_map = {"1 ano (~252 d.u.)": 252, "2 anos (~504 d.u.)": 504, "3 anos (~756 d.u.)": 756}
st.session_state["win_days"] = _win_map.get(win_label, 504)
_badge(f"Janela ativa: {win_label}")

st.sidebar.checkbox(
    "Aplicar regras de classe (RDMercado 30/70, CreditoTrad IDA-DI, CredEstrut +1pp/+1pp)",
    value=True,
    key="apply_class_rules"
)

st.sidebar.header("4) Restrições por ativo")
_prices_preview = get_prices(uploaded)
if _prices_preview is None:
    st.sidebar.info("Carregue dados em 'Dados' ou clique em 'Baixar & Montar dados' (modo real).")
else:
    assets_list = list(_prices_preview.columns)
    if "bounds_ranges" not in st.session_state or set(st.session_state.get("bounds_assets", [])) != set(assets_list):
        st.session_state["bounds_assets"] = assets_list
        st.session_state["bounds_ranges"] = {a: (0.0, 1.0) for a in assets_list}
    new_ranges = {}
    for a in assets_list:
        min_def, max_def = st.session_state["bounds_ranges"].get(a, (0.0, 1.0))
        rng = st.sidebar.slider(a, 0.0, 1.0, (float(min_def), float(max_def)), step=0.01)
        new_ranges[a] = (float(rng[0]), float(rng[1]))
    st.session_state["bounds_ranges"] = new_ranges
    if st.sidebar.button("Resetar limites"):
        st.session_state["bounds_ranges"] = {a: (0.0, 1.0) for a in assets_list}
        st.sidebar.success("Limites resetados para [0.00, 1.00].")

# Optional Σ override
st.sidebar.subheader("Σ fixa sugerida (opcional)")
if "sigma_override" not in st.session_state and _prices_preview is not None:
    sugg = {
        "SELIC": 0.003, "RDMercado": 0.08, "IMA_B5": 0.075,
        "CreditoTrad": 0.02, "CredEstrut": 0.04,
        "AcoesBR": 0.30, "AcoesEXT": 0.20, "Multimerc": 0.10
    }
    st.session_state["sigma_override"] = {a: sugg.get(a, 0.10) for a in _prices_preview.columns}
use_sigma_override = st.sidebar.checkbox("Usar Σ fixa (substitui diagonal com sigmas informados)", value=False)
if use_sigma_override and _prices_preview is not None:
    sigma_tbl = pd.DataFrame({"Ativo": _prices_preview.columns, "sigma": [st.session_state["sigma_override"].get(a, 0.10) for a in _prices_preview.columns]})
    sigma_tbl = st.sidebar.data_editor(sigma_tbl, hide_index=True, use_container_width=True)
    st.session_state["sigma_override"] = {row["Ativo"]: float(row["sigma"]) for _, row in sigma_tbl.iterrows()}

# Group constraints
st.sidebar.header("5) Restrições por GRUPO (opcional)")
st.sidebar.caption("Defina grupos e limites de soma de pesos.")
if _prices_preview is not None:
    if "group_map" not in st.session_state:
        st.session_state["group_map"] = {
            "RendaFixa": ["SELIC","RDMercado","IMA_B5","CreditoTrad","CredEstrut"],
            "RendaVariavel": ["AcoesBR","AcoesEXT","Multimerc"]
        }
    with st.sidebar.expander("Editar grupos"):
        all_assets = list(_prices_preview.columns)
        new_map = {}
        for gname, aset in st.session_state["group_map"].items():
            picked = st.multiselect(f"{gname}", options=all_assets, default=[a for a in aset if a in all_assets], key=f"grp_{gname}")
            new_map[gname] = picked
        add_g = st.text_input("Criar novo grupo (opcional)")
        if add_g and add_g not in new_map:
            new_map[add_g] = []
        st.session_state["group_map"] = new_map
    if "group_bounds" not in st.session_state:
        st.session_state["group_bounds"] = {g:(0.0,1.0) for g in st.session_state["group_map"].keys()}
    with st.sidebar.expander("Limites por grupo"):
        gb_new = {}
        for g, aset in st.session_state["group_map"].items():
            vmin,vmax = st.session_state["group_bounds"].get(g,(0.0,1.0))
            rng = st.slider(f"{g}", 0.0, 1.0, (float(vmin), float(vmax)), step=0.01, key=f"gbl_{g}")
            gb_new[g] = (float(rng[0]), float(rng[1]))
        st.session_state["group_bounds"] = gb_new

def _make_group_constraints(assets: List[str]) -> List[Tuple[List[int], float, float]]:
    gb = []
    for g, aset in st.session_state.get("group_map", {}).items():
        idxs = [assets.index(a) for a in aset if a in assets]
        gmin, gmax = st.session_state.get("group_bounds", {}).get(g, (0.0, 1.0))
        gb.append((idxs, gmin, gmax))
    return gb

# =====================
# Main Tabs
# =====================
tab_data, tab_est, tab_opt, tab_frontier, tab_report = st.tabs(["Dados", "Parâmetros Est.", "Otimização", "Fronteira", "Relatório"])

with tab_data:
    st.subheader("1) Carregamento e pré-processamento (DIÁRIO)")
    prices = get_prices(uploaded)
    if prices is None:
        st.info("Use o **upload de CSV** ou ative **Usar dados reais (online)** na barra lateral e clique em **Baixar & Montar dados**.")
        st.stop()
    st.write("Amostra de preços (topo):")
    st.dataframe(prices.head())

    returns = to_returns(prices, method=ret_method)
    win_days = st.session_state.get("win_days", 504)
    returns = returns.tail(win_days)
    st.write("Amostra de retornos (janela ativa):")
    st.dataframe(returns.head())

with tab_est:
    st.subheader("2) Estimação de parâmetros (μ, Σ, ρ) — diário → anualizado (252)")
    prices = get_prices(uploaded)
    if prices is None:
        st.warning("Carregue dados na aba 'Dados' ou monte a base real na barra lateral.")
        st.stop()
    returns = to_returns(prices, method=ret_method)
    win_days = st.session_state.get("win_days", 504)
    returns = returns.tail(win_days)

    if mu_method.startswith("Média"):
        mu = returns.mean()
    else:
        mu = ewma_mean(returns, lam)

    if sigma_method == "Amostral":
        Sigma = returns.cov()
    elif sigma_method.startswith("EWMA"):
        Sigma = ewma_cov(returns, lam)
    else:
        Sigma = ledoit_wolf_cov(returns)

    mu_ann, Sigma_ann = annualize(mu, Sigma, periods_per_year=252)
    Sigma_ann = regularize_spd(Sigma_ann, eps=1e-8)

    if st.session_state.get("apply_class_rules", False):
        win_key = '1a' if st.session_state.get('win_days',504)==252 else ('2a' if st.session_state.get('win_days',504)==504 else '3a')
        start_str = "2020-01-01"
        mu_ann, Sigma_ann = apply_class_overrides_from_prices(prices, mu_ann, Sigma_ann, win_key, start_str)

    if use_sigma_override:
        S = Sigma_ann.copy()
        for a, s in st.session_state.get("sigma_override", {}).items():
            if a in S.index:
                S = _apply_variance_override(S, a, float(s))
        Sigma_ann = regularize_spd(S, eps=1e-8)

    corr = pd.DataFrame(np.corrcoef(returns.values.T), index=returns.columns, columns=returns.columns)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Retornos esperados (anualizados)**")
        st.dataframe(mu_ann.to_frame("mu_ann"))
    with c2:
        st.markdown("**Volatilidades esperadas (anualizadas)**")
        vol = np.sqrt(np.diag(Sigma_ann.values))
        st.dataframe(pd.Series(vol, index=Sigma_ann.index, name="sigma_ann").to_frame())

    st.markdown("**Matriz de covariância anualizada (Σ)**")
    st.dataframe(Sigma_ann)

    st.markdown("**Matriz de correlação (ρ) + Heatmap**")
    st.dataframe(corr)

    fig_corr = plt.figure()
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Matriz de Correlação — Heatmap")
    plt.tight_layout()
    st.pyplot(fig_corr)

    buf_png = io.BytesIO()
    fig_corr.savefig(buf_png, format="png", bbox_inches="tight")
    st.download_button("Baixar Heatmap (PNG)", data=buf_png.getvalue(), file_name="heatmap_correlacao.png", mime="image/png")

    st.markdown("### Retornos esperados personalizados (anualizados)")
    st.caption("Edite os valores na tabela abaixo. Ao marcar a opção, a otimização usará estes valores.")
    if "custom_mu" not in st.session_state or set(st.session_state.get("custom_mu_index", [])) != set(mu_ann.index):
        st.session_state["custom_mu"] = mu_ann.copy()
        st.session_state["custom_mu_index"] = list(mu_ann.index)

    enable_custom = st.checkbox("Usar retornos esperados personalizados", value=False)
    st.session_state["enable_custom_mu"] = enable_custom

    df_edit_src = st.session_state["custom_mu"].reindex(mu_ann.index).fillna(mu_ann)
    df_edit = pd.DataFrame({"Ativo": df_edit_src.index, "mu_ann": df_edit_src.values})
    df_edit = st.data_editor(
        df_edit,
        hide_index=True,
        use_container_width=True,
        column_config={
            "mu_ann": st.column_config.NumberColumn("mu_ann", min_value=-1.0, max_value=1.0, step=0.0001, format="%.6f"),
            "Ativo": st.column_config.TextColumn("Ativo", disabled=True),
        },
    )
    try:
        st.session_state["custom_mu"] = pd.Series(df_edit["mu_ann"].values, index=df_edit["Ativo"].values)
    except Exception as e:
        st.warning(f"Não foi possível aplicar edições de μ: {e}")

    st.download_button("Baixar μ anualizado (CSV)", data=mu_ann.to_csv().encode(), file_name="mu_annualized.csv")
    st.download_button("Baixar Σ anualizada (CSV)", data=Sigma_ann.to_csv().encode(), file_name="sigma_annualized.csv")

with tab_opt:
    st.subheader("3) Otimização e carteiras recomendadas")
    prices = get_prices(uploaded)
    if prices is None:
        st.warning("Carregue dados na aba 'Dados' ou monte a base real na barra lateral.")
        st.stop()
    rets = to_returns(prices, method=ret_method)
    win_days = st.session_state.get("win_days", 504)
    rets = rets.tail(win_days)

    if mu_method.startswith("Média"):
        mu = rets.mean()
    else:
        mu = ewma_mean(rets, lam)

    if sigma_method == "Amostral":
        Sigma = rets.cov()
    elif sigma_method.startswith("EWMA"):
        Sigma = ewma_cov(rets, lam)
    else:
        Sigma = ledoit_wolf_cov(rets)

    mu_ann, Sigma_ann = annualize(mu, Sigma, periods_per_year=252)
    Sigma_ann = regularize_spd(Sigma_ann, eps=1e-8)

    if st.session_state.get("apply_class_rules", False):
        win_key = '1a' if st.session_state.get('win_days',504)==252 else ('2a' if st.session_state.get('win_days',504)==504 else '3a')
        mu_ann, Sigma_ann = apply_class_overrides_from_prices(get_prices(uploaded), mu_ann, Sigma_ann, win_key, "2020-01-01")

    if use_sigma_override:
        S = Sigma_ann.copy()
        for a, s in st.session_state.get("sigma_override", {}).items():
            if a in S.index:
                S = _apply_variance_override(S, a, float(s))
        Sigma_ann = regularize_spd(S, eps=1e-8)

    try:
        if st.session_state.get("enable_custom_mu", False) and st.session_state.get("custom_mu") is not None:
            mu_ann = st.session_state["custom_mu"].reindex(mu_ann.index).fillna(mu_ann)
    except Exception:
        pass

    assets = list(mu_ann.index)
    br = st.session_state.get("bounds_ranges", {})
    lb = np.array([float(br.get(a, (0.0,1.0))[0]) for a in assets])
    ub = np.array([float(br.get(a, (0.0,1.0))[1]) for a in assets])

    group_bounds = []
    for g, aset in st.session_state.get("group_map", {}).items():
        idxs = [assets.index(a) for a in aset if a in assets]
        gmin, gmax = st.session_state.get("group_bounds", {}).get(g, (0.0, 1.0))
        group_bounds.append((idxs, gmin, gmax))

    infeasible = False
    if np.sum(lb) > 1.0 + 1e-9 or np.sum(ub) < 1.0 - 1e-9:
        st.warning(f"Atenção: limites por ativo podem estar infactíveis (∑Min={np.sum(lb):.2f}, ∑Max={np.sum(ub):.2f}). Ajuste os sliders.")
        infeasible = True

    if objective == "Sharpe Máximo":
        w = None if infeasible else optimize_qp(mu_ann, Sigma_ann, None, rf, lb, ub, group_bounds, "Sharpe Máximo")
        if w is None:
            st.error("Falha ao otimizar Sharpe. Ajuste parâmetros/restrições.")
        else:
            metrics = portfolio_metrics(w, mu_ann, Sigma_ann)
            st.markdown("**Carteira de Tangência (Sharpe Máximo)**")
            dfw = pd.DataFrame({"Ativo": assets, "Peso": w}).set_index("Ativo")
            st.dataframe(dfw)
            st.write(f"Retorno (μ): {metrics['mu']:.2%} | Vol (σ): {metrics['sigma']:.2%} | Sharpe≈ {(metrics['mu']-rf)/max(metrics['sigma'],1e-12):.2f}")
            fig = plt.figure()
            plt.bar(assets, w)
            plt.title("Pesos — Carteira de Tangência")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_tangency.csv")
            st.session_state["last_weights"] = dfw
    else:
        w = None if infeasible else optimize_qp(mu_ann, Sigma_ann, None, rf, lb, ub, group_bounds, "Min-Var")
        if w is None:
            st.error("Falha ao otimizar Mín-Var. Ajuste as restrições.")
        else:
            metrics = portfolio_metrics(w, mu_ann, Sigma_ann)
            st.markdown("**Carteira de Mínima Variância**")
            dfw = pd.DataFrame({"Ativo": assets, "Peso": w}).set_index("Ativo")
            st.dataframe(dfw)
            st.write(f"Retorno (μ): {metrics['mu']:.2%} | Vol (σ): {metrics['sigma']:.2%}")
            fig = plt.figure()
            plt.bar(assets, w)
            plt.title("Pesos — Carteira Min-Var")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_minvar.csv")
            st.session_state["last_weights"] = dfw

with tab_frontier:
    st.subheader("4) Fronteira Eficiente (respeitando limites por ativo e por grupo)")
    prices = get_prices(uploaded)
    if prices is None:
        st.warning("Carregue dados na aba 'Dados' ou monte a base real na barra lateral.")
        st.stop()
    rets = to_returns(prices, method=ret_method)
    win_days = st.session_state.get("win_days", 504)
    rets = rets.tail(win_days)

    if mu_method.startswith("Média"):
        mu = rets.mean()
    else:
        mu = ewma_mean(rets, lam)

    if sigma_method == "Amostral":
        Sigma = rets.cov()
    elif sigma_method.startswith("EWMA"):
        Sigma = ewma_cov(rets, lam)
    else:
        Sigma = ledoit_wolf_cov(rets)

    mu_ann, Sigma_ann = annualize(mu, Sigma, periods_per_year=252)
    Sigma_ann = regularize_spd(Sigma_ann, eps=1e-8)

    if st.session_state.get("apply_class_rules", False):
        win_key = '1a' if st.session_state.get('win_days',504)==252 else ('2a' if st.session_state.get('win_days',504)==504 else '3a')
        mu_ann, Sigma_ann = apply_class_overrides_from_prices(get_prices(uploaded), mu_ann, Sigma_ann, win_key, "2020-01-01")

    if use_sigma_override:
        S = Sigma_ann.copy()
        for a, s in st.session_state.get("sigma_override", {}).items():
            if a in S.index:
                S = _apply_variance_override(S, a, float(s))
        Sigma_ann = regularize_spd(S, eps=1e-8)

    assets = list(mu_ann.index)
    br = st.session_state.get("bounds_ranges", {})
    lb = np.array([float(br.get(a, (0.0,1.0))[0]) for a in assets])
    ub = np.array([float(br.get(a, (0.0,1.0))[1]) for a in assets])
    group_bounds = []
    for g, aset in st.session_state.get("group_map", {}).items():
        idxs = [assets.index(a) for a in aset if a in assets]
        gmin, gmax = st.session_state.get("group_bounds", {}).get(g, (0.0, 1.0))
        group_bounds.append((idxs, gmin, gmax))

    frontier = compute_frontier(mu_ann, Sigma_ann, lb, ub, group_bounds, n_points=30)
    if len(frontier) == 0:
        st.error("Não foi possível construir a fronteira. Ajuste parâmetros/restrições.")
    else:
        sigmas = [p["sigma"] for p in frontier]
        mus = [p["mu"] for p in frontier]

        fig = plt.figure()
        plt.plot(sigmas, mus, marker="o", linestyle="-")
        plt.xlabel("Risco (σ)")
        plt.ylabel("Retorno (μ)")
        plt.title("Fronteira Eficiente")
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        buff = io.BytesIO()
        fig.savefig(buff, format="png", bbox_inches="tight")
        st.download_button("Baixar Fronteira (PNG)", data=buff.getvalue(), file_name="fronteira.png", mime="image/png")

        i_min = int(np.argmin(sigmas))
        minvar = frontier[i_min]
        i_tan = int(np.argmax([m/max(s,1e-12) for m, s in zip(mus, sigmas)]))
        tang = frontier[i_tan]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ponto de Mínima Variância (aprox.)**")
            st.write(f"μ={minvar['mu']:.2%} | σ={minvar['sigma']:.2%}")
        with c2:
            st.markdown("**Ponto de Tangência (aprox.)**")
            st.write(f"μ={tang['mu']:.2%} | σ={tang['sigma']:.2%} | Sharpe≈ {(tang['mu']/max(tang['sigma'],1e-12)):.2f}")

        options = {f"μ={p['mu']:.2%}, σ={p['sigma']:.2%}": p for p in frontier}
        sel = st.selectbox("Escolha um ponto para ver pesos", list(options.keys()))
        chosen = options[sel]
        dfw = pd.DataFrame({"Ativo": assets, "Peso": chosen["w"]}).set_index("Ativo")
        st.dataframe(dfw)
        st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_selected_frontier.csv")

with tab_report:
    st.subheader("5) Relatório consolidado / Exportações")
    prices = get_prices(uploaded)
    if prices is None:
        st.info("Monte uma base na aba Dados primeiro.")
        st.stop()

    bounds = st.session_state.get("bounds_ranges", {})
    cfg_info = {
        "janela_dias": st.session_state.get("win_days", 504),
        "apply_rules": st.session_state.get("apply_class_rules", False),
        "rf": rf,
        "mu_method": mu_method,
        "sigma_method": sigma_method,
        "groups": st.session_state.get("group_map", {}),
        "group_bounds": st.session_state.get("group_bounds", {}),
    }

    st.markdown("**Configuração ativa**")
    st.json(cfg_info)
    if "last_weights" in st.session_state:
        st.markdown("**Últimos pesos calculados**")
        st.dataframe(st.session_state["last_weights"])

    try:
        import xlsxwriter  # noqa
        excel_ok = True
    except Exception:
        excel_ok = False

    if st.button("Gerar Excel com parâmetros e pesos"):
        from io import BytesIO
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
            rets = to_returns(prices, method=ret_method).tail(st.session_state.get("win_days", 504))
            if mu_method.startswith("Média"):
                mu = rets.mean()
            else:
                mu = ewma_mean(rets, lam)
            if sigma_method == "Amostral":
                Sigma = rets.cov()
            elif sigma_method.startswith("EWMA"):
                Sigma = ewma_cov(rets, lam)
            else:
                Sigma = ledoit_wolf_cov(rets)
            mu_ann, Sigma_ann = annualize(mu, Sigma, 252)
            Sigma_ann = regularize_spd(Sigma_ann, 1e-8)
            if st.session_state.get("apply_class_rules", False):
                win_key = '1a' if st.session_state.get('win_days',504)==252 else ('2a' if st.session_state.get('win_days',504)==504 else '3a')
                mu_ann, Sigma_ann = apply_class_overrides_from_prices(prices, mu_ann, Sigma_ann, win_key, "2020-01-01")
            corr = pd.DataFrame(np.corrcoef(rets.values.T), index=rets.columns, columns=rets.columns)
            mu_ann.to_frame("mu").to_excel(xw, sheet_name="mu")
            Sigma_ann.to_excel(xw, sheet_name="Sigma")
            corr.to_excel(xw, sheet_name="Corr")
            if "last_weights" in st.session_state:
                st.session_state["last_weights"].to_excel(xw, sheet_name="Pesos")
            bd = pd.DataFrame.from_dict(bounds, orient="index", columns=["min","max"])
            bd.to_excel(xw, sheet_name="Bounds")
            pd.DataFrame(cfg_info).to_excel(xw, sheet_name="Config", index=False)
        st.download_button("Baixar Relatório.xlsx", data=bio.getvalue(), file_name="relatorio_portfolio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
_badge("Pronto para produção", "#10b981")
_badge("Shrinkage Σ (Ledoit-Wolf) disponível", "#6366F1")
_badge("Regras de classe aplicáveis", "#d97706")
_badge("Exportações Excel/PNG", "#0ea5e9")
_badge("Restrições por grupo", "#ef4444")
