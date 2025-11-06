
import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

# Optional libs
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


st.set_page_config(page_title="Fronteira Eficiente — Otimizador de Portfólio", layout="wide")
st.title("Fronteira Eficiente e Alocação Ótima (Cálculos Diários)")
st.caption("Dados reais (BCB + yfinance), retornos diários anualizados por 252, μ personalizado, limites por classe via sliders e fronteira eficiente.")

# =====================
# Utils
# =====================
def read_uploaded_csv(uploaded_file):
    """Safely read an uploaded CSV multiple times without EOF issues."""
    if uploaded_file is None:
        return None
    try:
        content = uploaded_file.getvalue()
        if content is None or len(content) == 0:
            st.error("O arquivo enviado está vazio. Envie um CSV com dados ou use o modo de dados reais.")
            return None
        return pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError:
        st.error("O arquivo enviado está vazio. Verifique o CSV.")
        return None
    except Exception as e:
        st.error(f"Falha ao ler o CSV: {e}")
        return None

def detect_format(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta se dados estão no formato longo (date,ticker,price) ou largo (colunas de ativos)."""
    cols_lower = [c.lower() for c in df.columns]
    if "date" in cols_lower and "ticker" in cols_lower and "price" in cols_lower:
        low = {c.lower(): c for c in df.columns}
        piv = df.rename(columns={low["date"]: "date", low["ticker"]: "ticker", low["price"]: "price"})
        piv["date"] = pd.to_datetime(piv["date"])
        wide = piv.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()
        return wide
    # Assume wide
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

def optimize_min_var(mu: pd.Series, Sigma: pd.DataFrame, target_ret: float, lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
    if not CVXPY_OK:
        return None
    n = len(mu)
    w = cp.Variable(n)
    Sigma_cp = cp.psd_wrap(Sigma.values)
    obj = cp.Minimize(cp.quad_form(w, Sigma_cp))
    cons = [cp.sum(w) == 1,
            mu.values @ w >= target_ret,
            w >= lb, w <= ub]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
        return None if w.value is None else np.array(w.value).ravel()
    except Exception:
        return None

def optimize_min_var_no_target(Sigma: pd.DataFrame, lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
    if not CVXPY_OK:
        return None
    n = Sigma.shape[0]
    w = cp.Variable(n)
    Sigma_cp = cp.psd_wrap(Sigma.values)
    obj = cp.Minimize(cp.quad_form(w, Sigma_cp))
    cons = [cp.sum(w) == 1, w >= lb, w <= ub]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
        return None if w.value is None else np.array(w.value).ravel()
    except Exception:
        return None

def optimize_max_sharpe(mu: pd.Series, Sigma: pd.DataFrame, rf: float, lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
    if not CVXPY_OK:
        return None
    mu_vals = mu.values
    mu_min = float(np.min(mu_vals))
    mu_max = float(np.max(mu_vals))
    targets = np.linspace(mu_min, mu_max, 25)
    best = None
    best_sharpe = -1e18
    for t in targets:
        w = optimize_min_var(mu, Sigma, t, lb, ub)
        if w is None:
            continue
        mu_p = float(mu_vals @ w)
        sig_p = float(np.sqrt(w @ Sigma.values @ w))
        if sig_p <= 0:
            continue
        sharpe = (mu_p - rf) / sig_p
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best = w
    return best

def compute_frontier(mu: pd.Series, Sigma: pd.DataFrame, lb: np.ndarray, ub: np.ndarray, n_points: int = 30) -> List[Dict]:
    if not CVXPY_OK:
        return []
    mu_vals = mu.values
    mu_min = float(np.min(mu_vals))
    mu_max = float(np.max(mu_vals))
    targets = np.linspace(mu_min, mu_max, n_points)
    out = []
    for t in targets:
        w = optimize_min_var(mu, Sigma, t, lb, ub)
        if w is None:
            continue
        m = portfolio_metrics(w, mu, Sigma)
        out.append({"target": float(t), "w": w, **m})
    return out

# =====================
# Providers — dados reais (online): BCB (SGS) + yfinance
# =====================
import datetime as _dt
import requests as _rq

def _fetch_yf_prices(ticker: str, start: str = "2018-01-01", end: Optional[str] = None) -> Optional[pd.Series]:
    """Baixa preço diário via yfinance com tratamento robusto."""
    try:
        import yfinance as yf
    except Exception:
        st.error("yfinance não está instalado. Adicione 'yfinance' no requirements.txt.")
        return None
    try:
        if end is None:
            end = _dt.date.today().isoformat()
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            st.warning(f"Sem dados para {ticker} via yfinance.")
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
            st.warning(f"Colunas de preço não encontradas para {ticker}.")
            return None
        s = pd.to_numeric(s, errors='coerce').dropna()
        if s.empty:
            st.warning(f"Série vazia para {ticker} após limpeza.")
            return None
        s.index = pd.to_datetime(s.index, errors='coerce')
        s = s[~s.index.isna()]
        s.name = ticker
        return s
    except Exception as e:
        st.warning(f"Falha ao baixar {ticker} via yfinance: {e}")
        return None

def _fetch_bcb_series(code: int, start: str = "2018-01-01") -> Optional[pd.Series]:
    """Baixa série SGS (JSON). Ex.: 1178 = Selic diária (BCB)."""
    try:
        start_br = pd.to_datetime(start).strftime("%d/%m/%Y")
        end_br = _dt.date.today().strftime("%d/%m/%Y")
        base = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        url = f"{base}?formato=json&dataInicial={start_br}&dataFinal={end_br}"
        r = _rq.get(url, timeout=30)
        try:
            r.raise_for_status()
        except Exception:
            # fallback últimos 5000 pontos
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
    except Exception as e:
        st.warning(f"Falha ao baixar série BCB {code}: {e}")
        return None

def _rate_series_to_price_index(rate_annual_pct: pd.Series) -> pd.Series:
    """Converte taxa anual (% a.a.) diária em índice de preços (base 100), assumindo 252 dias/ano."""
    if rate_annual_pct.empty:
        return rate_annual_pct
    r_daily = rate_annual_pct / 100.0 / 252.0
    idx = 100.0 * (1.0 + r_daily).cumprod()
    idx.name = rate_annual_pct.name
    return idx

def build_real_dataset(mapping: pd.DataFrame, start: str = "2018-01-01") -> pd.DataFrame:
    """Monta painel de preços diário a partir de um mapeamento (Classe, Fonte, Identificador).
       Retorna painel com colunas válidas; falhas reportadas em st.session_state['__build_failures__'].
    """
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
    # Fallbacks
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
# Sidebar — Inputs
# =====================
st.sidebar.header("1) Dados")
uploaded = st.sidebar.file_uploader("CSV de preços (formato largo) ou (date,ticker,price). Frequência: diária.", type=["csv"])

# Real data mode
st.sidebar.checkbox("Usar dados reais (online)", key="use_real_online", help="Baixa séries de mercado (yfinance/BCB) e monta automaticamente os preços.")

if st.session_state.get("use_real_online", False):
    st.sidebar.caption("Edite o mapeamento: Classe → Fonte ('yfinance' ou 'bcb') → Identificador (ticker ou código SGS).")
    default_map = pd.DataFrame({
        "Classe": ["SELIC", "IMA_B5", "RDMercado", "CreditoTrad", "CredEstrut", "AcoesBR", "AcoesEXT", "Multimerc"],
        "Fonte":  ["bcb",   "yfinance","yfinance",  "yfinance",    "yfinance",  "yfinance","yfinance","yfinance"],
        "Identificador": [
            "1178",
            "IMAB11.SA",
            "IRFM11.SA",
            "IRFM11.SA",
            "IRFM11.SA",
            "BOVA11.SA",
            "^GSPC",
            "BOVA11.SA"
        ]
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
            # Reports
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

st.sidebar.caption("Cálculos sempre **Diários**, com anualização por **252**.")
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

st.sidebar.header("4) Restrições")
st.sidebar.subheader("Limites por Classe (por ativo)")
st.sidebar.caption("Ajuste as faixas de alocação por classe com os sliders (0.00 a 1.00).")

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

# =====================
# Main Tabs
# =====================
tab_data, tab_est, tab_opt, tab_frontier = st.tabs(["Dados", "Parâmetros Est.", "Otimização", "Fronteira"])

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

    st.markdown("**Matriz de correlação (ρ)**")
    st.dataframe(corr)

    fig_corr = plt.figure()
    plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Matriz de Correlação — Heatmap")
    plt.tight_layout()
    st.pyplot(fig_corr)

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

    try:
        if st.session_state.get("enable_custom_mu", False) and st.session_state.get("custom_mu") is not None:
            mu_ann = st.session_state["custom_mu"].reindex(mu_ann.index).fillna(mu_ann)
    except Exception:
        pass

    assets = list(mu_ann.index)
    br = st.session_state.get("bounds_ranges", {})
    lb = np.array([float(br.get(a, (0.0,1.0))[0]) for a in assets])
    ub = np.array([float(br.get(a, (0.0,1.0))[1]) for a in assets])

    if np.sum(lb) > 1.0 + 1e-9 or np.sum(ub) < 1.0 - 1e-9:
        st.warning(f"Atenção: limites podem estar infactíveis (∑Min={np.sum(lb):.2f}, ∑Max={np.sum(ub):.2f}). Ajuste os sliders.")

    if objective == "Sharpe Máximo":
        w = optimize_max_sharpe(mu_ann, Sigma_ann, rf, lb, ub)
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
    else:
        w = optimize_min_var_no_target(Sigma_ann, lb, ub)
        if w is None:
            st.error("Falha ao otimizar Mín-Var (sem alvo). Ajuste as restrições.")
        else:
            metrics = portfolio_metrics(w, mu_ann, Sigma_ann)
            st.markdown("**Carteira de Mínima Variância (sem alvo)**")
            dfw = pd.DataFrame({"Ativo": assets, "Peso": w}).set_index("Ativo")
            st.dataframe(dfw)
            st.write(f"Retorno (μ): {metrics['mu']:.2%} | Vol (σ): {metrics['sigma']:.2%}")
            fig = plt.figure()
            plt.bar(assets, w)
            plt.title("Pesos — Carteira Min-Var (sem alvo)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_minvar.csv")

with tab_frontier:
    st.subheader("4) Fronteira Eficiente (respeitando limites por classe)")
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

    try:
        if st.session_state.get("enable_custom_mu", False) and st.session_state.get("custom_mu") is not None:
            mu_ann = st.session_state["custom_mu"].reindex(mu_ann.index).fillna(mu_ann)
    except Exception:
        pass

    assets = list(mu_ann.index)
    br = st.session_state.get("bounds_ranges", {})
    lb = np.array([float(br.get(a, (0.0,1.0))[0]) for a in assets])
    ub = np.array([float(br.get(a, (0.0,1.0))[1]) for a in assets])

    frontier = compute_frontier(mu_ann, Sigma_ann, lb, ub, n_points=30)
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

        i_min = int(np.argmin(sigmas))
        minvar = frontier[i_min]
        i_tan = int(np.argmax([(m - rf)/max(s,1e-12) for m, s in zip(mus, sigmas)]))
        tang = frontier[i_tan]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ponto de Mínima Variância (aprox.)**")
            st.write(f"μ={minvar['mu']:.2%} | σ={minvar['sigma']:.2%}")
        with c2:
            st.markdown("**Ponto de Tangência (aprox.)**")
            st.write(f"μ={tang['mu']:.2%} | σ={tang['sigma']:.2%} | Sharpe≈ {((tang['mu']-rf)/max(tang['sigma'],1e-12)):.2f}")

        options = {f"μ={p['mu']:.2%}, σ={p['sigma']:.2%}": p for p in frontier}
        sel = st.selectbox("Escolha um ponto para ver pesos", list(options.keys()))
        chosen = options[sel]
        dfw = pd.DataFrame({"Ativo": assets, "Peso": chosen["w"]}).set_index("Ativo")
        st.dataframe(dfw)
        st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_selected_frontier.csv")

st.markdown("---")
st.caption("Boas práticas: shrinkage (Ledoit-Wolf) p/ Σ; anualização por 252; janela de 1–3 anos; validar ∑Min ≤ 1 ≤ ∑Max.")
