
import streamlit as st
import pandas as pd
import numpy as np
import io
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

# === Optional imports for advanced estimation/optimization ===
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


st.set_page_config(page_title="Fronteira Eficiente — Otimizador de Portfólio",
                   layout="wide")

st.title("Fronteira Eficiente e Alocação Ótima")
st.caption("Carregue seus dados, estime parâmetros e gere carteiras ótimas com restrições.")

# =====================
# Utils
# =====================
def read_uploaded_csv(uploaded_file):
    """Safely read an uploaded CSV multiple times without EOF issues.
    Uses getvalue() -> BytesIO so pointer position doesn't matter.
    """
    if uploaded_file is None:
        return None
    try:
        content = uploaded_file.getvalue()
        if content is None or len(content) == 0:
            st.error("O arquivo enviado está vazio. Envie um CSV com dados ou use o exemplo.")
            return None
        import io as _io
        return pd.read_csv(_io.BytesIO(content))
    except pd.errors.EmptyDataError:
        st.error("O arquivo enviado está vazio. Verifique o CSV.")
        return None
    except Exception as e:
        st.error(f"Falha ao ler o CSV: {e}")
        return None

def get_prices(uploaded_file):
    """Return a prices DataFrame from uploaded file or from sample stored in session_state."""
    if uploaded_file is not None:
        dfu = read_uploaded_csv(uploaded_file)
        if dfu is None:
            return None
        return detect_format(dfu)
    # Fallback to sample if loaded
    if st.session_state.get('__sample_loaded__') and 'sample_prices' in st.session_state:
        return st.session_state['sample_prices']
    return None


def to_returns(df_prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Convert price-level DataFrame to returns DataFrame (drop first NaN row)."""
    if method == "pct":
        rets = df_prices.pct_change().dropna(how="all")
    else:
        rets = np.log(df_prices/df_prices.shift(1)).dropna(how="all")
    return rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")

def annualize(mu: pd.Series, sigma: pd.DataFrame, periods_per_year: int) -> Tuple[pd.Series, pd.DataFrame]:
    """Annualize mean returns and covariance matrix."""
    mu_ann = mu * periods_per_year
    Sigma_ann = sigma * periods_per_year
    return mu_ann, Sigma_ann

def ewma_mean(rets: pd.DataFrame, lam: float) -> pd.Series:
    """Exponentially weighted mean."""
    weights = np.array([lam**i for i in range(len(rets))])[::-1]
    weights = weights/weights.sum()
    return pd.Series(np.dot(weights, rets.values), index=rets.columns)

def ewma_cov(rets: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Exponentially weighted covariance (RiskMetrics-style)."""
    # de-mean with EWMA mean
    mu_ew = ewma_mean(rets, lam)
    X = rets - mu_ew
    Sigma = np.zeros((X.shape[1], X.shape[1]))
    w_sum = 0.0
    w = 1.0
    for t in range(X.shape[0]-1, -1, -1):
        x = X.iloc[t].values.reshape(-1, 1)
        Sigma = lam*Sigma + (1-lam)*(x @ x.T)
        w_sum = lam*w_sum + (1-lam)
    Sigma = Sigma / max(w_sum, 1e-12)
    return pd.DataFrame(Sigma, index=rets.columns, columns=rets.columns)

def ledoit_wolf_cov(rets: pd.DataFrame) -> pd.DataFrame:
    if not SKLEARN_OK:
        st.warning("scikit-learn não disponível. Caindo para covariância amostral.")
        return rets.cov()
    lw = LedoitWolf().fit(rets.values)
    return pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)

def regularize_spd(Sigma: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Ensure SPD by adding eps*I if needed."""
    # Try Cholesky
    try:
        np.linalg.cholesky(Sigma.values)
        return Sigma
    except np.linalg.LinAlgError:
        pass
    # add diagonal until SPD
    A = Sigma.values.copy()
    boost = eps
    for _ in range(12):
        try:
            np.linalg.cholesky(A + boost*np.eye(A.shape[0]))
            return pd.DataFrame(A + boost*np.eye(A.shape[0]), index=Sigma.index, columns=Sigma.columns)
        except np.linalg.LinAlgError:
            boost *= 10.0
    # final fallback
    return pd.DataFrame(A + boost*np.eye(A.shape[0]), index=Sigma.index, columns=Sigma.columns)




def optimize_min_var(mu: pd.Series, Sigma: pd.DataFrame, target_ret: float, lb: np.ndarray, ub: np.ndarray, classes: Optional[List[ClassConstraint]] = None) -> Optional[np.ndarray]:
    """Min-variance with return >= target and sum(weights)=1, lb<=w<=ub."""
    if not CVXPY_OK:
        return None
    n = len(mu)
    w = cp.Variable(n)
    Sigma_cp = cp.psd_wrap(Sigma.values)
    obj = cp.Minimize(cp.quad_form(w, Sigma_cp))
    cons = [cp.sum(w) == 1,
            mu.values @ w >= target_ret,
            w >= lb, w <= ub]
    # Class-level constraints
    if classes:
        for c in classes:
            cons += [cp.sum(w[c.indices]) >= c.min_w, cp.sum(w[c.indices]) <= c.max_w]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
        return None if w.value is None else np.array(w.value).ravel()
    except Exception:
        return None

def optimize_max_sharpe(mu: pd.Series, Sigma: pd.DataFrame, rf: float, lb: np.ndarray, ub: np.ndarray, classes: Optional[List[ClassConstraint]] = None) -> Optional[np.ndarray]:
    """Maximize Sharpe via convex transform: minimize variance for unit excess return scale."""
    if not CVXPY_OK:
        return None
    # Sweep targets and pick max Sharpe under constraints
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

def portfolio_metrics(w: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame) -> Dict[str, float]:
    mu_p = float(mu.values @ w)
    sig_p = float(np.sqrt(w @ Sigma.values @ w))
    return {"mu": mu_p, "sigma": sig_p}

def compute_frontier(mu: pd.Series, Sigma: pd.DataFrame, lb: np.ndarray, ub: np.ndarray, n_points: int = 25, classes: Optional[List[ClassConstraint]] = None) -> List[Dict]:
    """Sweep target returns between min and max feasible mean and solve min-var."""
    if not CVXPY_OK:
        return []
    # Feasible range from box portfolio extremes
    mu_sorted = np.sort(mu.values)
    mu_min = float(mu_sorted[0])
    mu_max = float(mu_sorted[-1])
    targets = np.linspace(mu_min, mu_max, n_points)
    out = []
    for t in targets:
        w = optimize_min_var(mu, Sigma, t, lb, ub)
        if w is None:
            continue
        m = portfolio_metrics(w, mu, Sigma)
        out.append({"target": float(t), "w": w, **m})
    return out

def detect_format(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to detect whether data is wide prices or long format."""
    cols = [c.lower() for c in df.columns]
    if "date" in cols:
        # long format expected columns: date, ticker, price
        low = {c.lower(): c for c in df.columns}
        if {"date", "ticker", "price"}.issubset(set(cols)):
            piv = df.rename(columns={low["date"]: "date", low["ticker"]: "ticker", low["price"]: "price"})
            piv["date"] = pd.to_datetime(piv["date"])
            wide = piv.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()
            return wide
    # assume wide
    df = df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    # try parse index as dates
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df

# =====================
# Sidebar — Inputs
# =====================
st.sidebar.header("1) Dados")
uploaded = st.sidebar.file_uploader("CSV com preços (formato largo) ou colunas: date,ticker,price", type=["csv"])

st.sidebar.caption('A frequência de cálculo agora é **Diária**, com anualização por 252 dias úteis.')
per_year = 252

ret_method = st.sidebar.selectbox("Retorno", ["Log", "Percentual"], index=0)
ret_method = "log" if ret_method == "Log" else "pct"

st.sidebar.header("2) Estimação")
mu_method = st.sidebar.selectbox("Retorno esperado (μ)", ["Média histórica", "EWMA (média)"], index=0)
sigma_method = st.sidebar.selectbox("Covariância (Σ)", ["Amostral", "EWMA (cov)", "Ledoit-Wolf (shrinkage)"], index=2)
lam = st.sidebar.slider("λ (EWMA)", min_value=0.80, max_value=0.99, value=0.94, step=0.01)

st.sidebar.header("3) Otimização")
objective = st.sidebar.selectbox("Objetivo", ["Sharpe Máximo", "Mínima Variância + Retorno-alvo"], index=0)
rf = st.sidebar.number_input("Taxa livre de risco (anualizada, ex.: 0.05 = 5%)", min_value=-0.5, max_value=0.5, value=0.0, step=0.005)
target_ret = st.sidebar.number_input("Retorno-alvo anualizado (se aplicável)", min_value=-0.5, max_value=1.0, value=0.10, step=0.01)

st.sidebar.header("4) Restrições")

# ---- Limites por Classe (simples) ----
st.sidebar.subheader("Limites por Classe (por ativo)")
st.sidebar.caption("Defina limites mínimos e máximos para cada classe (coluna). Ex.: 0.20 = 20%.")
_prices_preview = get_prices(uploaded)
if _prices_preview is None:
    st.sidebar.info("Carregue dados em 'Dados' ou clique em 'Usar exemplo' para ver/editar os limites.")
else:
    assets_list = list(_prices_preview.columns)
    if "bounds_table" not in st.session_state or set(st.session_state.get("bounds_assets", [])) != set(assets_list):
        st.session_state["bounds_assets"] = assets_list
        st.session_state["bounds_table"] = pd.DataFrame({
            "Classe": assets_list,
            "Min": [0.0]*len(assets_list),
            "Max": [1.0]*len(assets_list),
        })
    st.session_state["bounds_table"] = st.sidebar.data_editor(
        st.session_state["bounds_table"],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Classe": st.column_config.TextColumn("Classe", disabled=True),
            "Min": st.column_config.NumberColumn("Min", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            "Max": st.column_config.NumberColumn("Max", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
        }
    )
# ---- Class constraints editor ----
st.sidebar.subheader("Limites por Classe")
st.sidebar.caption("Defina a classe de cada ativo e os limites de alocação por classe (fração do portfólio).")
def default_class_for(name: str) -> str:
    n = name.lower()
    if "selic" in n: return "Renda Fixa - Selic"
    if "ima" in n: return "Renda Fixa - IPCA"
    if "credito" in n and "estrut" in n: return "Crédito Estruturado"
    if "credito" in n: return "Crédito Tradicional"
    if "acoesbr" in n or ("acoes" in n and "ext" not in n): return "Ações Brasil"
    if "acoesext" in n or "ext" in n: return "Ações Exterior"
    if "rdmercado" in n: return "Risco de Mercado"
    if "multi" in n: return "Multimercado"
    return "Outros"

if "class_table" not in st.session_state and uploaded is not None:
    # Build initial table from uploaded data columns
    df_tmp = detect_format(read_uploaded_csv(uploaded))
    assets_list = list(df_tmp.columns)
    st.session_state["class_table"] = pd.DataFrame({
        "Ativo": assets_list,
        "Classe": [default_class_for(a) for a in assets_list],
        "Min": [0.0]*len(assets_list),
        "Max": [1.0]*len(assets_list),
    })
elif "class_table" not in st.session_state:
    # Fallback: placeholder until data is loaded
    st.session_state["class_table"] = pd.DataFrame(columns=["Ativo","Classe","Min","Max"])

st.sidebar.caption("Edite os campos abaixo; 'Min' e 'Max' são frações (ex.: 0.2 = 20%).")
st.session_state["class_table"] = st.sidebar.data_editor(
    st.session_state["class_table"],
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ativo": st.column_config.TextColumn("Ativo"),
        "Classe": st.column_config.TextColumn("Classe"),
        "Min": st.column_config.NumberColumn("Min", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
        "Max": st.column_config.NumberColumn("Max", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
    }
)

long_only = st.sidebar.checkbox("Long only (sem short)", value=True)
lb_default = 0.0 if long_only else -1.0
ub_default = 1.0
st.sidebar.caption("Limites globais por ativo (podem ser refinados após carregar os dados).")
lb_global = st.sidebar.slider("Lower bound (cada ativo)", min_value=-1.0, max_value=1.0, value=lb_default, step=0.05)
ub_global = st.sidebar.slider("Upper bound (cada ativo)", min_value=0.0, max_value=2.0, value=ub_default, step=0.05)

# =====================
# Main Area
# =====================
tab_data, tab_est, tab_opt, tab_frontier = st.tabs(["Dados", "Parâmetros Est.", "Otimização", "Fronteira"])

with tab_data:
    st.subheader("1) Carregamento e pré-processamento")
    if uploaded is None:
        c_ex1, c_ex2 = st.columns([2,1])
        with c_ex1:
            st.info("Faça o upload de um CSV de preços para começar. Ou clique em **Usar exemplo** para carregar o dataset dummy incluído.")
        with c_ex2:
            if st.button("Usar exemplo"):
                try:
                    sample = pd.read_csv("dummy_prices_daily.csv")
                    prices = detect_format(sample)
                    st.session_state['__sample_loaded__'] = True
                st.session_state['sample_prices'] = prices
                except Exception as e:
                    st.error(f"Não foi possível carregar o exemplo: {e}")
        if 'prices' not in locals():
            st.stop()
    else:
        raw = read_uploaded_csv(uploaded)
        prices = detect_format(raw)
        st.write("Amostra de preços (topo):")
        st.dataframe(prices.head())

        # returns
        returns = to_returns(prices, method=ret_method)
        st.write("Amostra de retornos:")
        st.dataframe(returns.head())

with tab_est:
    st.subheader("2) Estimação de parâmetros (μ, Σ, ρ)")
    if uploaded is None:
        st.warning("Carregue dados na aba 'Dados'.")
    else:
        returns = to_returns(detect_format(read_uploaded_csv(uploaded)), method=ret_method)
        # μ
        if mu_method.startswith("Média"):
            mu = returns.mean()
        else:
            mu = ewma_mean(returns, lam)

        # Σ
        if sigma_method == "Amostral":
            Sigma = returns.cov()
        elif sigma_method.startswith("EWMA"):
            Sigma = ewma_cov(returns, lam)
        else:
            Sigma = ledoit_wolf_cov(returns)

        mu_ann, Sigma_ann = annualize(mu, Sigma, per_year)
        Sigma_ann = regularize_spd(Sigma_ann, eps=1e-8)

        # apply custom mu if enabled
        try:
            if st.session_state.get('custom_mu') is not None and st.session_state.get('enable_custom_mu', False):
                # align indices safely
                mu_ann = st.session_state['custom_mu'].reindex(mu_ann.index).fillna(mu_ann)
        except Exception:
            pass

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

        
        
        # ---- Custom Expected Returns (annualized) ----
        st.markdown("### Retornos esperados personalizados (anualizados)")
        st.caption("Edite os valores diretamente na tabela abaixo. Ao marcar a opção, a otimização usará estes valores.")
        # Initialize session state
        if "custom_mu" not in st.session_state or set(st.session_state.get("custom_mu_index", [])) != set(mu_ann.index):
            st.session_state["custom_mu"] = mu_ann.copy()
            st.session_state["custom_mu_index"] = list(mu_ann.index)

        enable_custom = st.checkbox("Usar retornos esperados personalizados", value=False, help="Quando marcado, a otimização usará os valores personalizados abaixo.")
        st.session_state['enable_custom_mu'] = enable_custom

        df_edit_src = st.session_state["custom_mu"].reindex(mu_ann.index).fillna(mu_ann)
        df_edit = pd.DataFrame({"Ativo": df_edit_src.index, "mu_ann": df_edit_src.values})
        df_edit = st.data_editor(
            df_edit,
            hide_index=True,
            column_config={
                "mu_ann": st.column_config.NumberColumn("mu_ann", min_value=-1.0, max_value=1.0, step=0.0001, format="%.4f"),
                "Ativo": st.column_config.TextColumn("Ativo", disabled=True),
            },
            use_container_width=True,
        )
        # Persist edits
        try:
            st.session_state["custom_mu"] = pd.Series(df_edit["mu_ann"].values, index=df_edit["Ativo"].values)
        except Exception as e:
            st.warning(f"Não foi possível aplicar edições de μ: {e}")


        csv_mu = mu_ann.to_csv().encode()
        csv_sigma = Sigma_ann.to_csv().encode()
        st.download_button("Baixar μ (CSV)", data=csv_mu, file_name="mu_annualized.csv")
        st.download_button("Baixar Σ (CSV)", data=csv_sigma, file_name="sigma_annualized.csv")

with tab_opt:
    st.subheader("3) Otimização e carteiras recomendadas")
    if uploaded is None:
        st.warning("Carregue dados na aba 'Dados'.")
    elif not CVXPY_OK:
        st.error("cvxpy não está disponível no ambiente. Instale as dependências e rode localmente.")
    else:
        prices = detect_format(read_uploaded_csv(uploaded))
        rets = to_returns(prices, method=ret_method)

        # estimate again per selections
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

        mu_ann, Sigma_ann = annualize(mu, Sigma, per_year)
        Sigma_ann = regularize_spd(Sigma_ann, eps=1e-8)

        # apply custom mu if enabled
        try:
            if st.session_state.get('custom_mu') is not None and st.session_state.get('enable_custom_mu', False):
                # align indices safely
                mu_ann = st.session_state['custom_mu'].reindex(mu_ann.index).fillna(mu_ann)
        except Exception:
            pass

        assets = list(mu_ann.index)
        n = len(assets)
        bt = st.session_state.get('bounds_table', pd.DataFrame(columns=['Classe','Min','Max']))
        bt = bt.set_index('Classe') if 'Classe' in bt.columns else bt
        lb = np.array([float(bt.loc[a,'Min']) if a in bt.index else 0.0 for a in assets])
        ub = np.array([float(bt.loc[a,'Max']) if a in bt.index else 1.0 for a in assets])
        class_df = st.session_state.get("class_table", pd.DataFrame(columns=['Ativo','Classe','Min','Max']))
        class_df = class_df[class_df['Ativo'].isin(assets)].copy()
        classes = build_class_constraints(assets, class_df)

        # Build class constraints
        class_df = st.session_state.get("class_table", pd.DataFrame(columns=['Ativo','Classe','Min','Max']))
        # Ensure Ativo column aligns with current assets
        class_df = class_df[class_df['Ativo'].isin(assets)].copy()
        classes = build_class_constraints(assets, class_df)


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
                # bar chart
                fig = plt.figure()
                plt.bar(assets, w)
                plt.title("Pesos — Carteira de Tangência")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

                # export
                st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_tangency.csv")

        else:
            w = optimize_min_var(mu_ann, Sigma_ann, target_ret, lb, ub)
            if w is None:
                st.error("Falha ao otimizar Mín-Var com retorno-alvo. Ajuste o alvo ou restrições.")
            else:
                metrics = portfolio_metrics(w, mu_ann, Sigma_ann)
                st.markdown("**Carteira de Mínima Variância (dado retorno-alvo)**")
                dfw = pd.DataFrame({"Ativo": assets, "Peso": w}).set_index("Ativo")
                st.dataframe(dfw)
                st.write(f"Retorno (μ): {metrics['mu']:.2%} | Vol (σ): {metrics['sigma']:.2%}")
                fig = plt.figure()
                plt.bar(assets, w)
                plt.title("Pesos — Carteira Min-Var (μ alvo)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_minvar.csv")

with tab_frontier:
    st.subheader("4) Fronteira Eficiente")
    if uploaded is None:
        st.warning("Carregue dados na aba 'Dados'.")
    elif not CVXPY_OK:
        st.error("cvxpy não está disponível no ambiente. Instale as dependências e rode localmente.")
    else:
        prices = detect_format(read_uploaded_csv(uploaded))
        rets = to_returns(prices, method=ret_method)

        # estimate again per selections
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

        mu_ann, Sigma_ann = annualize(mu, Sigma, per_year)
        Sigma_ann = regularize_spd(Sigma_ann, eps=1e-8)

        # apply custom mu if enabled
        try:
            if st.session_state.get('custom_mu') is not None and st.session_state.get('enable_custom_mu', False):
                # align indices safely
                mu_ann = st.session_state['custom_mu'].reindex(mu_ann.index).fillna(mu_ann)
        except Exception:
            pass

        assets = list(mu_ann.index)
        n = len(assets)
        bt = st.session_state.get('bounds_table', pd.DataFrame(columns=['Classe','Min','Max']))
        bt = bt.set_index('Classe') if 'Classe' in bt.columns else bt
        lb = np.array([float(bt.loc[a,'Min']) if a in bt.index else 0.0 for a in assets])
        ub = np.array([float(bt.loc[a,'Max']) if a in bt.index else 1.0 for a in assets])
        class_df = st.session_state.get("class_table", pd.DataFrame(columns=['Ativo','Classe','Min','Max']))
        class_df = class_df[class_df['Ativo'].isin(assets)].copy()
        classes = build_class_constraints(assets, class_df)

        # Build class constraints
        class_df = st.session_state.get("class_table", pd.DataFrame(columns=['Ativo','Classe','Min','Max']))
        # Ensure Ativo column aligns with current assets
        class_df = class_df[class_df['Ativo'].isin(assets)].copy()
        classes = build_class_constraints(assets, class_df)


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

            # pick min-var and tangency (approx tangency among frontier points given rf)
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

            # allow selection to view weights
            options = {f"μ={p['mu']:.2%}, σ={p['sigma']:.2%}": p for p in frontier}
            sel = st.selectbox("Escolha um ponto para ver pesos", list(options.keys()))
            chosen = options[sel]
            dfw = pd.DataFrame({"Ativo": assets, "Peso": chosen["w"]}).set_index("Ativo")
            st.dataframe(dfw)
            st.download_button("Baixar pesos (CSV)", data=dfw.to_csv().encode(), file_name="weights_selected_frontier.csv")

st.markdown("---")
st.caption("Boas práticas: usar shrinkage (Ledoit-Wolf) para Σ, anualizar pela frequência correta, e validar resultados OOS.")
