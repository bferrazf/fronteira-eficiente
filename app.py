
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
    diag = np.diag_indices_from(A)
    boost = eps
    for _ in range(12):
        try:
            np.linalg.cholesky(A + boost*np.eye(A.shape[0]))
            return pd.DataFrame(A + boost*np.eye(A.shape[0]), index=Sigma.index, columns=Sigma.columns)
        except np.linalg.LinAlgError:
            boost *= 10.0
    # final fallback
    return pd.DataFrame(A + boost*np.eye(A.shape[0]), index=Sigma.index, columns=Sigma.columns)

def optimize_min_var(mu: pd.Series, Sigma: pd.DataFrame, target_ret: float, lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
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
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
        return None if w.value is None else np.array(w.value).ravel()
    except Exception:
        return None

def optimize_max_sharpe(mu: pd.Series, Sigma: pd.DataFrame, rf: float, lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
    """Maximize Sharpe via convex transform: minimize variance for unit excess return scale."""
    if not CVXPY_OK:
        return None
    n = len(mu)
    mu_ex = mu.values - rf
    # Scale weights by k to get unit expected excess return: mu_ex' w = 1
    w = cp.Variable(n)
    Sigma_cp = cp.psd_wrap(Sigma.values)
    obj = cp.Minimize(cp.quad_form(w, Sigma_cp))
    cons = [mu_ex @ w == 1, w >= 0]  # enforce positive scale; sum-to-one will be applied after rescale
    # Add box bounds by converting back after normalization: we approximate by also bounding here
    cons += [w >= 0]  # ensure non-negativity for feasibility; we'll project to [lb,ub] after rescale if needed
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
        if w.value is None:
            return None
        w_raw = np.array(w.value).ravel()
        # Rescale to sum to 1 while respecting bounds with a simple projection
        w_pos = np.maximum(w_raw, 0)
        if w_pos.sum() <= 0:
            return None
        w_norm = w_pos / w_pos.sum()
        # Project to [lb, ub] and renormalize
        w_proj = np.minimum(np.maximum(w_norm, lb), ub)
        if w_proj.sum() <= 0:
            return None
        return w_proj / w_proj.sum()
    except Exception:
        return None

def portfolio_metrics(w: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame) -> Dict[str, float]:
    mu_p = float(mu.values @ w)
    sig_p = float(np.sqrt(w @ Sigma.values @ w))
    return {"mu": mu_p, "sigma": sig_p}

def compute_frontier(mu: pd.Series, Sigma: pd.DataFrame, lb: np.ndarray, ub: np.ndarray, n_points: int = 25) -> List[Dict]:
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

freq = st.sidebar.selectbox("Frequência dos dados", ["Mensal", "Semanal", "Diário"], index=0)
periods_map = {"Diário": 252, "Semanal": 52, "Mensal": 12}
per_year = periods_map[freq]

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
        st.info("Faça o upload de um CSV de preços para começar. Um exemplo de arquivo dummy está disponível no repositório ou na página.")
    else:
        raw = pd.read_csv(uploaded)
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
        returns = to_returns(detect_format(pd.read_csv(uploaded)), method=ret_method)
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
        prices = detect_format(pd.read_csv(uploaded))
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

        assets = list(mu_ann.index)
        n = len(assets)
        lb = np.array([lb_global]*n)
        ub = np.array([ub_global]*n)

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
        prices = detect_format(pd.read_csv(uploaded))
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

        assets = list(mu_ann.index)
        n = len(assets)
        lb = np.array([lb_global]*n)
        ub = np.array([ub_global]*n)

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
