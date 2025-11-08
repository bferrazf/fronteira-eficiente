# Fronteira Eficiente Pro+ (Diário)

App Streamlit para construção de fronteira eficiente com dados diários, regras de classe (RDMercado 30/70 Selic), Crédito Tradicional calibrado por IDA-DI (1a/2a/3a) e Crédito Estruturado (+1pp σ, +1pp μ vs Selic).

## Destaques
- Fontes: BCB/SGS e yfinance (cacheadas).
- Janela 1a/2a/3a, anualização por 252.
- μ editável, Σ shrinkage (Ledoit‑Wolf), heatmap.
- Limites por **ativo** com reset; **grupos** opcionais com limites agregados.
- Otimização: Sharpe Máximo, Mín-Var.
- Fronteira eficiente com export PNG; relatórios Excel (μ, Σ, ρ, pesos, bounds, config).

## Deploy rápido (Streamlit Cloud)
1. Faça fork do repositório com `app.py` e `requirements.txt`.
2. Em *share.streamlit.io*, selecione seu repo/branch e `app.py`.
3. Pronto.
