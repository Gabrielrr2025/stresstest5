python
import os
import io
import math
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Optional: yfinance only if needed
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

st.set_page_config(page_title="Finhealth • VaR & Estresse", page_icon="📉", layout="wide")

# ---------------- CSS (two-panel) ----------------
st.markdown("""
<style>
:root {
  --bg:#0b1220; --panel:#0f172a; --text:#e5e7eb; --muted:#94a3b8; --accent:#22d3ee; --border:rgba(255,255,255,.08);
}
html, body, [data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#0b1220,#0f172a);color:var(--text)}
.fin-card{background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:16px 18px;box-shadow:0 4px 24px rgba(0,0,0,.35)}
.kpi{display:flex;flex-direction:column;gap:6px;border:1px solid var(--border);border-radius:14px;padding:14px 16px;background:rgba(255,255,255,.03)}
.kpi .l{color:var(--muted);font-size:12px}.kpi .v{font-size:28px;font-weight:700}.kpi .s{color:var(--muted);font-size:12px}
.badge{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--border);padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.03);font-size:12px;color:var(--muted)}
hr.soft{border:none;height:1px;background:var(--border);margin:6px 0 14px}
.stTabs [data-baseweb="tab"]{background:rgba(255,255,255,.05);padding:10px 12px;border-radius:10px}
.stTabs [aria-selected="true"]{background:rgba(34,211,238,.12)!important;border:1px solid rgba(34,211,238,.26)}
.small{font-size:12px;color:var(--muted)}
.footer{margin-top:20px;text-align:center;color:var(--muted);font-size:13px}
</style>
""", unsafe_allow_html=True)

st.markdown("## 📉 Finhealth — VaR & Testes de Estresse")
st.markdown("<div class='badge'>Dados 100% reais — sempre a partir da sua alocação</div>", unsafe_allow_html=True)
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# ---------------- Utilities ----------------
def ensure_weights(df):
    df = df.copy()
    if 'Weight' not in df.columns:
        if {'Quantity','Price'}.issubset(df.columns):
            mv = df['Quantity'] * df['Price']
            total = mv.sum()
            df['Weight'] = mv / total if total != 0 else 0
        else:
            raise ValueError("Forneça 'Weight' ou ('Quantity' e 'Price').")
    df['Weight'] = df['Weight'].fillna(0.0)
    tot = df['Weight'].sum()
    if tot == 0:
        raise ValueError("A soma dos pesos é 0. Verifique a alocação.")
    if abs(tot - 1.0) > 1e-4:
        df['Weight'] = df['Weight'] / tot
    return df

def portfolio_returns(prices: pd.DataFrame, weights: pd.Series, ret_type="log"):
    prices = prices.ffill().dropna(how="all")
    rets = np.log(prices / prices.shift(1)) if ret_type == "log" else prices.pct_change()
    rets = rets.dropna(how="all")
    rets = rets.loc[:, weights.index.intersection(rets.columns)]
    W = weights.loc[rets.columns].values.reshape(-1, 1)
    port_ret = rets.values @ W
    port_ret = pd.Series(port_ret.flatten(), index=rets.index, name="Portfolio Return")
    return rets, port_ret

def quantile_var(series: pd.Series, cl=0.95):
    q = 1 - cl
    return float(-np.nanquantile(series.values, q))

def expected_shortfall(series: pd.Series, cl=0.95):
    q = 1 - cl
    thr = np.nanquantile(series.values, q)
    tail = series[series <= thr]
    return float(-tail.mean()) if len(tail) else 0.0

def var_variance_covariance(asset_rets: pd.DataFrame, weights: pd.Series, cl=0.95, horizon_days=1, ignore_mean=True):
    mu = asset_rets.mean().values
    cov = asset_rets.cov().values
    w = weights.loc[asset_rets.columns].values
    mu_p = float(w @ mu)
    var_p = float(w @ cov @ w.T)
    sd_p = math.sqrt(max(var_p, 0))
    # z ~ ppf without scipy
    from math import sqrt, pi, log
    # Approx quantile using inverse error func approximation
    # For simplicity, use numpy percent point function via normal sample approximation
    z = float(pd.Series(np.random.normal(size=500000)).quantile(cl))
    scale = math.sqrt(max(horizon_days, 1))
    v = (z * sd_p * scale) if ignore_mean else (z * sd_p * scale - mu_p * horizon_days)
    return float(v), float(sd_p), float(mu_p)

def var_monte_carlo(asset_rets: pd.DataFrame, weights: pd.Series, cl=0.95, horizon_days=1, n_sims=20000, seed=42):
    np.random.seed(seed)
    mu = asset_rets.mean().values
    cov = asset_rets.cov().values
    w = weights.loc[asset_rets.columns].values.reshape(-1, 1)
    cov = cov + np.eye(cov.shape[0]) * 1e-8
    L = np.linalg.cholesky(cov)
    sims = np.random.normal(size=(asset_rets.shape[1], n_sims))
    correlated = (L @ sims)
    sim_T = (mu.reshape(-1,1) + correlated) * math.sqrt(horizon_days)
    port_sim = (w.T @ sim_T).flatten()
    var = -np.nanquantile(port_sim, 1 - cl)
    es = -port_sim[port_sim <= np.quantile(port_sim, 1 - cl)].mean()
    return float(var), float(es)

def apply_stress(allocation: pd.DataFrame, scenarios: pd.DataFrame):
    scen_map = scenarios.set_index('Risk Factor')['Shock'].to_dict()
    betas = allocation.get('Beta', pd.Series([1.0]*len(allocation)))
    rf = allocation.get('RiskFactor', pd.Series(['Other']*len(allocation)))
    impacts = []
    for rf_name, w, b in zip(rf, allocation['Weight'], betas):
        shock = scen_map.get(rf_name, 0.0)
        impacts.append(w * b * shock)
    total_pct = float(np.nansum(impacts))
    detail = allocation[['Symbol','Weight']].copy()
    detail['Risk Factor'] = rf.values
    detail['Beta'] = betas.values
    detail['Applied Shock'] = [scen_map.get(x, 0.0) for x in rf.values]
    detail['Contribution (% NAV)'] = detail['Weight'] * detail['Beta'] * detail['Applied Shock']
    return total_pct, detail

def nice_pct(x, d=2): return f"{x*100:.{d}f}%"

@st.cache_data(show_spinner=False)
def yf_download_adjclose(tickers, start, end):
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance não está disponível. Faça upload dos preços ou adicione 'yfinance' ao requirements.")
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and 'Adj Close' in data.columns:
        pxs = data['Adj Close'].copy()
    else:
        pxs = data.copy()
    if isinstance(pxs, pd.Series):
        pxs = pxs.to_frame()
    pxs = pxs.dropna(how="all")
    pxs.columns = [c.upper() for c in pxs.columns]
    return pxs

# ---------------- Left Panel (controls) & Right Panel (results) ----------------
left, right = st.columns([1.05, 2.2])

with left:
    st.markdown("### 🧭 Painel de Controles")
    with st.expander("Identificação do Fundo", expanded=True):
        cnpj = st.text_input("CNPJ do Fundo", placeholder="00.000.000/0001-00")
        nome_fundo = st.text_input("Nome do Fundo", placeholder="Fundo XPTO")
        data_ref = st.date_input("Data de Referência", value=datetime.today())
        nav_input = st.number_input("Patrimônio Líquido (R$)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")

    with st.expander("Dados & Alocação (por Ativo)", expanded=True):
        st.caption("Forneça a alocação **real** por ativo. Campos esperados: Symbol, AssetName (opcional), Weight, Sector (opcional), RiskFactor (opcional), Beta (opcional).")
        sample_alloc = pd.DataFrame({
            "Symbol": ["PETR4.SA", "VALE3.SA", "USDBRL=X"],
            "AssetName": ["Petrobras PN", "Vale ON", "Dólar c/Real"],
            "Weight": [0.4, 0.4, 0.2],
            "Sector": ["Energia","Mineração","Câmbio"],
            "RiskFactor": ["Equity_BR","Equity_BR","FX_USDBRL"],
            "Beta": [1.0, 1.0, 1.0]
        })
        st.download_button("Baixar template de alocação (CSV)", data=sample_alloc.to_csv(index=False).encode("utf-8"),
                           file_name="allocation_template.csv", mime="text/csv")
        upl_alloc = st.file_uploader("Upload da alocação (.csv/.xlsx)", type=["csv","xlsx"])
        manual_alloc = st.toggle("Ou editar manualmente", value=False, help="Use para iniciar e depois exporte seu CSV.")
        if upl_alloc is not None:
            try:
                alloc_df = pd.read_csv(upl_alloc) if upl_alloc.name.endswith(".csv") else pd.read_excel(upl_alloc)
            except Exception as e:
                st.error(f"Erro ao ler alocação: {e}")
                alloc_df = None
        elif manual_alloc:
            alloc_df = st.data_editor(sample_alloc.copy(), num_rows="dynamic", use_container_width=True, key="alloc_editor")
        else:
            alloc_df = None

    with st.expander("Alocação por Setor (resumo para gráficos/relatório)", expanded=False):
        st.caption("Se sua alocação por ativo tiver coluna **Sector**, o resumo por setor é calculado automaticamente. Você também pode informar manualmente para fins de relatório.")
        if alloc_df is not None and 'Sector' in alloc_df.columns:
            sector_auto = alloc_df.groupby('Sector')['Weight'].sum().reset_index().rename(columns={'Weight':'Pct'})
            st.dataframe(sector_auto.assign(Pct=lambda d: (d['Pct']*100).round(2)), use_container_width=True)
            sector_manual = None
        else:
            sector_manual = st.data_editor(pd.DataFrame({'Sector':['Ações','Juros','Câmbio'], 'Pct':[0.0,0.0,0.0]}),
                                           num_rows="dynamic", use_container_width=True, key="sector_editor")
            st.caption("A soma deve dar 100%. Usado apenas para gráficos/relatório (não para o VaR).")

    with st.expander("Preços dos Ativos", expanded=True):
        price_source = st.radio("Origem dos preços", options=["Yahoo Finance (automático)","Upload de preços"],
                                help="Para controle total, faça upload do histórico de preços.")
        start = st.date_input("Início", value=datetime.today() - timedelta(days=365))
        end = st.date_input("Fim", value=datetime.today())
        if price_source == "Upload de preços":
            st.write("Formato esperado: **Date, Symbol, Price**")
            upl_px = st.file_uploader("Upload de preços (.csv/.xlsx)", type=["csv","xlsx"])
        else:
            upl_px = None
        run_load = st.button("📥 Carregar preços", use_container_width=True)

    with st.expander("Parâmetros de VaR", expanded=True):
        ret_type = st.selectbox("Tipo de retorno", ["log","simples"], help="Retorno logarítmico é aditivo no tempo.")
        cl = st.slider("Nível de confiança", 0.90, 0.999, 0.95, 0.001)
        horizon = st.number_input("Horizonte (dias)", min_value=1, max_value=60, value=1, step=1)
        method = st.segmented_control("Método", options=["Histórico","Variância-Covariância","Monte Carlo"],
                                      help="Histórico usa distribuição empírica; Var-Cov assume normalidade; Monte Carlo simula.")
        if method == "Histórico":
            lookback_days = st.slider("Janela (dias)", 60, 1500, 252)
        elif method == "Variância-Covariância":
            ignore_mean = st.toggle("Ignorar retorno médio (μ≈0)", value=True)
        else:
            n_sims = st.slider("Simulações", 2000, 100000, 20000, 1000)
            seed = st.number_input("Seed", 1, 999999, 42)

    with st.expander("Cenários de Estresse", expanded=True):
        default_scen = pd.DataFrame({
            "Risk Factor": ["Equity_BR","Rates_BR","FX_USDBRL","Other"],
            "Description": ["Queda de 15% no IBOVESPA","Alta de 200 bps na taxa de juros","Alta de 10% no USD/BRL","Choque customizado"],
            "Shock": [-0.15, -0.02, 0.10, 0.0]
        })
        scen = st.data_editor(default_scen, num_rows="dynamic", use_container_width=True, key="scen_editor")

    with st.expander("Relatório em Excel", expanded=True):
        st.caption("Opcional: envie um **template .xlsx**. O relatório será **anexado** como novas abas, preservando o template.")
        tpl_xlsx = st.file_uploader("Template Excel (opcional)", type=["xlsx"], key="tpl_upl")
        st.caption("Botão para baixar aparecerá no painel principal após o cálculo.")

    calc_now = st.button("▶️ Calcular VaR & Estresse", type="primary", use_container_width=True)

# ---------------- Loading prices ----------------
if run_load:
    if alloc_df is None or len(alloc_df)==0:
        right.warning("Carregue a **alocação** antes de buscar preços.")
    else:
        try:
            alloc_df = ensure_weights(alloc_df)
        except Exception as e:
            right.error(str(e))
            st.stop()
        symbols = [s.upper() for s in alloc_df['Symbol'].astype(str).tolist()]
        if price_source == "Yahoo Finance (automático)":
            if not YF_AVAILABLE:
                right.error("yfinance indisponível. Selecione **Upload de preços** ou adicione yfinance ao requirements.")
                st.stop()
            with right:
                with st.spinner("Baixando preços reais..."):
                    price_df = yf_download_adjclose(symbols, start, end)
            missing = set(symbols) - set(price_df.columns)
            if missing:
                right.warning(f"Sem preços para: {', '.join(sorted(missing))}. Verifique os tickers (ex.: .SA para B3).")
        else:
            if upl_px is None:
                right.error("Envie um arquivo de preços.")
                st.stop()
            try:
                px = pd.read_csv(upl_px) if upl_px.name.endswith(".csv") else pd.read_excel(upl_px)
                px['Date'] = pd.to_datetime(px['Date'])
                price_df = px.pivot(index='Date', columns='Symbol', values='Price').sort_index()
                price_df.columns = [c.upper() for c in price_df.columns]
            except Exception as e:
                right.error(f"Erro ao ler preços: {e}")
                st.stop()
        st.session_state['allocation'] = alloc_df
        st.session_state['prices'] = price_df
        st.session_state['start'] = pd.to_datetime(start)
        st.session_state['end'] = pd.to_datetime(end)
        right.success("✔️ Dados carregados.")

# ---------------- Calculations & Results ----------------
with right:
    st.markdown("### 📈 Resultados & Visualizações")
    if ('allocation' not in st.session_state) or ('prices' not in st.session_state):
        st.info("Carregue a alocação e os preços no painel esquerdo e clique em **Calcular**.")
    else:
        alloc = st.session_state['allocation'].copy()
        prices = st.session_state['prices'].copy()
        alloc['Symbol'] = alloc['Symbol'].str.upper()
        weights = ensure_weights(alloc).set_index('Symbol')['Weight']

        # sector summary
        if 'Sector' in alloc.columns:
            sector_df = alloc.groupby('Sector')['Weight'].sum().reset_index().rename(columns={'Weight':'Pct'})
        else:
            # fallback from manual (if provided and sums to ~100%)
            if 'sector_editor' in st.session_state:
                dfm = st.session_state['sector_editor']
                if isinstance(dfm, pd.DataFrame) and 'Sector' in dfm and 'Pct' in dfm and dfm['Pct'].sum() > 0:
                    sector_df = dfm.copy()
                    sector_df['Pct'] = sector_df['Pct'] / sector_df['Pct'].sum()
                else:
                    sector_df = None
            else:
                sector_df = None

        # If user hit calculate
        if calc_now:
            # Price window for historical
            px = prices.copy()
            if method == "Histórico":
                px = px.iloc[-lookback_days:]
            # Returns & portfolio
            asset_rets, port_rets = portfolio_returns(px, weights, ret_type=ret_type)

            # VaR + ES
            if method == "Histórico":
                result_var = quantile_var(port_rets, cl=cl) * math.sqrt(horizon)
                result_es = expected_shortfall(port_rets, cl=cl) * math.sqrt(horizon)
            elif method == "Variância-Covariância":
                v, sd, mu = var_variance_covariance(asset_rets, weights, cl=cl, horizon_days=horizon, ignore_mean=ignore_mean)
                result_var, result_es = v, None
            else:
                v, es = var_monte_carlo(asset_rets, weights, cl=cl, horizon_days=horizon, n_sims=n_sims, seed=seed)
                result_var, result_es = v, es

            # KPIs
            k1,k2,k3,k4 = st.columns(4)
            with k1:
                st.markdown(f"<div class='kpi'><div class='l'>VaR ({int(cl*100)}% / {horizon}d)</div><div class='v'>{nice_pct(result_var)}</div><div class='s'>Perda não excedida</div></div>", unsafe_allow_html=True)
            with k2:
                if result_es is not None:
                    st.markdown(f"<div class='kpi'><div class='l'>ES / CVaR</div><div class='v'>{nice_pct(result_es)}</div><div class='s'>Média das perdas na cauda</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='kpi'><div class='l'>ES / CVaR</div><div class='v'>—</div><div class='s'>Não disponível para este método</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown(f"<div class='kpi'><div class='l'>Observações</div><div class='v'>{px.shape[0]}</div><div class='s'>Dias de preço</div></div>", unsafe_allow_html=True)
            with k4:
                nav_loss = (result_var * nav_input) if nav_input else 0.0
                st.markdown(f"<div class='kpi'><div class='l'>Impacto em BRL</div><div class='v'>R$ {nav_loss:,.0f}</div><div class='s'>Com base no PL</div></div>", unsafe_allow_html=True)

            # Charts: sector allocation, returns histogram
            cA, cB = st.columns([1.2, 1])
            with cA:
                if sector_df is not None and len(sector_df)>0:
                    fig_alloc = px.pie(sector_df, values='Pct', names='Sector', title="Distribuição por Setor")
                    st.plotly_chart(fig_alloc, use_container_width=True)
                else:
                    st.caption("Informe a coluna **Sector** na alocação ou preencha 'Alocação por Setor' no painel esquerdo para ver o gráfico.")

            with cB:
                fig_hist = px.histogram(port_rets, x="Portfolio Return", nbins=60, marginal="violin", title="Distribuição de Retornos")
                st.plotly_chart(fig_hist, use_container_width=True)

            # Risk contributions
            try:
                cov = asset_rets.cov()
                w = weights.loc[cov.columns].values.reshape(-1,1)
                var_p = float(w.T @ cov.values @ w)
                contr = (w.flatten() * (cov.values @ w).flatten()) / max(var_p, 1e-12)
                contrib_df = pd.DataFrame({"Symbol": cov.columns, "Risk Contribution (%)": np.clip(contr, 0, None) * 100}).sort_values("Risk Contribution (%)", ascending=False)
                st.markdown("#### Contribuição de Risco por Ativo")
                st.dataframe(contrib_df, use_container_width=True, height=280)
            except Exception as e:
                st.info(f"Não foi possível calcular contribuição de risco: {e}")

            # Stress results (use mapping from allocation + scen)
            try:
                total_pct, detail = apply_stress(alloc, scen)
                st.markdown("#### Estresse — Resultado")
                st.write(f"Impacto total estimado: **{nice_pct(total_pct)}** do PL. " + (f"≈ R$ {total_pct * nav_input:,.0f}" if nav_input else ""))
                st.dataframe(detail.style.format({"Weight":"{:.2%}","Applied Shock":"{:.2%}","Contribution (% NAV)":"{:.2%}"}), use_container_width=True)
                stress_summary = pd.DataFrame(detail.groupby("Risk Factor")["Contribution (% NAV)"].sum()).rename(columns={"Contribution (% NAV)":"Impact (% NAV)"}).reset_index()
            except Exception as e:
                st.error(f"Erro no estresse: {e}")
                total_pct, detail, stress_summary = 0.0, pd.DataFrame(), pd.DataFrame()

            # ------------- Excel Export -------------
            # Prepare dataframes
            meta = pd.DataFrame({
                "Campo":["CNPJ","Fundo","Data de Referência","PL (BRL)","Confiança","Horizonte (d)","Método","Retorno"],
                "Valor":[cnpj, nome_fundo, str(data_ref), nav_input, cl, horizon, method, ret_type]
            })
            var_df = pd.DataFrame({
                "Métrica":["VaR","ES"],
                "Valor (%)":[result_var*100, (result_es*100 if result_es is not None else np.nan)],
                "Valor (BRL)":[(result_var*nav_input if nav_input else np.nan), (result_es*nav_input if (nav_input and result_es is not None) else np.nan)]
            })
            sector_out = None
            if sector_df is not None:
                sector_out = sector_df.copy()
                if 'Pct' in sector_out.columns: pass
                else: sector_out.rename(columns={"Weight":"Pct"}, inplace=True)

            # Build Excel in memory; if template provided, append sheets
            from openpyxl import load_workbook
            from pandas import ExcelWriter

            def build_excel_bytes(template_file):
                if template_file is not None:
                    # open the uploaded template and append sheets
                    in_mem = io.BytesIO(template_file.getbuffer())
                    try:
                        wb = load_workbook(in_mem)
                    except Exception as e:
                        st.warning(f"Template inválido: {e}. Gerando arquivo do zero.")
                        return build_excel_bytes(None)
                    # Re-save with appended sheets
                    out_mem = io.BytesIO()
                    with ExcelWriter(out_mem, engine="openpyxl") as writer:
                        writer.book = wb
                        meta.to_excel(writer, index=False, sheet_name="Inputs")
                        var_df.to_excel(writer, index=False, sheet_name="VaR")
                        contrib_df.to_excel(writer, index=False, sheet_name="Risk_Contrib")
                        detail.to_excel(writer, index=False, sheet_name="Stress_Detail")
                        stress_summary.to_excel(writer, index=False, sheet_name="Stress_Summary")
                        alloc.to_excel(writer, index=False, sheet_name="Allocation_Assets")
                        if sector_out is not None:
                            sector_out.to_excel(writer, index=False, sheet_name="Allocation_Sectors")
                        writer._save()
                    out_mem.seek(0)
                    return out_mem.getvalue()
                else:
                    out_mem = io.BytesIO()
                    with ExcelWriter(out_mem, engine="openpyxl") as writer:
                        meta.to_excel(writer, index=False, sheet_name="Inputs")
                        var_df.to_excel(writer, index=False, sheet_name="VaR")
                        if 'contrib_df' in locals() and isinstance(contrib_df, pd.DataFrame):
                            contrib_df.to_excel(writer, index=False, sheet_name="Risk_Contrib")
                        detail.to_excel(writer, index=False, sheet_name="Stress_Detail")
                        stress_summary.to_excel(writer, index=False, sheet_name="Stress_Summary")
                        alloc.to_excel(writer, index=False, sheet_name="Allocation_Assets")
                        if sector_out is not None:
                            sector_out.to_excel(writer, index=False, sheet_name="Allocation_Sectors")
                    out_mem.seek(0)
                    return out_mem.getvalue()

            xlsx_bytes = build_excel_bytes(tpl_xlsx)
            st.download_button("📥 Baixar Excel preenchido", data=xlsx_bytes, file_name=f"relatorio_var_estresse_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # If not calculated yet, still show a quick data preview
        if not calc_now:
            st.caption("Prévia de dados carregados")
            c1,c2 = st.columns([1.2,1])
            with c1:
                st.dataframe(alloc[['Symbol','Weight'] + ([c for c in ['AssetName','Sector','RiskFactor','Beta'] if c in alloc.columns])], use_container_width=True)
            with c2:
                st.dataframe(prices.tail(8), use_container_width=True)

st.markdown("<div class='footer'>Feito com ❤️ Finhealth</div>", unsafe_allow_html=True)
