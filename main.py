"""
EUR/USD Impact Analyzer
Aplicación para analizar el impacto del tipo de cambio EUR/USD en activos y carteras
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EUR/USD Impact Analyzer",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-card: #1a1a24;
        --accent-gold: #d4af37;
        --text-secondary: #a0a0b0;
        --border-color: #2a2a3a;
        --positive: #00d084;
        --negative: #ff6b6b;
    }
    
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d0d14 50%, #0a0a0f 100%); }
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    h1, h2, h3, p, span, label { font-family: 'DM Sans', sans-serif !important; }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.8rem !important;
        background: linear-gradient(135deg, #d4af37, #f4d03f, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p { color: #a0a0b0; letter-spacing: 0.08em; text-transform: uppercase; }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a24, #1f1f2e);
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card:hover { border-color: #d4af37; transform: translateY(-2px); }
    .metric-label { font-size: 0.7rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { font-family: 'Space Mono', monospace !important; font-size: 1.5rem; font-weight: 700; }
    .metric-value.positive { color: #00d084; }
    .metric-value.negative { color: #ff6b6b; }
    .metric-value.gold { color: #d4af37; }
    
    [data-testid="stSidebar"] { background: #12121a !important; border-right: 1px solid #2a2a3a; }
    
    .stTabs [data-baseweb="tab-list"] { background: #1a1a24; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #a0a0b0; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #d4af37, #f4d03f) !important; color: #0a0a0f !important; }
    
    .info-box {
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #a0a0b0;
    }
</style>
""", unsafe_allow_html=True)

# Colores para gráficos
COLORS = {
    'gold': '#d4af37', 'positive': '#00d084', 'negative': '#ff6b6b',
    'text': '#ffffff', 'text_secondary': '#a0a0b0', 'grid': '#2a2a3a',
    'palette': ['#d4af37', '#00d084', '#ff6b6b', '#4ecdc4', '#9b59b6', '#3498db', '#e67e22', '#1abc9c']
}

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.dropna()
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def calculate_cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices / prices.iloc[0]) * 100

def calculate_beta(returns: pd.DataFrame, asset: str, benchmark: str) -> dict:
    mask = returns[[asset, benchmark]].dropna().index
    y, x = returns.loc[mask, asset], returns.loc[mask, benchmark]
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    return {'beta': slope, 'alpha': intercept * 252, 'r_squared': r_value**2, 'p_value': p_value}

def calculate_rolling_correlation(returns: pd.DataFrame, base: str, window: int = 60) -> pd.DataFrame:
    correlations = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col != base:
            correlations[col] = returns[col].rolling(window=window).corr(returns[base])
    return correlations.dropna()

def base_layout(height=400):
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#ffffff'), height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor='#2a2a3a', zerolinecolor='#2a2a3a'),
        yaxis=dict(gridcolor='#2a2a3a', zerolinecolor='#2a2a3a')
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    
    # Período
    st.markdown("#### 📅 Período")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Inicio", value=datetime.now() - timedelta(days=365*3))
    end_date = c2.date_input("Fin", value=datetime.now())
    
    st.markdown("---")
    st.markdown("#### 📊 Activos")
    
    # Presets rápidos
    PRESETS = {
        "SPY": "S&P 500", "QQQ": "Nasdaq", "EZU": "Eurozone", "VGK": "Europe",
        "EWG": "Germany", "EEM": "EM", "GLD": "Gold", "TLT": "US Bonds"
    }
    
    preset_selection = st.multiselect(
        "Activos predefinidos:",
        options=list(PRESETS.keys()),
        default=["SPY", "EZU", "GLD"],
        format_func=lambda x: f"{x} ({PRESETS[x]})"
    )
    
    # Input manual
    custom_tickers = st.text_input(
        "Añadir tickers (separados por coma):",
        placeholder="AAPL, MSFT, TSLA...",
        help="Escribe cualquier ticker de Yahoo Finance"
    )
    
    # Combinar selección
    selected_assets = list(preset_selection)
    if custom_tickers:
        custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        selected_assets.extend([t for t in custom_list if t not in selected_assets])
    
    if selected_assets:
        st.caption(f"**Activos seleccionados:** {', '.join(selected_assets)}")
    
    st.markdown("---")
    st.markdown("#### 🔧 Parámetros")
    rolling_window = st.slider("Ventana Rolling", 20, 120, 60)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💱 EUR/USD Impact Analyzer</h1>
    <p>Impacto del tipo de cambio en activos y carteras</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

if not selected_assets:
    st.warning("⚠️ Selecciona o escribe al menos un activo")
    st.stop()

with st.spinner("Descargando datos..."):
    prices = get_data(["EURUSD=X"] + selected_assets, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if prices.empty:
    st.error("No se pudieron obtener datos. Verifica los tickers.")
    st.stop()

prices = prices.rename(columns={"EURUSD=X": "EUR/USD"})
# Filtrar activos que realmente se descargaron
selected_assets = [a for a in selected_assets if a in prices.columns]
returns = calculate_returns(prices)
cum_returns = calculate_cumulative_returns(prices)

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS EUR/USD
# ══════════════════════════════════════════════════════════════════════════════

eurusd_current = prices["EUR/USD"].iloc[-1]
eurusd_change = (prices["EUR/USD"].iloc[-1] / prices["EUR/USD"].iloc[0] - 1) * 100
eurusd_vol = returns["EUR/USD"].std() * np.sqrt(252) * 100

cols = st.columns(4)
for col, (label, value, cls) in zip(cols, [
    ("EUR/USD Actual", f"{eurusd_current:.4f}", "gold"),
    ("Cambio Período", f"{eurusd_change:+.2f}%", "positive" if eurusd_change >= 0 else "negative"),
    ("Volatilidad Anual", f"{eurusd_vol:.1f}%", "gold"),
    ("Días Analizados", f"{len(prices)}", "gold")
]):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {cls}">{value}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(["📈 Retornos Acumulados", "⚡ Impacto EUR/USD", "🔄 Correlación Rolling", "💼 Simulador Cartera"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: RETORNOS ACUMULADOS
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### Retornos Acumulados (Base 100)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns["EUR/USD"], mode='lines', name='EUR/USD',
                             line=dict(color=COLORS['gold'], width=3)))
    
    for i, col in enumerate(selected_assets):
        if col in cum_returns.columns:
            fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[col], mode='lines', name=col,
                                     line=dict(color=COLORS['palette'][(i+1) % len(COLORS['palette'])], width=2)))
    
    fig.add_hline(y=100, line_dash="dash", line_color=COLORS['grid'], opacity=0.5)
    fig.update_layout(**base_layout(500), hovermode='x unified',
                      title=dict(text="Evolución de Precios (Base 100)", font=dict(size=16, color=COLORS['gold'])),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla
    st.markdown("#### Rendimiento del Período")
    perf_data = []
    for col in cum_returns.columns:
        ret = (cum_returns[col].iloc[-1] / 100 - 1) * 100
        vol = returns[col].std() * np.sqrt(252) * 100
        perf_data.append({'Activo': col, 'Rendimiento': f"{ret:+.2f}%", 'Volatilidad': f"{vol:.1f}%", 'Sharpe': f"{ret/vol:.2f}" if vol > 0 else "N/A"})
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: IMPACTO EUR/USD
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Sensibilidad al EUR/USD")
    
    impact_data = []
    for asset in selected_assets:
        if asset in returns.columns:
            beta_info = calculate_beta(returns, asset, "EUR/USD")
            corr = returns[asset].corr(returns["EUR/USD"])
            impact_data.append({
                'Activo': asset, 'Beta': beta_info['beta'], 'Correlación': corr,
                'R²': beta_info['r_squared'], 'P-Value': beta_info['p_value'],
                'Sig.': '✓' if beta_info['p_value'] < 0.05 else '✗'
            })
    
    impact_df = pd.DataFrame(impact_data).sort_values('Beta', ascending=False)
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        colors_bar = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in impact_df['Beta']]
        fig = go.Figure(go.Bar(x=impact_df['Activo'], y=impact_df['Beta'], marker_color=colors_bar,
                               text=[f"{b:.2f}" for b in impact_df['Beta']], textposition='outside'))
        fig.add_hline(y=0, line_color=COLORS['grid'])
        fig.update_layout(**base_layout(400), title=dict(text="Beta vs EUR/USD", font=dict(size=16, color=COLORS['gold'])))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        colors_corr = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in impact_df['Correlación']]
        fig2 = go.Figure(go.Bar(y=impact_df['Activo'], x=impact_df['Correlación'], orientation='h',
                                marker_color=colors_corr, text=[f"{c:.2f}" for c in impact_df['Correlación']], textposition='outside'))
        fig2.add_vline(x=0, line_color=COLORS['grid'])
        layout2 = base_layout(400)
        layout2['xaxis']['range'] = [-1, 1]
        fig2.update_layout(**layout2, title=dict(text="Correlación vs EUR/USD", font=dict(size=16, color=COLORS['gold'])))
        st.plotly_chart(fig2, use_container_width=True)
    
    # Tabla
    st.markdown("#### Detalle")
    disp = impact_df.copy()
    disp['Beta'] = disp['Beta'].apply(lambda x: f"{x:.3f}")
    disp['Correlación'] = disp['Correlación'].apply(lambda x: f"{x:.3f}")
    disp['R²'] = disp['R²'].apply(lambda x: f"{x:.2%}")
    disp['P-Value'] = disp['P-Value'].apply(lambda x: f"{x:.4f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)
    
    st.markdown('<div class="info-box"><strong>💡</strong> Beta indica movimiento del activo por cada 1% de EUR/USD. ✓ = significativo (p<0.05).</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CORRELACIÓN ROLLING
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown(f"### Correlación Rolling ({rolling_window} días)")
    
    rolling_corr = calculate_rolling_correlation(returns, "EUR/USD", rolling_window)
    
    fig = go.Figure()
    for i, col in enumerate(rolling_corr.columns):
        fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr[col], mode='lines', name=col,
                                 line=dict(color=COLORS['palette'][i % len(COLORS['palette'])], width=2)))
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['text_secondary'], opacity=0.5)
    fig.add_hrect(y0=0.5, y1=1, fillcolor=COLORS['positive'], opacity=0.05, line_width=0)
    fig.add_hrect(y0=-1, y1=-0.5, fillcolor=COLORS['negative'], opacity=0.05, line_width=0)
    
    layout3 = base_layout(450)
    layout3['yaxis']['range'] = [-1, 1]
    layout3['yaxis']['dtick'] = 0.25
    fig.update_layout(**layout3, title=dict(text="Evolución de Correlaciones", font=dict(size=16, color=COLORS['gold'])),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Estadísticas")
    st.dataframe(pd.DataFrame({
        'Media': rolling_corr.mean(), 'Std': rolling_corr.std(),
        'Mín': rolling_corr.min(), 'Máx': rolling_corr.max(), 'Actual': rolling_corr.iloc[-1]
    }).round(3), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SIMULADOR CARTERA
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### Simulador de Impacto en Cartera")
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown("#### Composición")
        portfolio_assets = st.multiselect("Activos:", [a for a in selected_assets if a in returns.columns],
                                          default=[a for a in selected_assets if a in returns.columns][:3], key="port")
        
        weights = {}
        if portfolio_assets:
            st.markdown("**Pesos (%):**")
            for asset in portfolio_assets:
                weights[asset] = st.number_input(asset, 0, 100, 100//len(portfolio_assets), key=f"w_{asset}") / 100
            
            total = sum(weights.values()) * 100
            if abs(total - 100) > 0.1:
                st.warning(f"⚠️ Pesos suman {total:.0f}%")
    
    with c2:
        if portfolio_assets:
            port_returns = sum(returns[a] * weights[a] for a in portfolio_assets)
            mask = returns["EUR/USD"].dropna().index
            port_ret, fx_ret = port_returns.loc[mask], returns.loc[mask, "EUR/USD"]
            
            port_corr = port_ret.corr(fx_ret)
            slope, _, r_value, p_value, _ = stats.linregress(fx_ret, port_ret)
            
            st.markdown("#### Sensibilidad Cartera")
            cols = st.columns(4)
            for col, (lbl, val) in zip(cols, [("Corr", f"{port_corr:.3f}"), ("Beta", f"{slope:.3f}"),
                                               ("R²", f"{r_value**2:.2%}"), ("P-Val", f"{p_value:.4f}")]):
                col.metric(lbl, val)
            
            st.markdown("#### Escenarios")
            scenarios = [-10, -5, -2, 2, 5, 10]
            impacts = [slope * s for s in scenarios]
            fig = go.Figure(go.Bar(x=[f"{s:+.0f}%" for s in scenarios], y=impacts,
                                   marker_color=[COLORS['negative'] if s < 0 else COLORS['positive'] for s in scenarios],
                                   text=[f"{i:+.2f}%" for i in impacts], textposition='outside'))
            fig.update_layout(**base_layout(280), title=dict(text="Impacto por Escenario EUR/USD", font=dict(size=14, color=COLORS['gold'])))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selecciona activos")

# Footer
st.markdown("---")
st.markdown(f'<div style="text-align:center;color:#a0a0b0;font-size:0.8rem;">EUR/USD Impact Analyzer • Yahoo Finance • {start_date:%d/%m/%Y} - {end_date:%d/%m/%Y}<br>⚠️ Solo fines educativos</div>', unsafe_allow_html=True)
