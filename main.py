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
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --accent-gold: #d4af37;
        --accent-gold-light: #f4d03f;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --border-color: #2a2a3a;
        --positive: #00d084;
        --negative: #ff6b6b;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0d0d14 50%, var(--bg-primary) 100%);
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
    }
    
    p, span, label, .stMarkdown {
        font-family: 'DM Sans', sans-serif !important;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.8rem !important;
        background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 50%, var(--accent-gold) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, #1f1f2e 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-gold);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-value.positive { color: var(--positive); }
    .metric-value.negative { color: var(--negative); }
    .metric-value.gold { color: var(--accent-gold); }
    
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%) !important;
        color: var(--bg-primary) !important;
    }
    
    .info-box {
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Colores
COLORS = {
    'gold': '#d4af37',
    'gold_light': '#f4d03f',
    'positive': '#00d084',
    'negative': '#ff6b6b',
    'bg': '#0a0a0f',
    'card': '#1a1a24',
    'text': '#ffffff',
    'text_secondary': '#a0a0b0',
    'grid': '#2a2a3a',
    'palette': ['#d4af37', '#00d084', '#ff6b6b', '#4ecdc4', '#9b59b6', '#3498db', '#e67e22', '#1abc9c']
}

def get_plotly_layout(height=400):
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color=COLORS['text']),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        legend=dict(bgcolor='rgba(26,26,36,0.8)', bordercolor=COLORS['grid'])
    )

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga datos de yfinance"""
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            multi_level_index=False
        )['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        return data.dropna()
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return pd.DataFrame()

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos"""
    return np.log(prices / prices.shift(1)).dropna()

def calculate_cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos acumulados (base 100)"""
    return (prices / prices.iloc[0]) * 100

def calculate_beta(returns: pd.DataFrame, asset: str, benchmark: str) -> dict:
    """Calcula beta, alpha y R²"""
    mask = returns[[asset, benchmark]].dropna().index
    y = returns.loc[mask, asset]
    x = returns.loc[mask, benchmark]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        'beta': slope,
        'alpha': intercept * 252,
        'r_squared': r_value**2,
        'p_value': p_value
    }

def calculate_rolling_correlation(returns: pd.DataFrame, base: str, window: int = 60) -> pd.DataFrame:
    """Correlación rolling"""
    correlations = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col != base:
            correlations[col] = returns[col].rolling(window=window).corr(returns[base])
    return correlations.dropna()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    
    # Período
    st.markdown("#### 📅 Período")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Inicio", value=datetime.now() - timedelta(days=365*3))
    with col2:
        end_date = st.date_input("Fin", value=datetime.now())
    
    st.markdown("---")
    
    # Activos - Selección simple
    st.markdown("#### 📊 Activos")
    
    ASSETS = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "EZU": "Eurozone",
        "VGK": "Europe",
        "EWG": "Germany",
        "EEM": "Emerging Mkts",
        "GLD": "Gold",
        "SLV": "Silver",
        "USO": "Oil",
        "TLT": "US Long Bonds",
        "IEF": "US 7-10Y Bonds",
        "HYG": "High Yield"
    }
    
    selected_assets = st.multiselect(
        "Seleccionar activos:",
        options=list(ASSETS.keys()),
        default=["SPY", "EZU", "GLD", "TLT"],
        format_func=lambda x: f"{x} - {ASSETS[x]}"
    )
    
    st.markdown("---")
    
    # Parámetros
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

if len(selected_assets) == 0:
    st.warning("⚠️ Selecciona al menos un activo en el panel lateral")
    st.stop()

all_tickers = ["EURUSD=X"] + selected_assets

with st.spinner("Descargando datos..."):
    prices = get_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if prices.empty:
    st.error("No se pudieron obtener datos.")
    st.stop()

prices = prices.rename(columns={"EURUSD=X": "EUR/USD"})
returns = calculate_returns(prices)
cum_returns = calculate_cumulative_returns(prices)

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS EUR/USD
# ══════════════════════════════════════════════════════════════════════════════

eurusd_current = prices["EUR/USD"].iloc[-1]
eurusd_change = (prices["EUR/USD"].iloc[-1] / prices["EUR/USD"].iloc[0] - 1) * 100
eurusd_vol = returns["EUR/USD"].std() * np.sqrt(252) * 100

cols = st.columns(4)
metrics = [
    ("EUR/USD Actual", f"{eurusd_current:.4f}", "gold"),
    ("Cambio Período", f"{eurusd_change:+.2f}%", "positive" if eurusd_change >= 0 else "negative"),
    ("Volatilidad Anual", f"{eurusd_vol:.1f}%", "gold"),
    ("Días Analizados", f"{len(prices)}", "gold")
]

for col, (label, value, color_class) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

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
    
    # EUR/USD primero con línea más gruesa
    fig.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns["EUR/USD"],
        mode='lines',
        name='EUR/USD',
        line=dict(color=COLORS['gold'], width=3),
        hovertemplate="EUR/USD<br>%{x}<br>%{y:.2f}<extra></extra>"
    ))
    
    # Resto de activos
    for i, col in enumerate(selected_assets):
        if col in cum_returns.columns:
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[col],
                mode='lines',
                name=col,
                line=dict(color=COLORS['palette'][(i+1) % len(COLORS['palette'])], width=2),
                hovertemplate=f"{col}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>"
            ))
    
    fig.add_hline(y=100, line_dash="dash", line_color=COLORS['grid'], opacity=0.5)
    
    fig.update_layout(
        **get_plotly_layout(500),
        title=dict(text="Evolución de Precios (Base 100)", font=dict(size=16, color=COLORS['gold'])),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Tabla de rendimientos
    st.markdown("#### Rendimiento del Período")
    
    perf_data = []
    for col in cum_returns.columns:
        ret = (cum_returns[col].iloc[-1] / 100 - 1) * 100
        vol = returns[col].std() * np.sqrt(252) * 100
        sharpe = (ret / vol) if vol > 0 else 0
        perf_data.append({
            'Activo': col,
            'Rendimiento': f"{ret:+.2f}%",
            'Volatilidad': f"{vol:.1f}%",
            'Sharpe': f"{sharpe:.2f}"
        })
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, hide_index=True, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: IMPACTO EUR/USD
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Sensibilidad al EUR/USD")
    
    # Calcular betas y correlaciones
    impact_data = []
    for asset in selected_assets:
        if asset in returns.columns:
            beta_info = calculate_beta(returns, asset, "EUR/USD")
            corr = returns[asset].corr(returns["EUR/USD"])
            impact_data.append({
                'Activo': asset,
                'Beta': beta_info['beta'],
                'Correlación': corr,
                'R²': beta_info['r_squared'],
                'P-Value': beta_info['p_value'],
                'Significativo': '✓' if beta_info['p_value'] < 0.05 else '✗'
            })
    
    impact_df = pd.DataFrame(impact_data).sort_values('Beta', ascending=False)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Gráfico de barras de Beta
        colors_bar = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in impact_df['Beta']]
        
        fig = go.Figure(data=go.Bar(
            x=impact_df['Activo'],
            y=impact_df['Beta'],
            marker=dict(color=colors_bar),
            text=[f"{b:.2f}" for b in impact_df['Beta']],
            textposition='outside',
            textfont=dict(color=COLORS['text']),
            hovertemplate="<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>"
        ))
        
        fig.add_hline(y=0, line_color=COLORS['grid'])
        
        fig.update_layout(
            **get_plotly_layout(400),
            title=dict(text="Beta vs EUR/USD", font=dict(size=16, color=COLORS['gold']))
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Gráfico de correlación
        colors_corr = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in impact_df['Correlación']]
        
        fig2 = go.Figure(data=go.Bar(
            y=impact_df['Activo'],
            x=impact_df['Correlación'],
            orientation='h',
            marker=dict(color=colors_corr),
            text=[f"{c:.2f}" for c in impact_df['Correlación']],
            textposition='outside',
            textfont=dict(color=COLORS['text']),
            hovertemplate="<b>%{y}</b><br>Corr: %{x:.3f}<extra></extra>"
        ))
        
        fig2.add_vline(x=0, line_color=COLORS['grid'])
        
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans, sans-serif', color=COLORS['text']),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], range=[-1, 1]),
            yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
            title=dict(text="Correlación vs EUR/USD", font=dict(size=16, color=COLORS['gold']))
        )
        
        st.plotly_chart(fig2, width="stretch")
    
    # Tabla detallada
    st.markdown("#### Detalle de Impacto")
    
    display_df = impact_df.copy()
    display_df['Beta'] = display_df['Beta'].apply(lambda x: f"{x:.3f}")
    display_df['Correlación'] = display_df['Correlación'].apply(lambda x: f"{x:.3f}")
    display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.2%}")
    display_df['P-Value'] = display_df['P-Value'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, hide_index=True, width="stretch")
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Interpretación:</strong> Beta indica cuánto se mueve el activo por cada 1% de movimiento en EUR/USD. 
        Un beta de 0.5 significa que si EUR/USD sube 1%, el activo sube ~0.5%. ✓ indica significancia estadística (p<0.05).
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CORRELACIÓN ROLLING
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown(f"### Correlación Rolling ({rolling_window} días)")
    
    rolling_corr = calculate_rolling_correlation(returns, "EUR/USD", rolling_window)
    
    fig = go.Figure()
    
    for i, col in enumerate(rolling_corr.columns):
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr[col],
            mode='lines',
            name=col,
            line=dict(color=COLORS['palette'][i % len(COLORS['palette'])], width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x}}<br>Corr: %{{y:.3f}}<extra></extra>"
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['text_secondary'], opacity=0.5)
    fig.add_hrect(y0=0.5, y1=1, fillcolor=COLORS['positive'], opacity=0.05, line_width=0)
    fig.add_hrect(y0=-1, y1=-0.5, fillcolor=COLORS['negative'], opacity=0.05, line_width=0)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', color=COLORS['text']),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], range=[-1, 1], dtick=0.25),
        title=dict(text="Evolución de Correlaciones", font=dict(size=16, color=COLORS['gold'])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(26,26,36,0.8)')
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Stats
    st.markdown("#### Estadísticas")
    stats_df = pd.DataFrame({
        'Media': rolling_corr.mean(),
        'Std': rolling_corr.std(),
        'Mín': rolling_corr.min(),
        'Máx': rolling_corr.max(),
        'Actual': rolling_corr.iloc[-1]
    }).round(3)
    
    st.dataframe(stats_df, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SIMULADOR CARTERA
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### Simulador de Impacto en Cartera")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("#### Composición")
        
        portfolio_assets = st.multiselect(
            "Activos:",
            options=[a for a in selected_assets if a in returns.columns],
            default=[a for a in selected_assets if a in returns.columns][:3],
            key="portfolio_select"
        )
        
        if portfolio_assets:
            weights = {}
            st.markdown("**Pesos (%):**")
            
            for asset in portfolio_assets:
                w = st.number_input(f"{asset}", min_value=0, max_value=100, value=100//len(portfolio_assets), key=f"pw_{asset}")
                weights[asset] = w / 100
            
            total_weight = sum(weights.values()) * 100
            if abs(total_weight - 100) > 0.1:
                st.warning(f"⚠️ Pesos suman {total_weight:.0f}% (deberían sumar 100%)")
    
    with col2:
        if portfolio_assets and len(portfolio_assets) >= 1:
            # Calcular retornos de cartera
            port_returns = sum(returns[a] * weights[a] for a in portfolio_assets)
            port_cum = (1 + port_returns).cumprod() * 100
            
            # Sensibilidad
            mask = returns["EUR/USD"].dropna().index
            port_ret = port_returns.loc[mask]
            fx_ret = returns.loc[mask, "EUR/USD"]
            
            port_corr = port_ret.corr(fx_ret)
            slope, intercept, r_value, p_value, _ = stats.linregress(fx_ret, port_ret)
            
            # Métricas
            st.markdown("#### Sensibilidad de la Cartera")
            
            cols = st.columns(4)
            port_metrics = [
                ("Correlación", f"{port_corr:.3f}"),
                ("Beta FX", f"{slope:.3f}"),
                ("R²", f"{r_value**2:.2%}"),
                ("P-Value", f"{p_value:.4f}")
            ]
            
            for col, (label, value) in zip(cols, port_metrics):
                with col:
                    st.metric(label, value)
            
            # Escenarios
            st.markdown("#### Escenarios de Impacto")
            
            scenarios = [-10, -5, -2, 2, 5, 10]
            impacts = [slope * s for s in scenarios]
            colors_scenario = [COLORS['negative'] if s < 0 else COLORS['positive'] for s in scenarios]
            
            fig = go.Figure(data=go.Bar(
                x=[f"{s:+.0f}%" for s in scenarios],
                y=impacts,
                marker=dict(color=colors_scenario),
                text=[f"{i:+.2f}%" for i in impacts],
                textposition='outside',
                textfont=dict(color=COLORS['text']),
                hovertemplate="EUR/USD: %{x}<br>Impacto: %{y:.2f}%<extra></extra>"
            ))
            
            fig.update_layout(
                **get_plotly_layout(300),
                title=dict(text="Impacto Estimado por Escenario EUR/USD", font=dict(size=14, color=COLORS['gold'])),
                xaxis_title="Movimiento EUR/USD",
                yaxis_title="Impacto Cartera (%)"
            )
            
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Selecciona activos para la cartera")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['text_secondary']}; padding: 1rem 0; font-size: 0.8rem;">
    EUR/USD Impact Analyzer • Datos: Yahoo Finance • 
    {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}<br>
    ⚠️ Solo para fines educativos. No constituye asesoramiento financiero.
</div>
""", unsafe_allow_html=True)
