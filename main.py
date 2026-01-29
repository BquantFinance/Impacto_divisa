"""
EUR/USD Impact Analyzer
Aplicación premium para analizar el impacto del tipo de cambio EUR/USD en distintos activos y carteras
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN Y ESTILOS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EUR/USD Impact Analyzer",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Premium - Dark Mode con acentos dorados
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
        letter-spacing: -0.02em;
    }
    
    p, span, label, .stMarkdown {
        font-family: 'DM Sans', sans-serif !important;
    }
    
    /* Header principal */
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 3.5rem !important;
        background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 50%, var(--accent-gold) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    
    /* Cards métricas */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, #1f1f2e 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .metric-card:hover {
        border-color: var(--accent-gold);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(212, 175, 55, 0.15);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-value.positive { color: var(--positive); }
    .metric-value.negative { color: var(--negative); }
    .metric-value.gold { color: var(--accent-gold); }
    
    /* Sección headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .section-header h2 {
        font-size: 1.5rem !important;
        color: var(--text-primary);
        margin: 0;
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--accent-gold) !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%) !important;
        color: var(--bg-primary) !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Selectbox y inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stSlider > div > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Tooltip info */
    .info-box {
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    
    /* Animación de entrada */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease forwards;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE COLORES PLOTLY
# ══════════════════════════════════════════════════════════════════════════════

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

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color=COLORS['text']),
    xaxis=dict(
        gridcolor=COLORS['grid'],
        zerolinecolor=COLORS['grid'],
        tickfont=dict(color=COLORS['text_secondary'])
    ),
    yaxis=dict(
        gridcolor=COLORS['grid'],
        zerolinecolor=COLORS['grid'],
        tickfont=dict(color=COLORS['text_secondary'])
    ),
    legend=dict(
        bgcolor='rgba(26,26,36,0.8)',
        bordercolor=COLORS['grid'],
        font=dict(color=COLORS['text_secondary'])
    ),
    margin=dict(l=20, r=20, t=40, b=20)
)

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga datos de yfinance con manejo de errores"""
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

def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    """Calcula retornos logarítmicos o simples"""
    if method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()

def calculate_rolling_correlation(returns: pd.DataFrame, base: str, window: int = 60) -> pd.DataFrame:
    """Calcula correlación rolling contra un activo base"""
    correlations = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col != base:
            correlations[col] = returns[col].rolling(window=window).corr(returns[base])
    return correlations.dropna()

def calculate_beta(returns: pd.DataFrame, asset: str, benchmark: str) -> dict:
    """Calcula beta y R² de un activo contra un benchmark"""
    mask = returns[[asset, benchmark]].dropna().index
    y = returns.loc[mask, asset]
    x = returns.loc[mask, benchmark]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        'beta': slope,
        'alpha': intercept * 252,  # Anualizado
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

def calculate_portfolio_sensitivity(returns: pd.DataFrame, weights: dict, fx: str) -> dict:
    """Calcula la sensibilidad de una cartera al FX"""
    portfolio_returns = sum(returns[asset] * weight for asset, weight in weights.items())
    
    mask = returns[fx].dropna().index
    port_ret = portfolio_returns.loc[mask]
    fx_ret = returns.loc[mask, fx]
    
    corr = port_ret.corr(fx_ret)
    beta_info = stats.linregress(fx_ret, port_ret)
    
    return {
        'correlation': corr,
        'beta': beta_info.slope,
        'r_squared': beta_info.rvalue**2,
        'volatility_contribution': abs(beta_info.slope) * fx_ret.std() * np.sqrt(252)
    }

# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header animate-in">
    <h1>💱 EUR/USD Impact Analyzer</h1>
    <p>Análisis del impacto del tipo de cambio en activos y carteras</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # Período de análisis
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Inicio",
            value=datetime.now() - timedelta(days=365*3),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "Fin",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    st.markdown("---")
    st.markdown("### 📊 Activos a Analizar")
    
    # Activos predefinidos por categoría
    ASSET_CATEGORIES = {
        "🇺🇸 US Equities": {
            "SPY": "S&P 500 ETF",
            "QQQ": "Nasdaq 100 ETF",
            "IWM": "Russell 2000 ETF",
            "DIA": "Dow Jones ETF"
        },
        "🇪🇺 EU Equities": {
            "EZU": "Eurozone ETF",
            "VGK": "Europe ETF",
            "EWG": "Germany ETF",
            "EWQ": "France ETF"
        },
        "🌍 Emergentes": {
            "EEM": "Emerging Markets ETF",
            "EWZ": "Brazil ETF",
            "FXI": "China Large Cap"
        },
        "📦 Commodities": {
            "GLD": "Gold ETF",
            "SLV": "Silver ETF",
            "USO": "Oil ETF",
            "DBA": "Agriculture ETF"
        },
        "💵 Renta Fija": {
            "TLT": "US Long Bonds",
            "IEF": "US 7-10Y Bonds",
            "LQD": "Investment Grade Corp",
            "HYG": "High Yield Corp"
        }
    }
    
    selected_assets = []
    for category, assets in ASSET_CATEGORIES.items():
        with st.expander(category, expanded=True):
            for ticker, name in assets.items():
                if st.checkbox(f"{ticker} - {name}", value=ticker in ["SPY", "EZU", "GLD", "TLT"]):
                    selected_assets.append(ticker)
    
    st.markdown("---")
    st.markdown("### 🔧 Parámetros")
    
    rolling_window = st.slider("Ventana Rolling (días)", 20, 120, 60)
    return_method = st.selectbox("Tipo de Retorno", ["log", "simple"], index=0)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

if len(selected_assets) == 0:
    st.warning("⚠️ Selecciona al menos un activo en el panel lateral")
    st.stop()

# Añadir EUR/USD a la lista
all_tickers = ["EURUSD=X"] + selected_assets

with st.spinner("Descargando datos de mercado..."):
    prices = get_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if prices.empty:
    st.error("No se pudieron obtener datos. Verifica la conexión y los tickers.")
    st.stop()

# Renombrar EUR/USD para claridad
prices = prices.rename(columns={"EURUSD=X": "EUR/USD"})
returns = calculate_returns(prices, return_method)

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="section-header">
    <div class="section-icon">📈</div>
    <h2>Resumen EUR/USD</h2>
</div>
""", unsafe_allow_html=True)

# Métricas del EUR/USD
eurusd_current = prices["EUR/USD"].iloc[-1]
eurusd_change = (prices["EUR/USD"].iloc[-1] / prices["EUR/USD"].iloc[0] - 1) * 100
eurusd_vol = returns["EUR/USD"].std() * np.sqrt(252) * 100
eurusd_max = prices["EUR/USD"].max()
eurusd_min = prices["EUR/USD"].min()

cols = st.columns(5)

metrics = [
    ("Cotización Actual", f"{eurusd_current:.4f}", "gold"),
    ("Cambio Período", f"{eurusd_change:+.2f}%", "positive" if eurusd_change >= 0 else "negative"),
    ("Volatilidad Anual", f"{eurusd_vol:.1f}%", "gold"),
    ("Máximo", f"{eurusd_max:.4f}", "positive"),
    ("Mínimo", f"{eurusd_min:.4f}", "negative")
]

for col, (label, value, color_class) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(["📊 Correlaciones", "📉 Análisis Beta", "🔄 Rolling Analysis", "💼 Portfolio Impact"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: CORRELACIONES
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### Matriz de Correlación con EUR/USD")
    
    # Calcular correlaciones
    corr_matrix = returns.corr()
    eurusd_corr = corr_matrix["EUR/USD"].drop("EUR/USD").sort_values()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Heatmap de correlaciones
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[
                [0, COLORS['negative']],
                [0.5, COLORS['card']],
                [1, COLORS['positive']]
            ],
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, color=COLORS['text']),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlación: %{z:.3f}<extra></extra>"
        ))
        
        fig_heatmap.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            title=dict(text="Matriz de Correlación Completa", font=dict(size=16, color=COLORS['gold']))
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Bar chart de correlaciones con EUR/USD
        colors_bar = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in eurusd_corr.values]
        
        fig_bar = go.Figure(data=go.Bar(
            y=eurusd_corr.index,
            x=eurusd_corr.values,
            orientation='h',
            marker=dict(color=colors_bar, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Correlación: %{x:.3f}<extra></extra>"
        ))
        
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            title=dict(text="Correlación vs EUR/USD", font=dict(size=16, color=COLORS['gold'])),
            xaxis=dict(range=[-1, 1], dtick=0.25)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>💡 Interpretación:</strong> Una correlación positiva indica que el activo tiende a subir cuando el EUR se 
        fortalece frente al USD. Una correlación negativa indica lo contrario. Valores cercanos a ±1 indican una 
        relación muy fuerte.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ANÁLISIS BETA
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Sensibilidad (Beta) al EUR/USD")
    
    # Calcular betas para todos los activos
    betas_data = []
    for asset in selected_assets:
        if asset in returns.columns:
            beta_info = calculate_beta(returns, asset, "EUR/USD")
            betas_data.append({
                'Activo': asset,
                'Beta': beta_info['beta'],
                'Alpha (Ann.)': beta_info['alpha'] * 100,
                'R²': beta_info['r_squared'],
                'P-Value': beta_info['p_value']
            })
    
    betas_df = pd.DataFrame(betas_data).sort_values('Beta', ascending=False)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Scatter plot con línea de regresión para activo seleccionado
        selected_asset = st.selectbox("Seleccionar activo para detalle:", selected_assets)
        
        if selected_asset in returns.columns:
            fig_scatter = go.Figure()
            
            # Puntos de dispersión
            fig_scatter.add_trace(go.Scatter(
                x=returns["EUR/USD"],
                y=returns[selected_asset],
                mode='markers',
                marker=dict(
                    color=COLORS['gold'],
                    size=6,
                    opacity=0.5,
                    line=dict(width=1, color=COLORS['gold_light'])
                ),
                name='Retornos diarios',
                hovertemplate="EUR/USD: %{x:.3%}<br>" + selected_asset + ": %{y:.3%}<extra></extra>"
            ))
            
            # Línea de regresión
            beta_info = calculate_beta(returns, selected_asset, "EUR/USD")
            x_line = np.linspace(returns["EUR/USD"].min(), returns["EUR/USD"].max(), 100)
            y_line = beta_info['alpha']/252 + beta_info['beta'] * x_line
            
            fig_scatter.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color=COLORS['positive'], width=2, dash='dash'),
                name=f'Regresión (β={beta_info["beta"]:.2f})'
            ))
            
            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                height=450,
                title=dict(
                    text=f"Regresión {selected_asset} vs EUR/USD",
                    font=dict(size=16, color=COLORS['gold'])
                ),
                xaxis_title="Retorno EUR/USD",
                yaxis_title=f"Retorno {selected_asset}",
                showlegend=True
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Tabla de betas con formato
        st.markdown("#### Ranking por Beta")
        
        for _, row in betas_df.iterrows():
            beta_color = "positive" if row['Beta'] >= 0 else "negative"
            significance = "✓" if row['P-Value'] < 0.05 else "✗"
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 8px; padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 600; color: {COLORS['text']};">{row['Activo']}</span>
                        <span style="color: {COLORS['text_secondary']}; font-size: 0.8rem; margin-left: 8px;">R²: {row['R²']:.2%}</span>
                    </div>
                    <div>
                        <span class="metric-value {beta_color}" style="font-size: 1.2rem;">{row['Beta']:.3f}</span>
                        <span style="color: {COLORS['text_secondary']}; font-size: 0.7rem; margin-left: 4px;">{significance}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Interpretación del Beta:</strong> Un beta de 0.5 significa que por cada 1% que sube el EUR/USD, 
        el activo tiende a subir un 0.5%. Beta negativo indica relación inversa. ✓ indica significancia estadística (p<0.05).
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ROLLING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown(f"### Correlación Rolling ({rolling_window} días)")
    
    rolling_corr = calculate_rolling_correlation(returns, "EUR/USD", rolling_window)
    
    # Gráfico de correlaciones rolling
    fig_rolling = go.Figure()
    
    for i, col in enumerate(rolling_corr.columns):
        fig_rolling.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr[col],
            mode='lines',
            name=col,
            line=dict(color=COLORS['palette'][i % len(COLORS['palette'])], width=2),
            hovertemplate=f"<b>{col}</b><br>Fecha: %{{x}}<br>Correlación: %{{y:.3f}}<extra></extra>"
        ))
    
    # Añadir línea de referencia en 0
    fig_rolling.add_hline(y=0, line_dash="dash", line_color=COLORS['grid'], opacity=0.7)
    
    # Áreas de correlación fuerte
    fig_rolling.add_hrect(y0=0.5, y1=1, fillcolor=COLORS['positive'], opacity=0.05, line_width=0)
    fig_rolling.add_hrect(y0=-1, y1=-0.5, fillcolor=COLORS['negative'], opacity=0.05, line_width=0)
    
    fig_rolling.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        title=dict(text="Evolución Temporal de Correlaciones", font=dict(size=16, color=COLORS['gold'])),
        yaxis=dict(range=[-1, 1], dtick=0.25, **PLOTLY_LAYOUT['yaxis']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Estadísticas de rolling
    st.markdown("#### Estadísticas de Correlación Rolling")
    
    rolling_stats = pd.DataFrame({
        'Media': rolling_corr.mean(),
        'Std': rolling_corr.std(),
        'Mín': rolling_corr.min(),
        'Máx': rolling_corr.max(),
        'Actual': rolling_corr.iloc[-1]
    }).round(3)
    
    st.dataframe(rolling_stats.style.background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Insight:</strong> La correlación rolling muestra cómo cambia la relación entre activos a lo largo del tiempo. 
        Alta variabilidad sugiere que la relación no es estable y el hedge puede ser menos efectivo.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: PORTFOLIO IMPACT
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### Simulador de Impacto en Cartera")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Composición de Cartera")
        
        # Seleccionar activos para la cartera
        portfolio_assets = st.multiselect(
            "Activos en cartera:",
            options=[a for a in selected_assets if a in returns.columns],
            default=[a for a in ["SPY", "EZU", "GLD", "TLT"] if a in returns.columns][:3]
        )
        
        if portfolio_assets:
            weights = {}
            remaining = 100.0
            
            for i, asset in enumerate(portfolio_assets):
                if i == len(portfolio_assets) - 1:
                    # Último activo toma el resto
                    weight = remaining
                    st.slider(f"{asset}", 0, 100, int(weight), disabled=True, key=f"w_{asset}")
                else:
                    weight = st.slider(f"{asset}", 0, int(remaining), int(remaining//(len(portfolio_assets)-i)), key=f"w_{asset}")
                    remaining -= weight
                
                weights[asset] = weight / 100
    
    with col2:
        if portfolio_assets and len(portfolio_assets) >= 2:
            # Calcular métricas de la cartera
            portfolio_sensitivity = calculate_portfolio_sensitivity(returns, weights, "EUR/USD")
            
            # Métricas de la cartera
            st.markdown("#### Métricas de Sensibilidad")
            
            cols = st.columns(4)
            port_metrics = [
                ("Correlación", f"{portfolio_sensitivity['correlation']:.3f}"),
                ("Beta FX", f"{portfolio_sensitivity['beta']:.3f}"),
                ("R²", f"{portfolio_sensitivity['r_squared']:.2%}"),
                ("Vol. Contrib.", f"{portfolio_sensitivity['volatility_contribution']:.2%}")
            ]
            
            for col, (label, value) in zip(cols, port_metrics):
                with col:
                    st.metric(label, value)
            
            # Gráfico de composición
            fig_pie = go.Figure(data=go.Pie(
                labels=list(weights.keys()),
                values=[w*100 for w in weights.values()],
                hole=0.6,
                marker=dict(colors=COLORS['palette'][:len(weights)]),
                textinfo='label+percent',
                textfont=dict(color=COLORS['text'], size=12)
            ))
            
            fig_pie.update_layout(
                **PLOTLY_LAYOUT,
                height=350,
                showlegend=False,
                annotations=[dict(
                    text='Cartera',
                    x=0.5, y=0.5,
                    font=dict(size=16, color=COLORS['gold']),
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Selecciona al menos 2 activos para analizar la cartera")
    
    # Escenarios de impacto
    if portfolio_assets and len(portfolio_assets) >= 2:
        st.markdown("#### 📊 Escenarios de Impacto FX")
        
        scenarios = [-10, -5, -2, 2, 5, 10]  # % cambio en EUR/USD
        beta = portfolio_sensitivity['beta']
        
        scenario_data = []
        for s in scenarios:
            impact = beta * s
            scenario_data.append({
                'Escenario EUR/USD': f"{s:+.0f}%",
                'Impacto Estimado Cartera': f"{impact:+.2f}%",
                'Tipo': 'EUR ↑' if s > 0 else 'EUR ↓'
            })
        
        fig_scenarios = go.Figure()
        
        colors_scenario = [COLORS['positive'] if s > 0 else COLORS['negative'] for s in scenarios]
        impacts = [beta * s for s in scenarios]
        
        fig_scenarios.add_trace(go.Bar(
            x=[f"{s:+.0f}%" for s in scenarios],
            y=impacts,
            marker=dict(color=colors_scenario, line=dict(width=0)),
            text=[f"{i:+.2f}%" for i in impacts],
            textposition='outside',
            textfont=dict(color=COLORS['text']),
            hovertemplate="Cambio EUR/USD: %{x}<br>Impacto Cartera: %{y:.2f}%<extra></extra>"
        ))
        
        fig_scenarios.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            title=dict(text="Impacto Estimado por Escenario", font=dict(size=16, color=COLORS['gold'])),
            xaxis_title="Movimiento EUR/USD",
            yaxis_title="Impacto en Cartera (%)"
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>⚠️ Nota:</strong> Estos escenarios son estimaciones basadas en relaciones históricas. 
            El impacto real puede variar significativamente, especialmente en condiciones de mercado extremas.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLORS['text_secondary']}; padding: 2rem 0;">
    <p style="font-size: 0.85rem;">
        EUR/USD Impact Analyzer • Datos: Yahoo Finance • 
        Período: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}
    </p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem;">
        ⚠️ Esta herramienta es solo para fines educativos y de investigación. No constituye asesoramiento financiero.
    </p>
</div>
""", unsafe_allow_html=True)
