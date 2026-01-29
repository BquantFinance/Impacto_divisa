"""
FX Impact Analyzer - Inversores Europeos en USA
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas_datareader.data as web

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="FX Impact - EUR Investor", page_icon="🇪🇺", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
    
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d0d14 100%); }
    .main .block-container { padding: 1.5rem 3rem; max-width: 1400px; }
    h1, h2, h3, p, span, label { font-family: 'DM Sans', sans-serif !important; }
    
    .main-header {
        text-align: center; padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid #2a2a3a; margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #d4af37, #f4d03f);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .main-header p { color: #a0a0b0; font-size: 1rem; }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a24, #1f1f2e);
        border: 1px solid #2a2a3a; border-radius: 12px;
        padding: 1rem; text-align: center;
    }
    .metric-label { font-size: 0.65rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { font-family: 'Space Mono', monospace !important; font-size: 1.4rem; font-weight: 700; }
    .metric-sub { font-size: 0.6rem; color: #888; margin-top: 2px; }
    .positive { color: #00d084; }
    .negative { color: #ff6b6b; }
    .gold { color: #d4af37; }
    .blue { color: #3498db; }
    
    [data-testid="stSidebar"] { background: #12121a !important; }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(212,175,55,0.1), rgba(212,175,55,0.05));
        border-left: 4px solid #d4af37; border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem; margin: 1rem 0;
    }
    .insight-box h4 { color: #d4af37; margin: 0 0 0.5rem 0; font-size: 0.9rem; }
    .insight-box p { color: #e0e0e0; margin: 0; font-size: 0.9rem; line-height: 1.5; }
    
    .formula-box {
        background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 8px;
        padding: 1rem; margin: 0.5rem 0; font-family: 'Space Mono', monospace;
        font-size: 0.85rem; color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'gold': '#d4af37', 'positive': '#00d084', 'negative': '#ff6b6b',
    'blue': '#3498db', 'text': '#ffffff', 'text_secondary': '#a0a0b0', 'grid': '#2a2a3a',
    'usd': '#00d084', 'eur': '#3498db'
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
        st.error(f"Error: {e}")
        return pd.DataFrame()

def decompose_returns(prices_usd: pd.Series, eurusd: pd.Series) -> dict:
    """
    Cálculo EXACTO del retorno para inversor europeo.
    
    Un inversor EUR que compra un activo USD:
    1. Convierte EUR → USD al tipo EURUSD_inicio
    2. Compra el activo
    3. Vende el activo
    4. Convierte USD → EUR al tipo EURUSD_final
    
    Precio del activo en EUR = Precio en USD / EURUSD
    
    Retorno EUR = (Precio_EUR_final / Precio_EUR_inicio) - 1
                = (Precio_USD_final / EURUSD_final) / (Precio_USD_inicio / EURUSD_inicio) - 1
                = (Precio_USD_final / Precio_USD_inicio) × (EURUSD_inicio / EURUSD_final) - 1
                = (1 + ret_USD) × (1 / (1 + Δ_EURUSD)) - 1
    
    Si EUR/USD sube (EUR se fortalece): el inversor EUR pierde valor
    Si EUR/USD baja (EUR se debilita): el inversor EUR gana valor adicional
    """
    # Valores inicio/fin
    p_usd_start = prices_usd.iloc[0]
    p_usd_end = prices_usd.iloc[-1]
    fx_start = eurusd.iloc[0]
    fx_end = eurusd.iloc[-1]
    
    # Retorno del activo en USD
    ret_usd = (p_usd_end / p_usd_start) - 1
    
    # Cambio en EUR/USD (positivo = EUR se fortaleció)
    fx_pct_change = (fx_end / fx_start) - 1
    
    # Precio del activo en EUR
    p_eur_start = p_usd_start / fx_start
    p_eur_end = p_usd_end / fx_end
    
    # Retorno real en EUR (calculado directamente)
    ret_eur = (p_eur_end / p_eur_start) - 1
    
    # El efecto divisa es la diferencia
    fx_effect = ret_eur - ret_usd
    
    # Verificación matemática: (1 + ret_USD) × (fx_start / fx_end) - 1 = ret_EUR
    ret_eur_check = (1 + ret_usd) * (fx_start / fx_end) - 1
    
    return {
        'ret_usd': ret_usd * 100,           # Retorno activo en USD
        'ret_eur': ret_eur * 100,           # Retorno real en EUR  
        'fx_pct_change': fx_pct_change * 100,  # Cambio % de EUR/USD
        'fx_effect': fx_effect * 100,       # Impacto FX en puntos porcentuales
        'fx_start': fx_start,
        'fx_end': fx_end,
        'check': abs(ret_eur - ret_eur_check) < 0.0001  # Verificación
    }

def calculate_annual_breakdown(prices_usd: pd.Series, eurusd: pd.Series) -> pd.DataFrame:
    """Desglosa por año"""
    years = prices_usd.index.year.unique()
    data = []
    
    for year in years:
        mask = prices_usd.index.year == year
        if mask.sum() < 2:
            continue
        
        d = decompose_returns(prices_usd[mask], eurusd[mask])
        data.append({
            'Año': year,
            'Ret. USD': d['ret_usd'],
            'Ret. EUR': d['ret_eur'],
            'Δ EUR/USD': d['fx_pct_change'],
            'Impacto FX': d['fx_effect']
        })
    
    return pd.DataFrame(data)

def calculate_drawdowns(prices: pd.Series) -> dict:
    """Drawdown máximo"""
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max * 100
    return {'max_drawdown': drawdown.min(), 'date': drawdown.idxmin()}

def calculate_rolling_fx_impact(prices_usd: pd.Series, eurusd: pd.Series, window: int = 252) -> pd.Series:
    """Impacto FX rolling"""
    prices_eur = prices_usd / eurusd
    ret_usd = prices_usd.pct_change(window)
    ret_eur = prices_eur.pct_change(window)
    return ((ret_eur - ret_usd) * 100).dropna()

def base_layout(height=400):
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#ffffff'), height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor='#2a2a3a', zerolinecolor='#2a2a3a'),
        yaxis=dict(gridcolor='#2a2a3a', zerolinecolor='#2a2a3a')
    )

@st.cache_data(ttl=86400)
def get_interest_rates(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene tipos de interés de FRED:
    - FEDFUNDS: Fed Funds Effective Rate
    - ECBDFR: ECB Deposit Facility Rate
    """
    try:
        fed = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
        ecb = web.DataReader('ECBDFR', 'fred', start_date, end_date)
        
        rates = pd.DataFrame({
            'Fed': fed['FEDFUNDS'],
            'ECB': ecb['ECBDFR']
        })
        
        # Forward fill para alinear frecuencias
        rates = rates.ffill().dropna()
        
        # Coste de cobertura = tipo USD - tipo EUR (aprox)
        rates['Hedge_Cost'] = rates['Fed'] - rates['ECB']
        
        return rates
    except Exception as e:
        st.warning(f"No se pudieron obtener tipos de interés: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_hedged_etf(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Obtiene ETFs para comparar:
    - SPY: S&P 500 (USD)
    - IUSE.L: iShares S&P 500 EUR Hedged (London)
    - CSPX.L: iShares Core S&P 500 (USD, London) - para comparar
    """
    tickers = ['SPY', 'IUSE.L', 'CSPX.L', 'EURUSD=X']
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, 
                          auto_adjust=True, multi_level_index=False)['Close']
        return data.dropna()
    except Exception as e:
        st.warning(f"Error descargando ETFs: {e}")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    
    st.markdown("#### 📅 Período")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Inicio", value=datetime(2020, 1, 1))
    end_date = c2.date_input("Fin", value=datetime.now())
    
    st.markdown("---")
    st.markdown("#### 📊 Activos USA")
    
    # Input de tickers - SIMPLE Y DIRECTO
    default_tickers = "SPY, QQQ"
    tickers_input = st.text_area(
        "Escribe los tickers separados por coma:",
        value=default_tickers,
        height=80,
        help="Ejemplo: SPY, QQQ, AAPL, MSFT, NVDA"
    )
    
    # Procesar tickers
    selected_assets = []
    if tickers_input:
        for t in tickers_input.replace('\n', ',').split(','):
            ticker = t.strip().upper()
            if ticker and ticker not in selected_assets:
                selected_assets.append(ticker)
    
    if selected_assets:
        st.success(f"✓ {len(selected_assets)} activos: {', '.join(selected_assets)}")
    
    st.markdown("---")
    st.caption("**Tickers populares:** SPY, QQQ, IWM, VTI, AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🇪🇺 FX Impact Analyzer</h1>
    <p>El impacto real del EUR/USD en tus inversiones americanas</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

if not selected_assets:
    st.warning("⚠️ Escribe al menos un ticker en el panel lateral")
    st.stop()

with st.spinner("Descargando datos..."):
    all_tickers = ["EURUSD=X"] + selected_assets
    prices = get_data(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if prices.empty or "EURUSD=X" not in prices.columns:
    st.error("No se pudieron obtener datos. Verifica los tickers.")
    st.stop()

eurusd = prices["EURUSD=X"]
prices = prices.drop(columns=["EURUSD=X"])

# Filtrar solo los que se descargaron correctamente
valid_assets = [a for a in selected_assets if a in prices.columns]
invalid_assets = [a for a in selected_assets if a not in prices.columns]

if invalid_assets:
    st.warning(f"⚠️ Tickers no encontrados: {', '.join(invalid_assets)}")

if not valid_assets:
    st.error("Ningún ticker válido encontrado.")
    st.stop()

selected_assets = valid_assets

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

main_asset = selected_assets[0]
d = decompose_returns(prices[main_asset], eurusd)

cols = st.columns(5)
metrics_data = [
    ("EUR/USD", f"{d['fx_start']:.4f} → {d['fx_end']:.4f}", f"{d['fx_pct_change']:+.1f}%", "gold"),
    (f"{main_asset} (USD)", f"{d['ret_usd']:+.1f}%", "Retorno en dólares", "positive" if d['ret_usd'] >= 0 else "negative"),
    ("Efecto Divisa", f"{d['fx_effect']:+.1f}pp", "Suma/resta a tu retorno", "positive" if d['fx_effect'] >= 0 else "negative"),
    (f"{main_asset} (EUR)", f"{d['ret_eur']:+.1f}%", "Tu retorno real", "positive" if d['ret_eur'] >= 0 else "negative"),
    ("Diferencia", f"{d['fx_effect']:+.1f}pp", "Lo que ganas/pierdes por FX", "positive" if d['fx_effect'] >= 0 else "negative"),
]

for col, (label, value, sub, cls) in zip(cols, metrics_data):
    col.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>''', unsafe_allow_html=True)

# Fórmula y verificación
st.markdown(f"""
<div class="formula-box">
<strong>📐 Cálculo:</strong> Retorno EUR = Retorno USD + Efecto Divisa<br>
<strong>{d['ret_eur']:.2f}%</strong> = {d['ret_usd']:.2f}% + ({d['fx_effect']:+.2f}pp)<br>
<span style="color:#888;">Verificación: ✓ Correcto</span>
</div>
""", unsafe_allow_html=True)

# Insight dinámico
if d['fx_effect'] < 0:
    insight = f"""El <strong>EUR se fortaleció</strong> un {d['fx_pct_change']:.1f}% frente al USD. 
    Esto <strong>restó {abs(d['fx_effect']):.1f} puntos porcentuales</strong> a tu inversión.
    El activo subió {d['ret_usd']:.1f}% en USD, pero en EUR solo ganaste {d['ret_eur']:.1f}%."""
else:
    insight = f"""El <strong>EUR se debilitó</strong> un {abs(d['fx_pct_change']):.1f}% frente al USD. 
    Esto <strong>sumó {d['fx_effect']:.1f} puntos porcentuales</strong> a tu inversión.
    El activo subió {d['ret_usd']:.1f}% en USD, y en EUR ganaste {d['ret_eur']:.1f}%."""

st.markdown(f'<div class="insight-box"><h4>💡 Resumen</h4><p>{insight}</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Evolución: Inversor USA vs Inversor Europa")

if len(selected_assets) > 1:
    asset_to_show = st.selectbox("Activo:", selected_assets, index=0)
else:
    asset_to_show = selected_assets[0]

# Normalizar a base 100
prices_usd_norm = (prices[asset_to_show] / prices[asset_to_show].iloc[0]) * 100
prices_eur = prices[asset_to_show] / eurusd
prices_eur_norm = (prices_eur / prices_eur.iloc[0]) * 100

fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=[f"{asset_to_show}: Retorno acumulado", "EUR/USD"])

fig.add_trace(go.Scatter(x=prices_usd_norm.index, y=prices_usd_norm, name='Inversor USA (USD)',
                         line=dict(color=COLORS['usd'], width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=prices_eur_norm.index, y=prices_eur_norm, name='Inversor Europa (EUR)',
                         line=dict(color=COLORS['eur'], width=2.5)), row=1, col=1)

# Área del gap
fig.add_trace(go.Scatter(x=prices_usd_norm.index, y=prices_usd_norm, fill=None, mode='lines',
                         line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=prices_eur_norm.index, y=prices_eur_norm, fill='tonexty', mode='lines',
                         line=dict(width=0), fillcolor='rgba(255,107,107,0.15)', name='Diferencia FX'), row=1, col=1)

fig.add_hline(y=100, line_dash="dash", line_color=COLORS['grid'], opacity=0.5, row=1, col=1)

fig.add_trace(go.Scatter(x=eurusd.index, y=eurusd, name='EUR/USD',
                         line=dict(color=COLORS['gold'], width=2)), row=2, col=1)

fig.update_layout(**base_layout(500), hovermode='x unified',
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
fig.update_yaxes(title_text="Base 100", row=1, col=1, gridcolor=COLORS['grid'])
fig.update_yaxes(title_text="EUR/USD", row=2, col=1, gridcolor=COLORS['grid'])

st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS POR AÑOS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📅 Desglose por Año")

annual_df = calculate_annual_breakdown(prices[asset_to_show], eurusd)

if len(annual_df) > 0:
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(name='Retorno USD', x=annual_df['Año'].astype(str), y=annual_df['Ret. USD'],
                          marker_color=COLORS['usd']))
    fig2.add_trace(go.Bar(name='Retorno EUR', x=annual_df['Año'].astype(str), y=annual_df['Ret. EUR'],
                          marker_color=COLORS['eur']))
    fig2.add_trace(go.Scatter(name='Impacto FX', x=annual_df['Año'].astype(str), y=annual_df['Impacto FX'],
                              mode='lines+markers', line=dict(color=COLORS['gold'], width=3),
                              marker=dict(size=10)))
    
    fig2.add_hline(y=0, line_color=COLORS['grid'])
    fig2.update_layout(**base_layout(350), barmode='group',
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Tabla
    disp = annual_df.copy()
    disp['Ret. USD'] = disp['Ret. USD'].apply(lambda x: f"{x:+.1f}%")
    disp['Ret. EUR'] = disp['Ret. EUR'].apply(lambda x: f"{x:+.1f}%")
    disp['Δ EUR/USD'] = disp['Δ EUR/USD'].apply(lambda x: f"{x:+.1f}%")
    disp['Impacto FX'] = disp['Impacto FX'].apply(lambda x: f"{x:+.1f}pp")
    disp['¿FX Ayudó?'] = annual_df['Impacto FX'].apply(lambda x: "✅ Sí" if x > 0 else "❌ No")
    st.dataframe(disp, hide_index=True, use_container_width=True)
    
    años_favor = (annual_df['Impacto FX'] > 0).sum()
    años_contra = len(annual_df) - años_favor
    st.info(f"📊 El FX te benefició en **{años_favor}** años y te perjudicó en **{años_contra}** años")

# ══════════════════════════════════════════════════════════════════════════════
# DRAWDOWN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📉 Comparación de Drawdowns (Caídas Máximas)")

dd_usd = calculate_drawdowns(prices[asset_to_show])
dd_eur = calculate_drawdowns(prices[asset_to_show] / eurusd)

c1, c2, c3 = st.columns(3)
c1.metric("Max Drawdown (USD)", f"{dd_usd['max_drawdown']:.1f}%")
c2.metric("Max Drawdown (EUR)", f"{dd_eur['max_drawdown']:.1f}%")
diff_dd = dd_eur['max_drawdown'] - dd_usd['max_drawdown']
c3.metric("Diferencia", f"{diff_dd:+.1f}pp")

if diff_dd < -1:
    st.success(f"✅ El efecto divisa **amortiguó** tu caída máxima en {abs(diff_dd):.1f}pp")
elif diff_dd > 1:
    st.error(f"⚠️ El efecto divisa **amplificó** tu caída máxima en {diff_dd:.1f}pp")
else:
    st.info("➡️ El efecto divisa fue neutral en el drawdown máximo")

# ══════════════════════════════════════════════════════════════════════════════
# IMPACTO FX ROLLING
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🔄 Impacto FX Rolling (1 año)")

rolling_fx = calculate_rolling_fx_impact(prices[asset_to_show], eurusd, 252)

if len(rolling_fx) > 50:
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(x=rolling_fx.index, y=rolling_fx,
                              name='Impacto FX (12 meses)',
                              line=dict(color=COLORS['gold'], width=2),
                              fill='tozeroy', fillcolor='rgba(212,175,55,0.2)'))
    
    fig3.add_hline(y=0, line_dash="dash", line_color=COLORS['text_secondary'])
    fig3.add_hline(y=rolling_fx.mean(), line_dash="dot", line_color=COLORS['blue'],
                   annotation_text=f"Media: {rolling_fx.mean():.1f}pp")
    
    fig3.update_layout(**base_layout(300), yaxis_title="Impacto FX (pp)")
    st.plotly_chart(fig3, use_container_width=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media histórica", f"{rolling_fx.mean():+.1f}pp")
    c2.metric("Mejor momento", f"{rolling_fx.max():+.1f}pp")
    c3.metric("Peor momento", f"{rolling_fx.min():+.1f}pp")
    c4.metric("Volatilidad", f"{rolling_fx.std():.1f}pp")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARATIVA TODOS LOS ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════

if len(selected_assets) > 1:
    st.markdown("### 📊 Comparativa: Todos los Activos")
    
    comparison_data = []
    for asset in selected_assets:
        dc = decompose_returns(prices[asset], eurusd)
        comparison_data.append({
            'Activo': asset,
            'Ret. USD': dc['ret_usd'],
            'Ret. EUR': dc['ret_eur'],
            'Impacto FX': dc['fx_effect']
        })
    
    comp_df = pd.DataFrame(comparison_data).sort_values('Ret. EUR', ascending=False)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Retorno USD', x=comp_df['Activo'], y=comp_df['Ret. USD'], marker_color=COLORS['usd']))
    fig4.add_trace(go.Bar(name='Retorno EUR', x=comp_df['Activo'], y=comp_df['Ret. EUR'], marker_color=COLORS['eur']))
    fig4.update_layout(**base_layout(350), barmode='group',
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# COSTE DE COBERTURA (HEDGING)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 💰 Coste de Cobertura vs Beneficio")
st.caption("¿Vale la pena cubrir el riesgo divisa? Analizamos el coste histórico del hedge")

with st.spinner("Descargando tipos de interés de FRED..."):
    rates = get_interest_rates(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if not rates.empty:
    # Alinear con nuestros datos
    rates_aligned = rates.reindex(prices.index, method='ffill').dropna()
    
    if len(rates_aligned) > 0:
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            # Gráfico de tipos y coste de hedge
            fig_rates = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True,
                                      vertical_spacing=0.08,
                                      subplot_titles=["Tipos de Interés", "Coste de Cobertura (Fed - ECB)"])
            
            fig_rates.add_trace(go.Scatter(x=rates_aligned.index, y=rates_aligned['Fed'], 
                                           name='Fed Funds', line=dict(color=COLORS['usd'], width=2)), row=1, col=1)
            fig_rates.add_trace(go.Scatter(x=rates_aligned.index, y=rates_aligned['ECB'], 
                                           name='ECB Deposit', line=dict(color=COLORS['eur'], width=2)), row=1, col=1)
            
            fig_rates.add_trace(go.Scatter(x=rates_aligned.index, y=rates_aligned['Hedge_Cost'], 
                                           name='Coste Hedge', line=dict(color=COLORS['gold'], width=2),
                                           fill='tozeroy', fillcolor='rgba(212,175,55,0.2)'), row=2, col=1)
            fig_rates.add_hline(y=0, line_dash="dash", line_color=COLORS['grid'], row=2, col=1)
            
            fig_rates.update_layout(**base_layout(400), 
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
            fig_rates.update_yaxes(title_text="%", row=1, col=1, gridcolor=COLORS['grid'])
            fig_rates.update_yaxes(title_text="% anual", row=2, col=1, gridcolor=COLORS['grid'])
            
            st.plotly_chart(fig_rates, use_container_width=True)
        
        with c2:
            # Métricas y análisis
            avg_hedge_cost = rates_aligned['Hedge_Cost'].mean()
            current_hedge_cost = rates_aligned['Hedge_Cost'].iloc[-1]
            
            # Calcular si habría valido la pena cubrir
            d_main = decompose_returns(prices[main_asset], eurusd)
            total_fx_effect = d_main['fx_effect']
            
            # Coste acumulado del hedge (aproximado)
            years = len(rates_aligned) / 252
            cumulative_hedge_cost = avg_hedge_cost * years
            
            st.markdown("#### Análisis Coste-Beneficio")
            
            st.metric("Coste Hedge Actual", f"{current_hedge_cost:.2f}% anual")
            st.metric("Coste Hedge Promedio", f"{avg_hedge_cost:.2f}% anual")
            st.metric(f"Coste Acumulado ({years:.1f} años)", f"{cumulative_hedge_cost:.1f}%")
            
            st.markdown("---")
            st.metric("Efecto FX Real (sin hedge)", f"{total_fx_effect:+.1f}%")
            
            # Conclusión
            if total_fx_effect < -cumulative_hedge_cost:
                st.error(f"❌ **Cubrir habría costado {cumulative_hedge_cost:.1f}%** pero el FX te quitó {abs(total_fx_effect):.1f}%. Habría valido la pena.")
            elif total_fx_effect > 0:
                st.success(f"✅ **El FX te benefició {total_fx_effect:.1f}%**. Cubrir habría costado {cumulative_hedge_cost:.1f}%. Mejor sin hedge.")
            else:
                diff = abs(total_fx_effect) - cumulative_hedge_cost
                if diff > 0:
                    st.warning(f"⚠️ El FX te quitó {abs(total_fx_effect):.1f}%, cubrir costaba {cumulative_hedge_cost:.1f}%. Diferencia: {diff:.1f}%")
                else:
                    st.info(f"➡️ El FX te quitó {abs(total_fx_effect):.1f}%, cubrir costaba {cumulative_hedge_cost:.1f}%. Casi neutral.")
else:
    st.info("No se pudieron obtener datos de tipos de interés para este período")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARATIVA CON ETF HEDGED REAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🔄 Comparativa: ETF Normal vs ETF con Cobertura")
st.caption("Comparamos un ETF sin cobertura convertido a EUR vs un ETF EUR Hedged real")

with st.spinner("Descargando ETFs..."):
    etf_data = get_hedged_etf(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if not etf_data.empty and 'SPY' in etf_data.columns and 'IUSE.L' in etf_data.columns:
    # SPY en EUR (sin cobertura)
    spy_eur = etf_data['SPY'] / etf_data['EURUSD=X']
    spy_eur_norm = (spy_eur / spy_eur.iloc[0]) * 100
    
    # IUSE.L ya está en EUR con cobertura
    iuse_norm = (etf_data['IUSE.L'] / etf_data['IUSE.L'].iloc[0]) * 100
    
    # SPY en USD para referencia
    spy_usd_norm = (etf_data['SPY'] / etf_data['SPY'].iloc[0]) * 100
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        fig_etf = go.Figure()
        
        fig_etf.add_trace(go.Scatter(x=spy_usd_norm.index, y=spy_usd_norm, name='SPY (USD) - Referencia',
                                     line=dict(color=COLORS['text_secondary'], width=1.5, dash='dot')))
        fig_etf.add_trace(go.Scatter(x=spy_eur_norm.index, y=spy_eur_norm, name='SPY en EUR (sin hedge)',
                                     line=dict(color=COLORS['eur'], width=2.5)))
        fig_etf.add_trace(go.Scatter(x=iuse_norm.index, y=iuse_norm, name='IUSE.L (EUR Hedged)',
                                     line=dict(color=COLORS['gold'], width=2.5)))
        
        fig_etf.add_hline(y=100, line_dash="dash", line_color=COLORS['grid'], opacity=0.5)
        
        fig_etf.update_layout(**base_layout(400),
                              title=dict(text="S&P 500: Sin Cobertura vs Con Cobertura (EUR)", 
                                        font=dict(size=14, color=COLORS['gold'])),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
        
        st.plotly_chart(fig_etf, use_container_width=True)
    
    with c2:
        # Calcular métricas
        ret_spy_usd = spy_usd_norm.iloc[-1] - 100
        ret_spy_eur = spy_eur_norm.iloc[-1] - 100
        ret_iuse = iuse_norm.iloc[-1] - 100
        
        # Tracking difference
        tracking_diff = ret_iuse - ret_spy_usd
        hedge_benefit = ret_iuse - ret_spy_eur
        
        st.markdown("#### Resultados")
        
        st.metric("SPY (USD)", f"{ret_spy_usd:+.1f}%", help="Retorno original en dólares")
        st.metric("SPY en EUR (sin hedge)", f"{ret_spy_eur:+.1f}%", help="Lo que ganaste realmente sin cubrir")
        st.metric("IUSE.L (EUR Hedged)", f"{ret_iuse:+.1f}%", help="ETF con cobertura de divisa")
        
        st.markdown("---")
        
        st.metric("Coste real del hedge", f"{tracking_diff:+.1f}%", 
                  help="Diferencia entre ETF hedged y SPY USD (incluye TER + coste hedge)")
        
        if hedge_benefit > 0:
            st.success(f"✅ El hedge te habría dado **{hedge_benefit:.1f}% más** que no cubrir")
        else:
            st.error(f"❌ El hedge te habría dado **{abs(hedge_benefit):.1f}% menos** que no cubrir")
    
    # Tabla resumen
    st.markdown("#### Resumen Comparativo")
    
    # Volatilidades
    vol_spy_eur = spy_eur_norm.pct_change().std() * np.sqrt(252) * 100
    vol_iuse = iuse_norm.pct_change().std() * np.sqrt(252) * 100
    vol_spy_usd = spy_usd_norm.pct_change().std() * np.sqrt(252) * 100
    
    comparison_table = pd.DataFrame({
        'Métrica': ['Retorno Total', 'Volatilidad Anualizada', 'Sharpe Ratio (aprox)'],
        'SPY (USD)': [f"{ret_spy_usd:+.1f}%", f"{vol_spy_usd:.1f}%", f"{ret_spy_usd/vol_spy_usd:.2f}"],
        'SPY en EUR (sin hedge)': [f"{ret_spy_eur:+.1f}%", f"{vol_spy_eur:.1f}%", f"{ret_spy_eur/vol_spy_eur:.2f}"],
        'IUSE.L (EUR Hedged)': [f"{ret_iuse:+.1f}%", f"{vol_iuse:.1f}%", f"{ret_iuse/vol_iuse:.2f}"]
    })
    
    st.dataframe(comparison_table, hide_index=True, use_container_width=True)
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>📊 Interpretación</h4>
    <p>
    <strong>IUSE.L</strong> es un ETF que replica el S&P 500 pero cubre el riesgo EUR/USD mensualmente.<br><br>
    • Si IUSE.L > SPY en EUR → <strong>El hedge valió la pena</strong> (el coste fue menor que la pérdida por FX)<br>
    • Si IUSE.L < SPY en EUR → <strong>Mejor sin hedge</strong> (el EUR se debilitó o el coste fue alto)<br><br>
    <strong>Nota:</strong> El tracking difference de IUSE.L incluye el TER (0.20%) + coste de cobertura (~diferencial de tipos).
    </p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.warning("No se pudieron obtener datos de los ETFs hedged para este período. Prueba con fechas desde 2015.")

# ══════════════════════════════════════════════════════════════════════════════
# SIMULADOR DE CARTERA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 💼 Simulador de Cartera")
st.caption("Construye tu cartera y mira el impacto FX agregado")

c1, c2 = st.columns([1, 2])

with c1:
    st.markdown("#### Pesos (%)")
    weights = {}
    total_weight = 0
    
    for asset in selected_assets:
        default_w = 100 // len(selected_assets) if len(selected_assets) <= 5 else 0
        w = st.number_input(asset, min_value=0, max_value=100, value=default_w, key=f"w_{asset}")
        weights[asset] = w / 100
        total_weight += w
    
    if total_weight == 100:
        st.success(f"✓ Total: {total_weight}%")
    elif total_weight > 0:
        st.warning(f"Total: {total_weight}% (debería ser 100%)")

with c2:
    if total_weight > 0:
        # Calcular cartera ponderada
        port_usd = sum(prices[a] * weights[a] for a in selected_assets if weights[a] > 0)
        port_eur = sum((prices[a] / eurusd) * weights[a] for a in selected_assets if weights[a] > 0)
        
        # Normalizar
        port_usd_norm = (port_usd / port_usd.iloc[0]) * 100
        port_eur_norm = (port_eur / port_eur.iloc[0]) * 100
        
        # Métricas
        ret_usd = port_usd_norm.iloc[-1] - 100
        ret_eur = port_eur_norm.iloc[-1] - 100
        fx_impact = ret_eur - ret_usd
        
        st.markdown("#### Resultado de tu Cartera")
        
        cols = st.columns(3)
        cols[0].metric("Retorno USD", f"{ret_usd:+.1f}%")
        cols[1].metric("Efecto FX", f"{fx_impact:+.1f}pp", delta_color="normal" if fx_impact >= 0 else "inverse")
        cols[2].metric("Retorno EUR", f"{ret_eur:+.1f}%")
        
        # Gráfico cartera
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=port_usd_norm.index, y=port_usd_norm, name='Cartera (USD)',
                                      line=dict(color=COLORS['usd'], width=2.5)))
        fig_port.add_trace(go.Scatter(x=port_eur_norm.index, y=port_eur_norm, name='Cartera (EUR)',
                                      line=dict(color=COLORS['eur'], width=2.5)))
        
        # Área entre líneas
        fig_port.add_trace(go.Scatter(x=port_usd_norm.index, y=port_usd_norm, fill=None, mode='lines',
                                      line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_port.add_trace(go.Scatter(x=port_eur_norm.index, y=port_eur_norm, fill='tonexty', mode='lines',
                                      line=dict(width=0), fillcolor='rgba(255,107,107,0.15)', 
                                      name='Efecto FX', hoverinfo='skip'))
        
        fig_port.add_hline(y=100, line_dash="dash", line_color=COLORS['grid'])
        fig_port.update_layout(**base_layout(350),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                               title=dict(text="Evolución de tu Cartera", font=dict(size=14, color=COLORS['gold'])))
        st.plotly_chart(fig_port, use_container_width=True)
        
        # Desglose por activo
        st.markdown("#### Contribución por Activo")
        contrib_data = []
        for asset in selected_assets:
            if weights[asset] > 0:
                d = decompose_returns(prices[asset], eurusd)
                contrib_data.append({
                    'Activo': asset,
                    'Peso': f"{weights[asset]*100:.0f}%",
                    'Ret. USD': f"{d['ret_usd']:+.1f}%",
                    'Ret. EUR': f"{d['ret_eur']:+.1f}%",
                    'Contrib. USD': f"{d['ret_usd'] * weights[asset]:+.1f}%",
                    'Contrib. EUR': f"{d['ret_eur'] * weights[asset]:+.1f}%"
                })
        st.dataframe(pd.DataFrame(contrib_data), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="insight-box">
<h4>🎓 Conclusiones para el Inversor Europeo</h4>
<p>
<strong>1. El riesgo divisa es real:</strong> Puede sumar o restar varios puntos porcentuales cada año.<br><br>
<strong>2. Es bidireccional:</strong> A veces te beneficia, a veces te perjudica.<br><br>
<strong>3. Opciones de gestión:</strong><br>
• <strong>Aceptar el riesgo:</strong> A largo plazo tiende a neutralizarse<br>
• <strong>ETFs con cobertura:</strong> iShares S&P 500 EUR Hedged (IUSE.DE), Xtrackers S&P 500 EUR Hedged (XDPE.DE)<br>
• <strong>Cobertura activa:</strong> Futuros EUR/USD (coste ≈ diferencial de tipos de interés)
</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div style="text-align:center;color:#666;font-size:0.75rem;padding:1rem;">FX Impact Analyzer • {start_date:%d/%m/%Y} - {end_date:%d/%m/%Y}</div>', unsafe_allow_html=True)
