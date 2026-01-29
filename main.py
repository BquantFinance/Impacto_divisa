"""
FX Impact Analyzer - Inversores Europeos en USA
Demuestra el impacto del tipo de cambio EUR/USD en carteras de inversores europeos
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
    'blue': '#3498db', 'purple': '#9b59b6', 'text': '#ffffff',
    'text_secondary': '#a0a0b0', 'grid': '#2a2a3a',
    'usd': '#00d084', 'eur': '#3498db', 'fx': '#ff6b6b'
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

def decompose_returns_exact(prices_usd: pd.Series, eurusd: pd.Series) -> dict:
    """
    Descomposición EXACTA del retorno para inversor europeo.
    
    Fórmula: (1 + r_EUR) = (1 + r_USD) × (1 + r_FX_inverso)
    Donde r_FX_inverso = (EURUSD_inicio / EURUSD_final) - 1
    
    El efecto FX incluye el término de interacción.
    """
    # Retornos
    ret_usd = prices_usd.iloc[-1] / prices_usd.iloc[0] - 1
    
    # Cambio en EUR/USD
    fx_change = eurusd.iloc[-1] / eurusd.iloc[0] - 1  # Si positivo, EUR se fortaleció
    
    # Factor de conversión FX (inverso del cambio)
    # Si EUR/USD sube 10%, el factor es 1/1.10 = 0.909, es decir -9.1%
    fx_factor = eurusd.iloc[0] / eurusd.iloc[-1]
    fx_return = fx_factor - 1  # Retorno por efecto divisa puro
    
    # Retorno total en EUR (calculado directamente)
    prices_eur = prices_usd / eurusd
    ret_eur = prices_eur.iloc[-1] / prices_eur.iloc[0] - 1
    
    # Verificación: (1 + ret_usd) * (1 + fx_return) - 1 ≈ ret_eur
    ret_eur_check = (1 + ret_usd) * (1 + fx_return) - 1
    
    # Descomposición aditiva aproximada (para mostrar)
    # ret_eur ≈ ret_usd + fx_return + (ret_usd × fx_return)  <- término de interacción
    interaction_term = ret_usd * fx_return
    
    return {
        'ret_usd': ret_usd * 100,
        'fx_change': fx_change * 100,           # Cambio real de EUR/USD
        'fx_return': fx_return * 100,           # Impacto en tu inversión
        'ret_eur': ret_eur * 100,
        'interaction': interaction_term * 100,   # Término cruzado
        'eurusd_start': eurusd.iloc[0],
        'eurusd_end': eurusd.iloc[-1]
    }

def calculate_annual_breakdown(prices_usd: pd.Series, eurusd: pd.Series) -> pd.DataFrame:
    """Desglosa el impacto FX por año"""
    prices_eur = prices_usd / eurusd
    
    # Agrupar por año
    years = prices_usd.index.year.unique()
    data = []
    
    for year in years:
        mask = prices_usd.index.year == year
        if mask.sum() < 2:
            continue
            
        p_usd = prices_usd[mask]
        p_eur = prices_eur[mask]
        fx = eurusd[mask]
        
        ret_usd = (p_usd.iloc[-1] / p_usd.iloc[0] - 1) * 100
        ret_eur = (p_eur.iloc[-1] / p_eur.iloc[0] - 1) * 100
        fx_chg = (fx.iloc[-1] / fx.iloc[0] - 1) * 100
        fx_impact = ret_eur - ret_usd
        
        data.append({
            'Año': year,
            'Ret. USD': ret_usd,
            'Ret. EUR': ret_eur,
            'Δ EUR/USD': fx_chg,
            'Impacto FX': fx_impact,
            'FX Ayudó': fx_impact > 0
        })
    
    return pd.DataFrame(data)

def calculate_drawdowns(prices: pd.Series) -> dict:
    """Calcula drawdown máximo"""
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Recuperación
    peak_date = prices[:max_dd_date].idxmax()
    
    return {
        'max_drawdown': max_dd,
        'max_dd_date': max_dd_date,
        'peak_date': peak_date
    }

def calculate_rolling_fx_impact(prices_usd: pd.Series, eurusd: pd.Series, window: int = 252) -> pd.DataFrame:
    """Calcula el impacto FX rolling"""
    prices_eur = prices_usd / eurusd
    
    ret_usd_roll = prices_usd.pct_change(window) * 100
    ret_eur_roll = prices_eur.pct_change(window) * 100
    fx_impact_roll = ret_eur_roll - ret_usd_roll
    
    return pd.DataFrame({
        'ret_usd': ret_usd_roll,
        'ret_eur': ret_eur_roll,
        'fx_impact': fx_impact_roll
    }).dropna()

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
    
    st.markdown("#### 📅 Período")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Inicio", value=datetime(2020, 1, 1))
    end_date = c2.date_input("Fin", value=datetime.now())
    
    st.markdown("---")
    st.markdown("#### 📊 Activos USA")
    
    PRESETS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", 
               "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Google"}
    
    preset_selection = st.multiselect("Predefinidos:", list(PRESETS.keys()), default=["SPY"],
                                       format_func=lambda x: f"{x} ({PRESETS[x]})")
    
    custom_tickers = st.text_input("Añadir tickers:", placeholder="TSLA, META...")
    
    selected_assets = list(preset_selection)
    if custom_tickers:
        selected_assets.extend([t.strip().upper() for t in custom_tickers.split(",") if t.strip() and t.strip().upper() not in selected_assets])

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
    st.warning("⚠️ Selecciona al menos un activo")
    st.stop()

with st.spinner("Descargando datos..."):
    prices = get_data(["EURUSD=X"] + selected_assets, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if prices.empty or "EURUSD=X" not in prices.columns:
    st.error("No se pudieron obtener datos.")
    st.stop()

eurusd = prices["EURUSD=X"]
prices = prices.drop(columns=["EURUSD=X"])
selected_assets = [a for a in selected_assets if a in prices.columns]

if not selected_assets:
    st.error("Ningún activo válido.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

main_asset = selected_assets[0]
d = decompose_returns_exact(prices[main_asset], eurusd)

cols = st.columns(5)
metrics_data = [
    ("EUR/USD", f"{d['eurusd_start']:.4f} → {d['eurusd_end']:.4f}", f"{d['fx_change']:+.1f}%", "gold"),
    (f"{main_asset} en USD", f"{d['ret_usd']:+.1f}%", "Lo que sube el activo", "positive" if d['ret_usd'] >= 0 else "negative"),
    ("Efecto Divisa", f"{d['fx_return']:+.1f}%", "Por cambio EUR/USD", "positive" if d['fx_return'] >= 0 else "negative"),
    (f"{main_asset} en EUR", f"{d['ret_eur']:+.1f}%", "Tu retorno real", "positive" if d['ret_eur'] >= 0 else "negative"),
    ("Diferencia", f"{d['ret_eur'] - d['ret_usd']:+.1f}pp", "EUR vs USD", "positive" if d['ret_eur'] >= d['ret_usd'] else "negative"),
]

for col, (label, value, sub, cls) in zip(cols, metrics_data):
    col.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>''', unsafe_allow_html=True)

# Fórmula explicativa
st.markdown(f"""
<div class="formula-box">
<strong>📐 Fórmula exacta:</strong><br>
(1 + {d['ret_eur']:.2f}%) = (1 + {d['ret_usd']:.2f}%) × (1 + {d['fx_return']:.2f}%)<br>
<span style="color:#888;">Retorno EUR = Retorno activo × Factor divisa</span>
</div>
""", unsafe_allow_html=True)

# Insight
if d['fx_return'] < 0:
    insight = f"""El <strong>EUR se fortaleció</strong> un {d['fx_change']:.1f}% frente al USD. 
    Esto <strong>redujo tu retorno en {abs(d['fx_return']):.1f} puntos porcentuales</strong>.
    Sin el efecto divisa, habrías ganado {d['ret_usd']:.1f}%, pero en EUR solo ganaste {d['ret_eur']:.1f}%."""
else:
    insight = f"""El <strong>EUR se debilitó</strong> un {abs(d['fx_change']):.1f}% frente al USD. 
    Esto <strong>aumentó tu retorno en {d['fx_return']:.1f} puntos porcentuales</strong>.
    El activo subió {d['ret_usd']:.1f}% en USD, y en EUR ganaste {d['ret_eur']:.1f}%."""

st.markdown(f'<div class="insight-box"><h4>💡 Resumen</h4><p>{insight}</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Evolución: USD vs EUR")

asset_to_show = st.selectbox("Activo:", selected_assets, index=0)

prices_usd_norm = (prices[asset_to_show] / prices[asset_to_show].iloc[0]) * 100
prices_eur = prices[asset_to_show] / eurusd
prices_eur_norm = (prices_eur / prices_eur.iloc[0]) * 100

fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=[f"Retorno acumulado de {asset_to_show}", "EUR/USD"])

fig.add_trace(go.Scatter(x=prices_usd_norm.index, y=prices_usd_norm, name='En USD (inversor USA)',
                         line=dict(color=COLORS['usd'], width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=prices_eur_norm.index, y=prices_eur_norm, name='En EUR (inversor Europa)',
                         line=dict(color=COLORS['eur'], width=2.5)), row=1, col=1)

# Área del gap
fig.add_trace(go.Scatter(x=prices_usd_norm.index, y=prices_usd_norm, fill=None, mode='lines',
                         line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=prices_eur_norm.index, y=prices_eur_norm, fill='tonexty', mode='lines',
                         line=dict(width=0), fillcolor='rgba(255,107,107,0.15)', name='Efecto FX'), row=1, col=1)

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
st.caption("¿En qué años te benefició o perjudicó el tipo de cambio?")

annual_df = calculate_annual_breakdown(prices[asset_to_show], eurusd)

if len(annual_df) > 0:
    # Gráfico de barras por año
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(name='Retorno USD', x=annual_df['Año'], y=annual_df['Ret. USD'],
                          marker_color=COLORS['usd']))
    fig2.add_trace(go.Bar(name='Retorno EUR', x=annual_df['Año'], y=annual_df['Ret. EUR'],
                          marker_color=COLORS['eur']))
    fig2.add_trace(go.Scatter(name='Impacto FX', x=annual_df['Año'], y=annual_df['Impacto FX'],
                              mode='lines+markers', line=dict(color=COLORS['gold'], width=2),
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
    disp['FX Ayudó'] = disp['FX Ayudó'].apply(lambda x: "✅ Sí" if x else "❌ No")
    st.dataframe(disp, hide_index=True, use_container_width=True)
    
    # Stats
    años_favor = annual_df['FX Ayudó'].sum()
    años_contra = len(annual_df) - años_favor
    st.markdown(f"**Resumen:** El FX te benefició en **{años_favor}** años y te perjudicó en **{años_contra}** años.")

# ══════════════════════════════════════════════════════════════════════════════
# DRAWDOWN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📉 Comparación de Drawdowns")
st.caption("¿Cuánto más (o menos) caíste por el efecto divisa?")

dd_usd = calculate_drawdowns(prices[asset_to_show])
dd_eur = calculate_drawdowns(prices[asset_to_show] / eurusd)

c1, c2, c3 = st.columns(3)
c1.metric("Max Drawdown (USD)", f"{dd_usd['max_drawdown']:.1f}%")
c2.metric("Max Drawdown (EUR)", f"{dd_eur['max_drawdown']:.1f}%")
diff_dd = dd_eur['max_drawdown'] - dd_usd['max_drawdown']
c3.metric("Diferencia", f"{diff_dd:+.1f}pp", delta_color="inverse")

if diff_dd < 0:
    st.success(f"✅ El efecto divisa **amortiguó** la caída máxima en {abs(diff_dd):.1f}pp")
else:
    st.error(f"⚠️ El efecto divisa **amplificó** la caída máxima en {diff_dd:.1f}pp")

# ══════════════════════════════════════════════════════════════════════════════
# IMPACTO FX ROLLING (1 AÑO)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🔄 Impacto FX Rolling (1 año)")
st.caption("Cómo ha variado el impacto del tipo de cambio en ventanas de 1 año")

rolling_data = calculate_rolling_fx_impact(prices[asset_to_show], eurusd, 252)

if len(rolling_data) > 50:
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(x=rolling_data.index, y=rolling_data['fx_impact'],
                              name='Impacto FX (1Y rolling)',
                              line=dict(color=COLORS['gold'], width=2),
                              fill='tozeroy', fillcolor='rgba(212,175,55,0.2)'))
    
    fig3.add_hline(y=0, line_dash="dash", line_color=COLORS['text_secondary'])
    fig3.add_hline(y=rolling_data['fx_impact'].mean(), line_dash="dot", line_color=COLORS['blue'],
                   annotation_text=f"Media: {rolling_data['fx_impact'].mean():.1f}pp")
    
    fig3.update_layout(**base_layout(300), yaxis_title="Impacto FX (pp)")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media", f"{rolling_data['fx_impact'].mean():+.1f}pp")
    c2.metric("Mejor momento", f"{rolling_data['fx_impact'].max():+.1f}pp")
    c3.metric("Peor momento", f"{rolling_data['fx_impact'].min():+.1f}pp")
    c4.metric("Volatilidad", f"{rolling_data['fx_impact'].std():.1f}pp")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARATIVA TODOS LOS ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════

if len(selected_assets) > 1:
    st.markdown("### 📊 Comparativa: Todos los Activos")
    
    comparison_data = []
    for asset in selected_assets:
        dc = decompose_returns_exact(prices[asset], eurusd)
        comparison_data.append({
            'Activo': asset,
            'Ret. USD': dc['ret_usd'],
            'Ret. EUR': dc['ret_eur'],
            'Impacto FX': dc['ret_eur'] - dc['ret_usd']
        })
    
    comp_df = pd.DataFrame(comparison_data).sort_values('Ret. EUR', ascending=False)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Retorno USD', x=comp_df['Activo'], y=comp_df['Ret. USD'], marker_color=COLORS['usd']))
    fig4.add_trace(go.Bar(name='Retorno EUR', x=comp_df['Activo'], y=comp_df['Ret. EUR'], marker_color=COLORS['eur']))
    fig4.update_layout(**base_layout(350), barmode='group',
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
    st.plotly_chart(fig4, use_container_width=True)
    
    disp2 = comp_df.copy()
    disp2['Ret. USD'] = disp2['Ret. USD'].apply(lambda x: f"{x:+.1f}%")
    disp2['Ret. EUR'] = disp2['Ret. EUR'].apply(lambda x: f"{x:+.1f}%")
    disp2['Impacto FX'] = disp2['Impacto FX'].apply(lambda x: f"{x:+.1f}pp")
    st.dataframe(disp2, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="insight-box">
<h4>🎓 Conclusiones para el Inversor Europeo</h4>
<p>
<strong>1. El riesgo divisa es significativo y bidireccional:</strong> Puede sumar o restar varios puntos porcentuales a tu rentabilidad cada año.
<br><br>
<strong>2. No es predecible:</strong> El EUR/USD depende de política monetaria, geopolítica y flujos de capital. Cubrirlo tiene coste.
<br><br>
<strong>3. A largo plazo tiende a neutralizarse:</strong> En períodos largos (+10 años), el impacto acumulado suele ser menor.
<br><br>
<strong>4. Opciones de gestión:</strong><br>
• <strong>Aceptar el riesgo:</strong> Diversificación natural si también tienes gastos en USD<br>
• <strong>ETFs hedged:</strong> Ej: iShares S&P 500 EUR Hedged (IUSE.DE)<br>
• <strong>Cobertura activa:</strong> Futuros/forwards de EUR/USD (coste ≈ diferencial de tipos)
</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div style="text-align:center;color:#666;font-size:0.75rem;padding:1rem;">FX Impact Analyzer • {start_date:%d/%m/%Y} - {end_date:%d/%m/%Y} • Solo fines educativos</div>', unsafe_allow_html=True)
