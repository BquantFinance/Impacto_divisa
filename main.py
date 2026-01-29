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
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
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
    
    .comparison-table {
        background: #1a1a24; border-radius: 12px; padding: 1.5rem;
        border: 1px solid #2a2a3a;
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

def calculate_eur_returns(prices_usd: pd.Series, eurusd: pd.Series) -> pd.Series:
    """
    Convierte retornos USD a EUR para un inversor europeo.
    Si EUR/USD sube (EUR se fortalece), el inversor europeo pierde valor en sus activos USD.
    Precio en EUR = Precio en USD / EUR/USD
    """
    return prices_usd / eurusd

def decompose_returns(prices_usd: pd.Series, eurusd: pd.Series) -> pd.DataFrame:
    """Descompone el retorno total en EUR en: retorno activo (USD) + efecto divisa"""
    # Retorno del activo en USD
    ret_usd = (prices_usd.iloc[-1] / prices_usd.iloc[0] - 1) * 100
    
    # Retorno del tipo de cambio (si EUR/USD sube, efecto negativo para inversor EUR)
    fx_effect = -(eurusd.iloc[-1] / eurusd.iloc[0] - 1) * 100
    
    # Retorno total en EUR
    prices_eur = prices_usd / eurusd
    ret_eur = (prices_eur.iloc[-1] / prices_eur.iloc[0] - 1) * 100
    
    return {
        'ret_usd': ret_usd,
        'fx_effect': fx_effect,
        'ret_eur': ret_eur,
        'fx_contribution_pct': (fx_effect / abs(ret_eur) * 100) if ret_eur != 0 else 0
    }

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
    start_date = c1.date_input("Inicio", value=datetime.now() - timedelta(days=365*5))
    end_date = c2.date_input("Fin", value=datetime.now())
    
    st.markdown("---")
    st.markdown("#### 📊 Activos USA")
    
    PRESETS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "VTI": "Total US Market",
               "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon", "NVDA": "Nvidia"}
    
    preset_selection = st.multiselect("Activos predefinidos:", list(PRESETS.keys()), default=["SPY", "QQQ"],
                                       format_func=lambda x: f"{x} ({PRESETS[x]})")
    
    custom_tickers = st.text_input("Añadir tickers:", placeholder="TSLA, META...")
    
    selected_assets = list(preset_selection)
    if custom_tickers:
        selected_assets.extend([t.strip().upper() for t in custom_tickers.split(",") if t.strip() and t.strip().upper() not in selected_assets])
    
    st.markdown("---")
    st.markdown("#### 💼 Cartera Personalizada")
    use_portfolio = st.checkbox("Activar simulador de cartera", value=False)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🇪🇺 FX Impact Analyzer</h1>
    <p>Cómo afecta el EUR/USD a tus inversiones en USA</p>
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
    st.error("Ningún activo válido encontrado.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

# Calcular para el primer activo como ejemplo principal
main_asset = selected_assets[0]
decomp = decompose_returns(prices[main_asset], eurusd)
fx_change = (eurusd.iloc[-1] / eurusd.iloc[0] - 1) * 100

cols = st.columns(5)
metrics = [
    ("EUR/USD", f"{eurusd.iloc[-1]:.4f}", f"{fx_change:+.1f}%", "gold"),
    (f"{main_asset} (USD)", f"{decomp['ret_usd']:+.1f}%", "Retorno local", "positive" if decomp['ret_usd'] >= 0 else "negative"),
    ("Efecto Divisa", f"{decomp['fx_effect']:+.1f}%", "Para inversor EUR", "positive" if decomp['fx_effect'] >= 0 else "negative"),
    (f"{main_asset} (EUR)", f"{decomp['ret_eur']:+.1f}%", "Retorno real", "positive" if decomp['ret_eur'] >= 0 else "negative"),
    ("Impacto FX", f"{abs(decomp['fx_effect']):.1f}pp", "del retorno total", "blue"),
]

for col, (label, value, subtitle, cls) in zip(cols, metrics):
    col.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
        <div class="metric-label" style="margin-top:4px;font-size:0.6rem;">{subtitle}</div>
    </div>''', unsafe_allow_html=True)

# Insight dinámico
if decomp['fx_effect'] < 0:
    insight_text = f"""El EUR se ha <strong>fortalecido</strong> un {abs(fx_change):.1f}% frente al USD en este período. 
    Esto ha <strong>restado {abs(decomp['fx_effect']):.1f} puntos porcentuales</strong> a tu inversión en {main_asset}. 
    Mientras el activo subió {decomp['ret_usd']:.1f}% en USD, un inversor europeo solo ganó {decomp['ret_eur']:.1f}% en EUR."""
else:
    insight_text = f"""El EUR se ha <strong>debilitado</strong> un {abs(fx_change):.1f}% frente al USD en este período. 
    Esto ha <strong>sumado {decomp['fx_effect']:.1f} puntos porcentuales</strong> a tu inversión en {main_asset}. 
    El activo subió {decomp['ret_usd']:.1f}% en USD, pero un inversor europeo ganó {decomp['ret_eur']:.1f}% en EUR."""

st.markdown(f'<div class="insight-box"><h4>📊 Resumen del Impacto</h4><p>{insight_text}</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO PRINCIPAL: USD vs EUR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Retorno Acumulado: USD vs EUR")
st.caption("Compara lo que ve un inversor americano (USD) vs lo que realmente gana un inversor europeo (EUR)")

# Selector de activo
asset_to_show = st.selectbox("Seleccionar activo:", selected_assets, index=0)

# Calcular series
prices_usd_norm = (prices[asset_to_show] / prices[asset_to_show].iloc[0]) * 100
prices_eur = prices[asset_to_show] / eurusd
prices_eur_norm = (prices_eur / prices_eur.iloc[0]) * 100
eurusd_norm = (eurusd / eurusd.iloc[0]) * 100

# Gráfico
fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=[f"{asset_to_show}: Retorno en USD vs EUR", "EUR/USD"])

# Panel superior: USD vs EUR
fig.add_trace(go.Scatter(x=prices_usd_norm.index, y=prices_usd_norm, name=f'{asset_to_show} (USD)',
                         line=dict(color=COLORS['usd'], width=2.5),
                         hovertemplate='USD: %{y:.1f}<extra></extra>'), row=1, col=1)

fig.add_trace(go.Scatter(x=prices_eur_norm.index, y=prices_eur_norm, name=f'{asset_to_show} (EUR)',
                         line=dict(color=COLORS['eur'], width=2.5),
                         hovertemplate='EUR: %{y:.1f}<extra></extra>'), row=1, col=1)

# Área entre ambas líneas para mostrar el gap
fig.add_trace(go.Scatter(x=prices_usd_norm.index, y=prices_usd_norm, fill=None, mode='lines',
                         line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
fig.add_trace(go.Scatter(x=prices_eur_norm.index, y=prices_eur_norm, fill='tonexty', mode='lines',
                         line=dict(width=0), fillcolor='rgba(255,107,107,0.15)',
                         name='Efecto FX', hoverinfo='skip'), row=1, col=1)

fig.add_hline(y=100, line_dash="dash", line_color=COLORS['grid'], opacity=0.5, row=1, col=1)

# Panel inferior: EUR/USD
fig.add_trace(go.Scatter(x=eurusd.index, y=eurusd, name='EUR/USD',
                         line=dict(color=COLORS['gold'], width=2),
                         hovertemplate='EUR/USD: %{y:.4f}<extra></extra>'), row=2, col=1)

fig.update_layout(
    **base_layout(550),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    hovermode='x unified'
)
fig.update_yaxes(title_text="Base 100", row=1, col=1, gridcolor=COLORS['grid'])
fig.update_yaxes(title_text="EUR/USD", row=2, col=1, gridcolor=COLORS['grid'])
fig.update_xaxes(gridcolor=COLORS['grid'])

st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABLA COMPARATIVA TODOS LOS ACTIVOS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📊 Comparativa: Todos los Activos")

comparison_data = []
for asset in selected_assets:
    d = decompose_returns(prices[asset], eurusd)
    comparison_data.append({
        'Activo': asset,
        'Retorno USD': d['ret_usd'],
        'Efecto FX': d['fx_effect'],
        'Retorno EUR': d['ret_eur'],
        'Diferencia': d['ret_eur'] - d['ret_usd']
    })

comp_df = pd.DataFrame(comparison_data).sort_values('Retorno EUR', ascending=False)

# Gráfico de barras comparativo
fig2 = go.Figure()

fig2.add_trace(go.Bar(name='Retorno USD', x=comp_df['Activo'], y=comp_df['Retorno USD'],
                      marker_color=COLORS['usd'], text=[f"{v:+.1f}%" for v in comp_df['Retorno USD']],
                      textposition='outside'))

fig2.add_trace(go.Bar(name='Retorno EUR', x=comp_df['Activo'], y=comp_df['Retorno EUR'],
                      marker_color=COLORS['eur'], text=[f"{v:+.1f}%" for v in comp_df['Retorno EUR']],
                      textposition='outside'))

fig2.update_layout(**base_layout(400), barmode='group',
                   title=dict(text="Retorno USD vs EUR por Activo", font=dict(size=16, color=COLORS['gold'])),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))

st.plotly_chart(fig2, use_container_width=True)

# Tabla con formato
st.markdown("#### Detalle Numérico")
display_df = comp_df.copy()
display_df['Retorno USD'] = display_df['Retorno USD'].apply(lambda x: f"{x:+.2f}%")
display_df['Efecto FX'] = display_df['Efecto FX'].apply(lambda x: f"{x:+.2f}%")
display_df['Retorno EUR'] = display_df['Retorno EUR'].apply(lambda x: f"{x:+.2f}%")
display_df['Diferencia'] = display_df['Diferencia'].apply(lambda x: f"{x:+.2f}pp")
st.dataframe(display_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULADOR DE CARTERA
# ══════════════════════════════════════════════════════════════════════════════

if use_portfolio:
    st.markdown("---")
    st.markdown("### 💼 Simulador de Cartera")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("#### Composición")
        weights = {}
        for asset in selected_assets:
            weights[asset] = st.number_input(f"{asset} (%)", 0, 100, 100 // len(selected_assets), key=f"pw_{asset}") / 100
        
        total = sum(weights.values()) * 100
        if abs(total - 100) > 0.1:
            st.warning(f"⚠️ Total: {total:.0f}%")
        else:
            st.success(f"✓ Total: {total:.0f}%")
    
    with c2:
        if sum(weights.values()) > 0:
            # Calcular cartera
            port_usd = sum(prices[a] * weights[a] for a in selected_assets if weights[a] > 0)
            port_usd = port_usd / port_usd.iloc[0] * 100  # Normalizar
            
            port_eur = sum((prices[a] / eurusd) * weights[a] for a in selected_assets if weights[a] > 0)
            port_eur = port_eur / port_eur.iloc[0] * 100
            
            ret_usd = port_usd.iloc[-1] - 100
            ret_eur = port_eur.iloc[-1] - 100
            fx_eff = ret_eur - ret_usd
            
            st.markdown("#### Resultado de tu Cartera")
            
            cols = st.columns(3)
            cols[0].metric("Retorno USD", f"{ret_usd:+.2f}%")
            cols[1].metric("Efecto FX", f"{fx_eff:+.2f}%", delta_color="inverse" if fx_eff < 0 else "normal")
            cols[2].metric("Retorno EUR", f"{ret_eur:+.2f}%")
            
            # Gráfico cartera
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=port_usd.index, y=port_usd, name='Cartera (USD)',
                                      line=dict(color=COLORS['usd'], width=2.5)))
            fig3.add_trace(go.Scatter(x=port_eur.index, y=port_eur, name='Cartera (EUR)',
                                      line=dict(color=COLORS['eur'], width=2.5)))
            fig3.add_hline(y=100, line_dash="dash", line_color=COLORS['grid'])
            fig3.update_layout(**base_layout(350),
                               title=dict(text="Evolución de tu Cartera", font=dict(size=14, color=COLORS['gold'])),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
            st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ESCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🎯 Escenarios: ¿Y si el EUR/USD cambia?")
st.caption("Impacto en una inversión de 10.000€ con el retorno actual del activo")

scenario_asset = st.selectbox("Activo para escenarios:", selected_assets, key="scenario_asset")
d = decompose_returns(prices[scenario_asset], eurusd)

investment = 10000
scenarios = [
    ("EUR/USD -15%", -15, "EUR débil"),
    ("EUR/USD -10%", -10, "EUR débil"),
    ("EUR/USD -5%", -5, "EUR débil"),
    ("Actual", 0, "Actual"),
    ("EUR/USD +5%", 5, "EUR fuerte"),
    ("EUR/USD +10%", 10, "EUR fuerte"),
    ("EUR/USD +15%", 15, "EUR fuerte"),
]

scenario_data = []
for name, fx_chg, tipo in scenarios:
    # Nuevo efecto FX (inverso: si EUR/USD sube, pierdes)
    new_fx_effect = -fx_chg
    new_ret_eur = d['ret_usd'] + new_fx_effect
    final_value = investment * (1 + new_ret_eur / 100)
    profit = final_value - investment
    
    scenario_data.append({
        'Escenario': name,
        'Ret. Activo (USD)': d['ret_usd'],
        'Efecto FX': new_fx_effect,
        'Ret. Total (EUR)': new_ret_eur,
        'Valor Final': final_value,
        'Ganancia': profit
    })

scen_df = pd.DataFrame(scenario_data)

# Gráfico de escenarios
fig4 = go.Figure()

colors_scen = [COLORS['positive'] if x >= 0 else COLORS['negative'] for x in scen_df['Ganancia']]

fig4.add_trace(go.Bar(
    x=scen_df['Escenario'], y=scen_df['Ganancia'],
    marker_color=colors_scen,
    text=[f"€{v:+,.0f}" for v in scen_df['Ganancia']],
    textposition='outside'
))

fig4.add_hline(y=0, line_color=COLORS['grid'])
fig4.update_layout(**base_layout(350),
                   title=dict(text=f"Ganancia/Pérdida en 10.000€ invertidos en {scenario_asset}", 
                              font=dict(size=14, color=COLORS['gold'])),
                   yaxis_title="€")

st.plotly_chart(fig4, use_container_width=True)

# Tabla escenarios
st.markdown("#### Detalle de Escenarios")
disp_scen = scen_df.copy()
disp_scen['Ret. Activo (USD)'] = disp_scen['Ret. Activo (USD)'].apply(lambda x: f"{x:+.1f}%")
disp_scen['Efecto FX'] = disp_scen['Efecto FX'].apply(lambda x: f"{x:+.1f}%")
disp_scen['Ret. Total (EUR)'] = disp_scen['Ret. Total (EUR)'].apply(lambda x: f"{x:+.1f}%")
disp_scen['Valor Final'] = disp_scen['Valor Final'].apply(lambda x: f"€{x:,.0f}")
disp_scen['Ganancia'] = disp_scen['Ganancia'].apply(lambda x: f"€{x:+,.0f}")
st.dataframe(disp_scen, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONCLUSIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="insight-box">
<h4>🎓 Conclusión para Inversores Europeos</h4>
<p>
<strong>El riesgo divisa es real y significativo.</strong> Cuando inviertes en activos denominados en USD:
<br><br>
• Si el <strong>EUR se fortalece</strong> (EUR/USD sube) → Pierdes valor al convertir a EUR<br>
• Si el <strong>EUR se debilita</strong> (EUR/USD baja) → Ganas valor adicional al convertir a EUR<br>
<br>
<strong>Opciones para gestionar este riesgo:</strong><br>
1. Aceptar el riesgo divisa como parte de la diversificación global<br>
2. Usar ETFs con cobertura de divisa (hedged) como CSPX.L (EUR hedged)<br>
3. Cubrir manualmente con futuros o forwards de EUR/USD
</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown(f'<div style="text-align:center;color:#a0a0b0;font-size:0.75rem;padding:1rem;">FX Impact Analyzer • {start_date:%d/%m/%Y} - {end_date:%d/%m/%Y} • Solo fines educativos</div>', unsafe_allow_html=True)
