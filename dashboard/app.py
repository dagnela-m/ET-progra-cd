"""
Dashboard - Estimación de emisiones de CO₂ por país
ET Programación para Ciencia de Datos

Se debe correr desde la raíz del proyecto:
    python dashboard/app.py

El modelo se reentrena al arrancar el dashboard (no toca el cuaderno).
"""

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
CSV_PATH = "data/processed/sustainable_energy_clean.csv"

# ---------------------------------------------------------------------------
# 0. Paleta y estilos compartidos
# ---------------------------------------------------------------------------

NAVY = "#002138"
NAVY2 = "#0E3A5C"
GOLD = "#F8B31C"
CORAL = "#FF7F50"
GRAY = "#596B7D"
GRAY_L = "#8A97A6"
BG = "#F4F7FA"
CARD = "#FFFFFF"
SAND = "#FEF7E8"
LINE = "#E4EAF1"

FUENTE = "'Quicksand', 'Trebuchet MS', sans-serif"

# Escala de color propia, del arena al navy pasando por el coral
ESCALA_CO2 = ["#FEF7E8", "#FBD98A", GOLD, CORAL, "#C4452F", NAVY]


def estilizar(fig, altura=430):
    """Aplica el mismo tratamiento visual a todas las figuras."""
    fig.update_layout(
        height=altura,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FUENTE, size=12, color=GRAY),
        title=dict(font=dict(size=15, color=NAVY), x=0, xanchor="left", pad=dict(b=12)),
        margin=dict(l=55, r=25, t=55, b=45),
        hoverlabel=dict(
            bgcolor=CARD, bordercolor=LINE,
            font=dict(family=FUENTE, size=12, color=NAVY)
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=LINE, borderwidth=0,
            font=dict(size=11)
        ),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(size=11))
    return fig


# Estilos de tarjeta reutilizables
CARD_STYLE = {
    "backgroundColor": CARD,
    "borderRadius": "18px",
    "padding": "22px 26px",
    "boxShadow": "0 2px 14px rgba(0, 33, 56, 0.07)",
    "marginBottom": "22px",
}

KPI_STYLE = {
    "backgroundColor": CARD,
    "borderRadius": "18px",
    "padding": "20px 24px",
    "boxShadow": "0 2px 14px rgba(0, 33, 56, 0.07)",
    "flex": "1",
    "minWidth": "180px",
}

# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

feature_cols = [
    "Acceso_electricidad", "Acceso_combustible_limpio", "Capacidad_renovable_pc",
    "Renovable_pct", "Elec_fossil", "Elec_nuclear", "Elec_renovables",
    "Bajo_carbon_pct", "Consumo_ener_pc", "Intensidad_energia",
    "Gdp_crecimiento", "Gdp_pc", "Densidad", "Area_Kkm2",
    "Poblacion", "Urbanizacion", "renovable_ratio"
]

X = df[feature_cols]
y = df["Emision_CO2"]

# ---------------------------------------------------------------------------
# 2. Reentrenamiento del modelo (mismo split y random_state que el cuaderno)
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo_final = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
modelo_final.fit(X_train_scaled, y_train)

y_pred_final = modelo_final.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

df_pred = pd.DataFrame({
    "real": y_test.values,
    "predicho": y_pred_final,
    "Pais": df.loc[X_test.index, "Pais"].values,
    "Anio": df.loc[X_test.index, "Anio"].values
})

df_importancia = pd.DataFrame({
    "feature": feature_cols,
    "importancia": modelo_final.feature_importances_
}).sort_values("importancia", ascending=True)

# ---------------------------------------------------------------------------
# 3. Datos auxiliares para los gráficos
# ---------------------------------------------------------------------------

anio_min = int(df["Anio"].min())
anio_max = int(df["Anio"].max())

paises_disponibles = sorted(df["Pais"].unique())

top_emisores_default = (
    df.groupby("Pais")["Emision_CO2"].mean()
    .sort_values(ascending=False).head(5).index.tolist()
)

PALETA_LINEAS = [CORAL, NAVY2, GOLD, "#4EA5A0", "#B85C8A", "#7B68A6", "#C4452F", "#5A8FBF"]

# ---------------------------------------------------------------------------
# 4. Funciones que generan cada figura
# ---------------------------------------------------------------------------

def crear_choropleth(anio):
    """1.1 Choropleth de emisiones por país en un año dado, escala de color log."""
    dfa = df[df["Anio"] == anio].copy()
    dfa["Emision_CO2_log"] = np.log1p(dfa["Emision_CO2"])

    fig = px.choropleth(
        dfa,
        locations="iso3",
        color="Emision_CO2_log",
        hover_name="Pais",
        hover_data={"Emision_CO2": ":,.0f", "Emision_CO2_log": False, "iso3": False},
        color_continuous_scale=ESCALA_CO2,
        labels={"Emision_CO2": "Emisiones (kt)"},
        title=f"Emisiones de CO₂ por país · {anio}",
    )
    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        showland=True, landcolor="#EDF1F6",
        showocean=True, oceancolor="#F8FAFC",
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(marker_line_color=CARD, marker_line_width=0.6)
    estilizar(fig, altura=470)
    fig.update_layout(
        margin=dict(l=0, r=0, t=55, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="log(kt CO₂)", font=dict(size=11)),
            thickness=12, len=0.7, outlinewidth=0, tickfont=dict(size=10),
        ),
    )
    return fig


def crear_series_temporales(paises_seleccionados):
    """1.2 Series temporales de emisiones por país, con dropdown."""
    if not paises_seleccionados:
        paises_seleccionados = top_emisores_default

    dff = df[df["Pais"].isin(paises_seleccionados)]

    fig = px.line(
        dff, x="Anio", y="Emision_CO2", color="Pais",
        labels={"Anio": "Año", "Emision_CO2": "Emisiones de CO₂ (kt)"},
        title=f"Evolución de emisiones de CO₂ · {anio_min}-{anio_max}",
        color_discrete_sequence=PALETA_LINEAS,
    )
    fig.update_traces(line=dict(width=2.5), mode="lines")
    fig.update_layout(hovermode="x unified")
    return estilizar(fig)


def crear_predicho_vs_real():
    """1.3 Scatter predicho vs real en escala log, con diagonal de referencia."""
    lim_min = min(df_pred["real"].min(), df_pred["predicho"].min())
    lim_max = max(df_pred["real"].max(), df_pred["predicho"].max())

    fig = px.scatter(
        df_pred, x="real", y="predicho",
        log_x=True, log_y=True,
        opacity=0.55,
        hover_data=["Pais", "Anio"],
        labels={"real": "Emisiones reales (kt, escala log)",
                "predicho": "Emisiones predichas (kt, escala log)"},
        title=f"Predicho vs real · Random Forest",
    )
    fig.update_traces(marker=dict(size=6, color=CORAL, line=dict(width=0)))

    fig.add_trace(go.Scatter(
        x=[lim_min, lim_max], y=[lim_min, lim_max],
        mode="lines",
        line=dict(color=GRAY_L, width=1.5, dash="dash"),
        name="Predicción perfecta",
        hoverinfo="skip",
    ))
    # La diagonal se dibuja detrás de los puntos
    fig.data = (fig.data[1], fig.data[0])
    return estilizar(fig)


def crear_importancia():
    """1.4 Barra horizontal de importancia de variables del Random Forest."""
    fig = px.bar(
        df_importancia, x="importancia", y="feature",
        orientation="h",
        color="importancia",
        color_continuous_scale=[SAND, GOLD, CORAL],
        labels={"importancia": "Importancia", "feature": ""},
        title="Importancia de variables · Random Forest",
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    )
    estilizar(fig, altura=470)
    fig.update_layout(
        margin=dict(l=150, r=25, t=55, b=45),
        coloraxis_showscale=False,
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=False, linecolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------------------
# 5. Componentes reutilizables del layout
# ---------------------------------------------------------------------------

def kpi(valor, etiqueta, detalle):
    return html.Div(
        className="_card",
        style=KPI_STYLE,
        children=[
            html.Div(valor, style={
                "fontSize": "30px", "fontWeight": "700", "color": GOLD,
                "lineHeight": "1.1", "marginBottom": "6px"
            }),
            html.Div(etiqueta, style={
                "fontSize": "13px", "fontWeight": "600", "color": NAVY, "marginBottom": "3px"
            }),
            html.Div(detalle, style={"fontSize": "11.5px", "color": GRAY_L}),
        ],
    )


def seccion(numero, titulo, bajada, hijos):
    return html.Div(
        className="_card",
        style=CARD_STYLE,
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "4px"},
                children=[
                    html.Div(numero, style={
                        "backgroundColor": GOLD, "color": NAVY, "borderRadius": "50%",
                        "width": "28px", "height": "28px", "display": "flex",
                        "alignItems": "center", "justifyContent": "center",
                        "fontSize": "13px", "fontWeight": "700", "flexShrink": "0",
                    }),
                    html.H3(titulo, style={
                        "margin": "0", "fontSize": "18px", "fontWeight": "700", "color": NAVY
                    }),
                ],
            ),
            html.P(bajada, style={
                "margin": "0 0 16px 40px", "fontSize": "12.5px", "color": GRAY_L
            }),
            *hijos,
        ],
    )


# ---------------------------------------------------------------------------
# 6. App y layout
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap"
    ],
)
app.title = "Estimación de emisiones de CO₂ por país"

# CSS que Dash no permite escribir inline: hover, scrollbar, controles
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background: linear-gradient(160deg, #F4F7FA 0%, #EEF3F9 55%, #FDF6E9 100%);
                background-attachment: fixed;
                font-family: 'Quicksand', 'Trebuchet MS', sans-serif;
            }
            ._card:hover {
                box-shadow: 0 6px 24px rgba(0, 33, 56, 0.13) !important;
                transform: translateY(-2px);
            }
            ._card {
                transition: box-shadow .25s ease, transform .25s ease;
            }
            .Select-control, .is-focused .Select-control {
                border-radius: 12px !important;
                border-color: #E4EAF1 !important;
            }
            .Select--multi .Select-value {
                background-color: #FEF7E8 !important;
                border-color: #F8B31C !important;
                color: #002138 !important;
                border-radius: 8px !important;
                font-weight: 600;
            }
            .rc-slider-track { background-color: #F8B31C !important; height: 5px !important; }
            .rc-slider-rail  { background-color: #E4EAF1 !important; height: 5px !important; }
            .rc-slider-handle {
                border-color: #F8B31C !important;
                background-color: #FFFFFF !important;
                width: 17px !important; height: 17px !important;
                margin-top: -6px !important;
                box-shadow: 0 1px 5px rgba(0,33,56,.2) !important;
            }
            .rc-slider-dot { border-color: #E4EAF1 !important; }
            .rc-slider-dot-active { border-color: #F8B31C !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""

app.layout = html.Div(
    style={"maxWidth": "1180px", "margin": "0 auto", "padding": "30px 24px 50px 24px"},
    children=[

        # ---- Encabezado ----
        html.Div(
            style={
                "background": f"linear-gradient(120deg, {NAVY} 0%, {NAVY2} 100%)",
                "borderRadius": "22px",
                "padding": "34px 38px",
                "marginBottom": "22px",
                "boxShadow": "0 4px 20px rgba(0, 33, 56, 0.18)",
            },
            children=[
                html.Div("PROGRAMACIÓN PARA CIENCIA DE DATOS", style={
                    "color": GOLD, "fontSize": "11px", "fontWeight": "700",
                    "letterSpacing": "2px", "marginBottom": "10px",
                }),
                html.H1("Estimación de emisiones de CO₂ por país", style={
                    "margin": "0 0 10px 0", "color": "#FFFFFF",
                    "fontSize": "33px", "fontWeight": "700",
                }),
                html.P(
                    "Un modelo de machine learning para estimar las emisiones de CO₂ de países "
                    "que reportan cómo generan su energía, pero no cuánto contaminan.",
                    style={"margin": "0", "color": "#B9CADA", "fontSize": "14px", "maxWidth": "680px"},
                ),
            ],
        ),

        # ---- KPIs ----
        html.Div(
            className="_kpis",
            style={"display": "flex", "gap": "18px", "marginBottom": "22px", "flexWrap": "wrap"},
            children=[
                kpi(f"{r2:.4f}", "R² del modelo final", "Random Forest sobre el conjunto de test"),
                kpi(f"{mae:,.0f}".replace(",", "."), "MAE en kt de CO₂", "Error absoluto medio"),
                kpi(f"{rmse:,.0f}".replace(",", "."), "RMSE en kt de CO₂", "Castiga más los errores grandes"),
                kpi(f"{len(df):,}".replace(",", "."), "Observaciones país-año", f"{df['Pais'].nunique()} países · {anio_min}-{anio_max}"),
            ],
        ),

        # ---- 1. Choropleth ----
        seccion(
            "1", "Emisiones de CO₂ en el mapa",
            "La escala de color va en logarítmica: en lineal, China y Estados Unidos se comen todo el rango y el resto del mundo queda del mismo color. Mueve el slider para recorrer los años.",
            [
                dcc.Graph(id="grafico-choropleth", config={"displayModeBar": False}),
                html.Div(
                    style={"padding": "6px 30px 0 30px"},
                    children=[
                        dcc.Slider(
                            id="slider-anio",
                            min=anio_min, max=anio_max, step=1, value=anio_max,
                            marks={a: {"label": str(a), "style": {"fontSize": "11px", "color": GRAY_L}}
                                   for a in range(anio_min, anio_max + 1, 2)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
            ],
        ),

        # ---- 2. Series temporales ----
        seccion(
            "2", "Evolución en el tiempo",
            "Elige los países que quieras comparar. Por defecto vienen los cinco de mayor emisión promedio.",
            [
                dcc.Dropdown(
                    id="dropdown-paises",
                    options=[{"label": p, "value": p} for p in paises_disponibles],
                    value=top_emisores_default,
                    multi=True,
                    placeholder="Selecciona países a comparar...",
                    style={"marginBottom": "14px"},
                ),
                dcc.Graph(id="grafico-series-temporales", config={"displayModeBar": False}),
            ],
        ),

        # ---- 3 y 4 en dos columnas ----
        html.Div(
            style={"display": "flex", "gap": "22px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1", "minWidth": "440px"},
                    children=[
                        seccion(
                            "3", "Predicho vs real",
                            "Cada punto es una observación del conjunto de test. Mientras más cerca de la diagonal, mejor la estimación.",
                            [dcc.Graph(id="grafico-predicho-real",
                                       figure=crear_predicho_vs_real(),
                                       config={"displayModeBar": False})],
                        )
                    ],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "440px"},
                    children=[
                        seccion(
                            "4", "¿De dónde sale la estimación?",
                            "Peso de cada variable dentro del Random Forest, ordenado de menor a mayor.",
                            [dcc.Graph(id="grafico-importancia",
                                       figure=crear_importancia(),
                                       config={"displayModeBar": False})],
                        )
                    ],
                ),
            ],
        ),

        # ---- Pie ----
        html.Div(
            style={
                "textAlign": "center", "color": GRAY_L, "fontSize": "11.5px",
                "marginTop": "10px", "paddingTop": "18px", "borderTop": f"1px solid {LINE}",
            },
            children=[
                html.Span("Areliz Isla Treuque  ·  Daniela Montefinale Middleton"),
                html.Br(),
                html.Span("Dataset: Global Data on Sustainable Energy (Kaggle) enriquecido con la API del World Bank",
                          style={"fontSize": "11px"}),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# 7. Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("grafico-choropleth", "figure"),
    Input("slider-anio", "value")
)
def actualizar_choropleth(anio):
    return crear_choropleth(anio)


@app.callback(
    Output("grafico-series-temporales", "figure"),
    Input("dropdown-paises", "value")
)
def actualizar_series_temporales(paises_seleccionados):
    return crear_series_temporales(paises_seleccionados)


# ---------------------------------------------------------------------------
# 8. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
