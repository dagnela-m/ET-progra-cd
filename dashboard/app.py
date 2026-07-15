"""
Dashboard - Estimación de emisiones de CO2 por país
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

# DataFrame para el scatter predicho vs real
df_pred = pd.DataFrame({
    "real": y_test.values,
    "predicho": y_pred_final,
    "Pais": df.loc[X_test.index, "Pais"].values,
    "Anio": df.loc[X_test.index, "Anio"].values
})

# DataFrame de importancia de variables
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
        hover_data={"Emision_CO2": ":.0f", "Emision_CO2_log": False, "iso3": False},
        color_continuous_scale="YlOrRd",
        labels={"Emision_CO2_log": "log(CO2 + 1)"},
        title=f"Emisiones de CO2 por país - {anio}",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="log(kt CO2)"),
    )
    return fig


def crear_series_temporales(paises_seleccionados):
    """1.2 Series temporales de emisiones por país, con dropdown."""
    if not paises_seleccionados:
        paises_seleccionados = top_emisores_default

    dff = df[df["Pais"].isin(paises_seleccionados)]

    fig = px.line(
        dff, x="Anio", y="Emision_CO2", color="Pais",
        labels={"Anio": "Año", "Emision_CO2": "Emisiones de CO2 (kt)"},
        title="Evolución de emisiones de CO2 (2000-2019)",
    )
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return fig


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
        title=f"Predicho vs real (Random Forest) - R2 = {r2:.4f}",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=5, color="#FF7F50"))

    fig.add_trace(go.Scatter(
        x=[lim_min, lim_max], y=[lim_min, lim_max],
        mode="lines",
        line=dict(color="#4A4A4A", width=1.5, dash="dash"),
        name="Predicción perfecta"
    ))
    fig.data = (fig.data[1], fig.data[0])
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return fig


def crear_importancia():
    """1.4 Barra horizontal de importancia de variables del Random Forest."""
    fig = px.bar(
        df_importancia, x="importancia", y="feature",
        orientation="h",
        labels={"importancia": "Importancia", "feature": "Variable"},
        title="Importancia de variables (Random Forest)",
    )
    fig.update_layout(margin=dict(l=140, r=20, t=40, b=40))
    return fig


# ---------------------------------------------------------------------------
# 5. Layout
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Emisiones de CO2 - LogiMarket / Sustainable Energy"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("Estimación de emisiones de CO2 por país", style={"textAlign": "center"}),
        html.P(
            f"Modelo Random Forest - R2 = {r2:.4f} | MAE = {mae:,.0f} kt | RMSE = {rmse:,.0f} kt",
            style={"textAlign": "center", "color": "#555"}
        ),

        html.Hr(),

        # --- 1.1 Choropleth ---
        html.H3("1. Emisiones de CO2 por país"),
        dcc.Graph(id="grafico-choropleth"),
        html.Label("Año:"),
        dcc.Slider(
            id="slider-anio",
            min=anio_min, max=anio_max, step=1, value=anio_max,
            marks={a: str(a) for a in range(anio_min, anio_max + 1, 2)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),

        html.Hr(),

        # --- 1.2 Series temporales ---
        html.H3("2. Evolución de emisiones por país"),
        html.Label("Selecciona países a comparar:"),
        dcc.Dropdown(
            id="dropdown-paises",
            options=[{"label": p, "value": p} for p in paises_disponibles],
            value=top_emisores_default,
            multi=True,
        ),
        dcc.Graph(id="grafico-series-temporales"),

        html.Hr(),

        # --- 1.3 Predicho vs real ---
        html.H3("3. Predicho vs real (conjunto de test)"),
        dcc.Graph(id="grafico-predicho-real", figure=crear_predicho_vs_real()),

        html.Hr(),

        # --- 1.4 Importancia de variables ---
        html.H3("4. Importancia de variables del modelo"),
        dcc.Graph(id="grafico-importancia", figure=crear_importancia()),
    ],
)

# ---------------------------------------------------------------------------
# 6. Callbacks
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
# 7. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
