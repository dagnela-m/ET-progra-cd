"""
Función de transformación: normalización de columnas (2.5) y toda
la fase 4 del cuaderno (eliminación de columnas y filas, merge con
la API, imputación, feature engineering).
"""

import pandas as pd

RENAME_MAP = {
    "Entity": "Pais",
    "Year": "Anio",
    "Access to electricity (% of population)": "Acceso_electricidad",
    "Access to clean fuels for cooking": "Acceso_combustible_limpio",
    "Renewable-electricity-generating-capacity-per-capita": "Capacidad_renovable_pc",
    "Financial flows to developing countries (US $)": "Flujos_financieros",
    "Renewable energy share in the total final energy consumption (%)": "Renovable_pct",
    "Renewables (% equivalent primary energy)": "Renovable_energia_primaria_pct",
    "Electricity from fossil fuels (TWh)": "Elec_fossil",
    "Electricity from nuclear (TWh)": "Elec_nuclear",
    "Electricity from renewables (TWh)": "Elec_renovables",
    "Low-carbon electricity (% electricity)": "Bajo_carbon_pct",
    "Primary energy consumption per capita (kWh/person)": "Consumo_ener_pc",
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": "Intensidad_energia",
    "Value_co2_emissions_kt_by_country": "Emision_CO2",
    "Land Area(Km2)": "Area_Kkm2",
    "Latitude": "Latitud",
    "Longitude": "Longitud",
    "gdp_growth": "Gdp_crecimiento",
    "gdp_per_capita": "Gdp_pc",
}


def normalizar_nombres_tipos(df):
    """Sección 2.5: renombra columnas y castea Densidad a numérico."""
    df = df.rename(columns=RENAME_MAP)

    col_densidad = [c for c in df.columns if c.startswith("Density")][0]
    df["Densidad"] = pd.to_numeric(
        df[col_densidad].str.replace(",", ""),
        errors="coerce"
    )
    df = df.drop(columns=[col_densidad])
    return df


def transform(df, df_population, df_urbanization):
    """Aplica la fase de transformación (sección 4) sobre el DataFrame
    ya normalizado (normalizar_nombres_tipos) y los dos DataFrames
    crudos que devuelve extract_worldbank (columnas iso3/year/value).

    Nota: el rename de 'value' -> Poblacion/Urbanizacion vivía en la
    celda 25 del cuaderno (junto a la llamada a extract_worldbank) y
    el rename de 'year' -> Anio vivía en la celda 58 (justo antes del
    merge). Se consolidan ambos acá porque son preparación del mismo
    merge.
    """

    # 4.1 Eliminación de columnas con exceso de nulos
    df = df.drop(columns=[
        "Flujos_financieros",
        "Renovable_energia_primaria_pct"
    ], errors="ignore")

    # 4.2 Eliminación de filas sin target
    df = df.dropna(subset=["Emision_CO2"])

    # 4.3 Merge con los datos de la API
    df_population = df_population.rename(columns={"year": "Anio", "value": "Poblacion"})
    df_urbanization = df_urbanization.rename(columns={"year": "Anio", "value": "Urbanizacion"})
    df = df.merge(df_population, on=["iso3", "Anio"], how="left")
    df = df.merge(df_urbanization, on=["iso3", "Anio"], how="left")

    # 4.4 Imputación de nulos restantes (mediana por país, luego mediana global)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df.groupby("Pais")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # 4.5 Tratamiento de outliers: se conservan (decisión documentada en el
    # cuaderno, sección 4.5). No requiere transformación de código.

    # 4.6 Feature engineering: ratio de electricidad renovable
    total_generacion = df["Elec_fossil"] + df["Elec_nuclear"] + df["Elec_renovables"]
    df["renovable_ratio"] = df["Elec_renovables"] / total_generacion
    df["renovable_ratio"] = df["renovable_ratio"].fillna(0)

    return df
