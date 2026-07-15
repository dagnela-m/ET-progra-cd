"""
Orquesta las fases de extracción, transformación y carga.

Debe ejecutarse desde la raíz del proyecto (mismo supuesto que el
cuaderno): las rutas por defecto son relativas a data/raw y
data/processed.
"""

from src.extract import extract_csv, aplicar_iso3, extract_worldbank
from src.transform import normalizar_nombres_tipos, transform
from src.load import load_data


def run_pipeline(
    csv_path="data/raw/global-data-on-sustainable-energy.csv",
    output_path="data/processed/sustainable_energy_clean.csv",
):
    """Corre el pipeline completo: extracción, transformación y carga."""

    # Extracción (2.2, 2.3)
    df = extract_csv(csv_path)
    df = aplicar_iso3(df)

    # Normalización (2.5) -- antes de pedir los indicadores de la API,
    # tal como en el cuaderno.
    df = normalizar_nombres_tipos(df)

    # Extracción desde la API (2.4)
    paises_iso3 = df["iso3"].unique().tolist()
    df_population = extract_worldbank("SP.POP.TOTL", paises_iso3)
    df_urbanization = extract_worldbank("SP.URB.TOTL.IN.ZS", paises_iso3)

    # Transformación (fase 4)
    df = transform(df, df_population, df_urbanization)

    # Carga (fase 5)
    load_data(df, output_path)

    return df


if __name__ == "__main__":
    run_pipeline()
