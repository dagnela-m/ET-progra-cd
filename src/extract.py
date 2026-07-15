"""
Funciones de extracción: CSV principal, mapeo de países a ISO3,
y extracción de indicadores desde la API del World Bank.

Empaqueta el código de las secciones 2.2, 2.3 y 2.4 del cuaderno.
"""

import os
import time

import pandas as pd
import pycountry
import requests
from tqdm.auto import tqdm


def extract_csv(path):
    """Carga el CSV principal desde data/raw/."""
    df = pd.read_csv(path)
    return df


def country_to_iso3(name):
    """Convierte nombre de país a código ISO3. Devuelve None si no hay match."""
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def aplicar_iso3(df, columna_pais="Entity"):
    """Agrega la columna iso3 al DataFrame, con corrección manual para
    los países que pycountry no reconoce (ej. Turkey -> Türkiye).

    Nota: esta función no existía como tal en el cuaderno (sección 2.3
    era código suelto sobre el df global); se envuelve acá para que
    pipeline.py pueda invocarla.
    """
    mapeo_manual = {"Turkey": "TUR"}

    df["iso3"] = df[columna_pais].apply(country_to_iso3)
    df["iso3"] = df["iso3"].fillna(df[columna_pais].map(mapeo_manual))

    return df


def extract_worldbank(indicator, countries_iso3, start=2000, end=2020,
                       output_path=None, max_retries=3, timeout=15):
    """Extrae un indicador desde la API del World Bank para una lista
    de países. Reanuda automáticamente si ya existe un archivo parcial
    en output_path (cache en data/raw/)."""

    if output_path is None:
        output_path = f"data/raw/{indicator}.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        downloaded = set(df_existing["iso3"].unique())
        print(f"Se encontraron {len(downloaded)} países ya descargados.")
    else:
        downloaded = set()

    pending = [iso3 for iso3 in countries_iso3 if iso3 not in downloaded]
    print(f"Pendientes: {len(pending)} de {len(countries_iso3)} países.")

    for iso3 in tqdm(pending, desc=indicator, unit="país"):
        url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}"
        params = {"date": f"{start}:{end}", "format": "json", "per_page": 100}

        data = None
        for intento in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                break
            except (requests.exceptions.RequestException,
                    requests.exceptions.JSONDecodeError):
                espera = 2 ** (intento + 1)
                print(f"[{iso3}] intento {intento+1}/{max_retries}. "
                      f"Reintentando en {espera}s...")
                time.sleep(espera)

        if data is None:
            print(f"Se omitió {iso3}")
            continue

        if len(data) > 1 and data[1]:
            rows = [{"iso3": iso3, "year": int(entry["date"]), "value": entry["value"]}
                    for entry in data[1]]
            df_country = pd.DataFrame(rows)
            df_country.to_csv(output_path, mode="a",
                               header=not os.path.exists(output_path), index=False)
        time.sleep(0.5)

    return pd.read_csv(output_path)
