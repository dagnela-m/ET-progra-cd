"""
Función de carga: guarda el DataFrame procesado como CSV limpio.
"""

import os


def load_data(df, path="data/processed/sustainable_energy_clean.csv"):
    """Guarda el DataFrame procesado como CSV limpio."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("Dataset limpio guardado:", df.shape)
