# Proyecto Final — Programación para Ciencia de Datos

Estimación de emisiones de CO₂ por país usando un pipeline ETL y modelo Random Forest.

## Estructura del proyecto

proyecto_final/
├── data/
│   ├── raw/          # dataset original y caches de API (no se suben al repo)
│   └── processed/    # CSV limpio generado por el pipeline
├── notebooks/
│   └── ET_Programacion_Ciencia_Datos.ipynb
├── src/
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   └── pipeline.py
├── dashboard/
│   └── app.py
├── docs/
├── Dockerfile
├── requirements.txt
└── README.md

## Requisitos

Instalar dependencias:

pip install -r requirements.txt

## Cómo correr el pipeline

1. Descargar `global-data-on-sustainable-energy.csv` desde Kaggle (https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy) y pegarlo en `data/raw/`

2. Ejecutar desde la raíz del proyecto:

python src/pipeline.py

La primera corrida descarga datos desde la API del Banco Mundial (tarda varios minutos). Las siguientes usan el cache en `data/raw/`.

## Cómo levantar el dashboard

### Local

python dashboard/app.py

Abrir en el navegador: `http://localhost:8050`

### Con Docker

docker build -t proyecto-ds .
docker run -p 8050:8050 proyecto-ds