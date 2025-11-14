import sys
import os
import pandas as pd
import numpy as np


# Ahora podemos importar nuestros gestores de datos
try:
    from datos import gestor_aapl
    from datos import gestor_smh
except ImportError:
    print("Error: No se pudieron importar los gestores desde 'datos/'.")
    print("Asegúrate de que la estructura de carpetas y los __init__.py son correctos.")
    sys.exit(1)

# --- CONFIGURACIÓN DEL MODELO ---
TARGET_PREDICCION = "AAPL"
FACTOR_PREDICTOR = "SMH"
# La variable DIAS_A_PREDECIR se ha eliminado, ahora es un parámetro
PERIODO_CORRELACION = 30

def crear_features_y_target(dias_a_predecir=5):
    """
    Función principal que carga, combina y procesa todos los datos
    para crear la tabla de entrenamiento (X) y el objetivo (y).
    
    Args:
        dias_a_predecir (int): El horizonte de tiempo para el target (ej. 1, 5, 21).
    """
    print(f"[Procesador] Iniciando la creación de features para {dias_a_predecir} días...")

    # --- 1. Cargar Datos Crudos ---
    print(f"[Procesador] Cargando datos de {TARGET_PREDICCION}...")
    df_aapl_raw = gestor_aapl.cargar_datos_modelo()
    
    print(f"[Procesador] Cargando datos de {FACTOR_PREDICTOR}...")
    df_smh_raw = gestor_smh.cargar_datos_modelo()

    if df_aapl_raw is None or df_smh_raw is None:
        print("Error fatal: No se pudieron cargar los datos de uno de los gestores.")
        return None, None

    # --- 2. Preparar y Combinar ---
    # Usaremos 'Adj Close' (Cierre Ajustado) si está, si no 'Close'
    col_aapl = 'Adj Close' if 'Adj Close' in df_aapl_raw.columns else 'Close'
    col_smh = 'Adj Close' if 'Adj Close' in df_smh_raw.columns else 'Close'

    # Renombramos y creamos un DataFrame único
    df = pd.DataFrame(index=df_aapl_raw.index)
    df['AAPL'] = df_aapl_raw[col_aapl]
    df['SMH'] = df_smh_raw[col_smh]
    
    # Eliminamos cualquier día en que uno de los dos no haya cotizado
    df = df.dropna()

    # --- 3. Ingeniería de Atributos (Features - X) ---
    print("[Procesador] Creando features (Retornos, Correlación, SMA)...")
    
    # a) Retornos (Lags) de AAPL y SMH
    for lag in [1, 3, 5, 10]:
        df[f'AAPL_retorno_{lag}d'] = df['AAPL'].pct_change(periods=lag)
        df[f'SMH_retorno_{lag}d'] = df['SMH'].pct_change(periods=lag)

    # b) Correlación Móvil (¡La pista clave!)
    retornos_1d = df['AAPL'].pct_change(1).dropna()
    retornos_smh_1d = df['SMH'].pct_change(1).dropna()
    df[f'Correl_AAPL_SMH_{PERIODO_CORRELACION}d'] = retornos_1d.rolling(PERIODO_CORRELACION).corr(retornos_smh_1d)

    # c) Indicador de Tendencia (Media Móvil)
    sma_20 = df['AAPL'].rolling(window=20).mean()
    df['AAPL_vs_SMA20'] = (df['AAPL'] - sma_20) / sma_20 # % de distancia de la media

    
    # --- 4. Creación del Objetivo (Target - y) ---
    print(f"[Procesador] Creando target (y) a {dias_a_predecir} días...")

    # Usamos .shift(-N) para "traer" el precio futuro N días a la fila de hoy
    precio_futuro = df['AAPL'].shift(-dias_a_predecir)
    
    # La condición: 1 si el precio futuro es > al precio de hoy, 0 si no
    df['Target'] = (precio_futuro > df['AAPL']).astype(int)

    
    # --- 5. Limpieza Final ---
    print("[Procesador] Limpiando datos (eliminando NaNs)...")
    df = df.dropna()
    
    # Convertimos los tipos de 'y' a entero
    y = df['Target'].astype(int)
    
    # X (Features): TODO lo demás (excepto los precios crudos y el target)
    columnas_features = [col for col in df.columns if col not in ['AAPL', 'SMH', 'Target']]
    X = df[columnas_features]

    print(f"[Procesador] ¡Listo! Features (X) y Target (y) creados.")
    print(f"- Total de muestras de entrenamiento: {len(X)}")
    
    return X, y
