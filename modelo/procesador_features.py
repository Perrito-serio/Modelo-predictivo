import sys
import os
import pandas as pd
import numpy as np

# --- Configuración de Ruta (Path) ---
# Esto es OBLIGATORIO para que 'modelo/' pueda encontrar 'datos/'
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- Fin de Configuración ---

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
# ¡CAMBIO! Eliminamos DIAS_A_PREDECIR = 5 de aquí.
PERIODO_CORRELACION = 30  # Ventana de 30 días para la correlación móvil

# ¡CAMBIO! Añadimos el argumento (con valor por defecto 5)
def crear_features_y_target(dias_a_predecir=5):
    """
    Función principal que carga, combina y procesa todos los datos
    para crear la tabla de entrenamiento (X) y el objetivo (y).
    """
    print("[Procesador] Iniciando la creación de features...")

    # --- 1. Cargar Datos Crudos ---
    print(f"[Procesador] Cargando datos de {TARGET_PREDICCION}...")
    df_aapl_raw = gestor_aapl.cargar_datos_modelo()
    
    print(f"[Procesador] Cargando datos de {FACTOR_PREDICTOR}...")
    df_smh_raw = gestor_smh.cargar_datos_modelo()

    if df_aapl_raw is None or df_smh_raw is None:
        print("Error fatal: No se pudieron cargar los datos de uno de los gestores.")
        return None, None

    # --- 2. Preparar y Combinar ---
    # Solo necesitamos 'Close' (o Adj Close) para los retornos
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
    # Convertimos los precios en "Retornos" (% de cambio)
    # Esto es mucho más útil para el modelo que el precio crudo
    print("[Procesador] Creando features (Retornos, Correlación, SMA)...")
    
    # a) Retornos (Lags) de AAPL y SMH
    for lag in [1, 3, 5, 10]:
        df[f'AAPL_retorno_{lag}d'] = df['AAPL'].pct_change(periods=lag)
        df[f'SMH_retorno_{lag}d'] = df['SMH'].pct_change(periods=lag)

    # b) Correlación Móvil (¡La pista clave!)
    # Calculamos la correlación de los retornos de 1 día
    retornos_1d = df['AAPL'].pct_change(1).dropna()
    retornos_smh_1d = df['SMH'].pct_change(1).dropna()
    df[f'Correl_AAPL_SMH_{PERIODO_CORRELACION}d'] = retornos_1d.rolling(PERIODO_CORRELACION).corr(retornos_smh_1d)

    # c) Indicador de Tendencia (Media Móvil)
    sma_20 = df['AAPL'].rolling(window=20).mean()
    df['AAPL_vs_SMA20'] = (df['AAPL'] - sma_20) / sma_20 # % de distancia de la media

    
    # --- 4. Creación del Objetivo (Target - y) ---
    # Queremos predecir: ¿El precio de AAPL será más alto 
    # en 'DIAS_A_PREDECIR' días de lo que es hoy?
    
    # ¡CAMBIO! Usamos el argumento de la función 'dias_a_predecir'
    print(f"[Procesador] Creando target (y) a {dias_a_predecir} días...")

    # Usamos .shift(-N) para "traer" el precio futuro N días a la fila de hoy
    # ¡CAMBIO! Usamos el argumento de la función 'dias_a_predecir'
    precio_futuro = df['AAPL'].shift(-dias_a_predecir)
    
    # La condición: 1 si el precio futuro es > al precio de hoy, 0 si no
    df['Target'] = (precio_futuro > df['AAPL']).astype(int)

    
    # --- 5. Limpieza Final ---
    # Todo este 'rolling' y 'shift' crea valores NaN (vacíos)
    # al principio y al final del DataFrame.
    # Debemos eliminarlos para que el modelo solo entrene con datos completos.
    print("[Procesador] Limpiando datos (eliminando NaNs)...")
    df = df.dropna()
    
    # Separamos X (las "pistas") de y (la "respuesta")
    
    # y (Target): La columna que queremos predecir
    y = df['Target']
    
    # X (Features): TODO lo demás (excepto los precios crudos y el target)
    columnas_features = [col for col in df.columns if col not in ['AAPL', 'SMH', 'Target']]
    X = df[columnas_features]

    print(f"[Procesador] ¡Listo! Features (X) y Target (y) creados.")
    print(f"Total de muestras de entrenamiento: {len(X)}")
    
    return X, y