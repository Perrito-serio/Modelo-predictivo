import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuración de Ruta (Path) ---
# OBLIGATORIO para que 'graficos/' pueda encontrar 'modelo/' y 'datos/'
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- Fin de Configuración ---

try:
    from datos import gestor_aapl
    from modelo.procesador_features import crear_features_y_target
    from modelo.gestor_modelo import entrenar_nuevo_modelo
except ImportError as e:
    print("Error: No se pudieron importar los módulos de 'modelo/' o 'datos/'.")
    print(f"Detalle: {e}")
    sys.exit(1)

def graficar_predicciones_vs_realidad(modelo, X_test, y_test, y_pred):
    """
    Grafica el precio real de AAPL y superpone los aciertos y 
    errores de las predicciones del modelo.
    """
    print("[Graficador] Creando gráfico de Predicción vs. Realidad...")
    
    # --- 1. Obtener el DataFrame de precios original ---
    # Necesitamos los precios crudos para graficarlos en el eje Y
    try:
        df_aapl_raw = gestor_aapl.cargar_datos_modelo()
        if df_aapl_raw is None:
            raise Exception("No se cargaron datos de gestor_aapl")
            
        # Determinar la columna de precio (igual que en procesador_features)
        col_aapl = 'Adj Close' if 'Adj Close' in df_aapl_raw.columns else 'Close'
        
        # Alineamos los precios con el índice de nuestros datos de prueba (X_test)
        precios_test = df_aapl_raw.loc[X_test.index][col_aapl]
        
    except Exception as e:
        print(f"Error fatal al cargar los precios de AAPL: {e}")
        return

    # --- 2. Preparar los puntos para graficar ---
    # Convertimos la predicción (array) a una Serie de Pandas
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    
    # Aciertos
    acierto_sube = precios_test[(y_pred_series == 1) & (y_test == 1)]
    acierto_no_sube = precios_test[(y_pred_series == 0) & (y_test == 0)]
    
    # Errores
    error_predijo_sube = precios_test[(y_pred_series == 1) & (y_test == 0)]
    error_predijo_no_sube = precios_test[(y_pred_series == 0) & (y_test == 1)]

    # --- 3. Dibujar el Gráfico ---
    try:
        plt.figure(figsize=(15, 8))
        
        # Línea de precio base
        plt.plot(precios_test.index, precios_test, 
                 color='gray', 
                 alpha=0.7, 
                 label='Precio Real AAPL (Datos de Prueba)')
        
        # Aciertos
        plt.plot(acierto_sube.index, acierto_sube, 
                 '^', markersize=10, color='green', 
                 label='Acierto: Sube (Verdadero Positivo)')
        
        plt.plot(acierto_no_sube.index, acierto_no_sube, 
                 'v', markersize=10, color='blue', 
                 label='Acierto: No Sube (Verdadero Negativo)')
        
        # Errores
        plt.plot(error_predijo_sube.index, error_predijo_sube, 
                 '^', markersize=10, color='yellow', 
                 label='Error: Predijo Sube (Falso Positivo)')
        
        plt.plot(error_predijo_no_sube.index, error_predijo_no_sube, 
                 'v', markersize=10, color='red', 
                 label='Error: Predijo No Sube (Falso Negativo)')

        # Títulos y Leyenda
        plt.title('Visualización de Predicciones vs. Realidad (Datos de Prueba)')
        plt.ylabel('Precio (USD)')
        plt.xlabel('Fecha')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error al dibujar el gráfico: {e}")
