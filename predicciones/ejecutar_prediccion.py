import sys
import os
import pandas as pd
from datetime import datetime

# --- Configuración de Ruta (Path) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- Fin de Configuración ---

try:
    from modelo.gestor_modelo import entrenar_nuevo_modelo
    from modelo.procesador_features import crear_features_y_target
    from datos import gestor_aapl # Para obtener el último precio
except ImportError as e:
    print("--- ERROR FATAL AL IMPORTAR ---")
    print(f"Error: {e}")
    input("Presiona Enter para salir.")
    sys.exit(1)

# --- Configuración del Script ---
HORIZONTES_DE_PREDICCION = [1, 5, 21] # Diario, Semanal, Mensual
LOG_FILE_PATH = os.path.join(script_dir, 'registro_predicciones.csv')


def ejecutar_predicciones():
    """
    Script principal que entrena, predice y guarda un registro.
    Esta función es llamada por main.py O por este script.
    """
    limpiar_pantalla()
    print("==============================================")
    print("    INICIANDO SCRIPT DE PREDICCIÓN DIARIA    ")
    print("==============================================")
    
    modelos_entrenados = {}
    
    # --- 1. Entrenar los Modelos ---
    print("\n--- Fase 1: Entrenando Modelos ---")
    for dias in HORIZONTES_DE_PREDICCION:
        print(f"\nEntrenando modelo de {dias} día(s)...")
        modelo = entrenar_nuevo_modelo(dias_a_predecir=dias)
        if modelo is None:
            print(f"¡Error fatal! No se pudo entrenar el modelo de {dias} días.")
            return
        modelos_entrenados[dias] = modelo
        print(f"Modelo de {dias} días listo.")

    # --- 2. Obtener Datos para Predecir ---
    print("\n--- Fase 2: Obteniendo Datos Frescos para Predecir ---")
    
    X_full, _ = crear_features_y_target(dias_a_predecir=1)
    
    if X_full is None or X_full.empty:
        print("¡Error fatal! No se pudieron generar las features.")
        return
        
    ultimo_dato_features = X_full.tail(1)
    df_aapl_raw = gestor_aapl.cargar_datos_modelo()
    col_aapl = 'Adj Close' if 'Adj Close' in df_aapl_raw.columns else 'Close'
    ultimo_cierre = df_aapl_raw[col_aapl].iloc[-1]
    fecha_ultimo_cierre = df_aapl_raw.index[-1].strftime('%Y-%m-%d')
    
    print(f"Datos del último cierre ({fecha_ultimo_cierre}):")
    print(f"Precio: ${ultimo_cierre:.2f}")
    print("\nFeatures de entrada para el modelo:")
    print(ultimo_dato_features.to_string())

    # --- 3. Realizar Predicciones ---
    print("\n--- Fase 3: Generando Predicciones ---")
    predicciones = {}
    for dias in HORIZONTES_DE_PREDICCION:
        modelo = modelos_entrenados[dias]
        pred_binaria = modelo.predict(ultimo_dato_features)[0]
        probabilidad = modelo.predict_proba(ultimo_dato_features)[0]
        confianza = probabilidad[pred_binaria] * 100
        predicciones[dias] = {
            "prediccion": int(pred_binaria),
            "confianza_pct": f"{confianza:.2f}%"
        }

    # --- 4. Mostrar Reporte y Guardar Log ---
    print("\n--- Fase 4: Reporte y Registro ---")
    print(f"\nPredicciones generadas el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Basado en el cierre de {fecha_ultimo_cierre} (${ultimo_cierre:.2f})")
    
    reporte_log = {
        "fecha_prediccion": datetime.now().strftime('%Y-%m-%d'),
        "fecha_ultimo_cierre": fecha_ultimo_cierre,
        "precio_ultimo_cierre": round(ultimo_cierre, 2)
    }
    
    for dias, pred_info in predicciones.items():
        resultado = "SUBE" if pred_info['prediccion'] == 1 else "NO SUBE"
        confianza = pred_info['confianza_pct']
        print(f"\n - Predicción a {dias} día(s):")
        print(f"   Resultado: {resultado} ({pred_info['prediccion']})")
        print(f"   Confianza: {confianza}")
        
        reporte_log[f'pred_{dias}d'] = pred_info['prediccion']
        reporte_log[f'conf_{dias}d'] = confianza

    try:
        df_log = pd.DataFrame([reporte_log])
        if not os.path.exists(LOG_FILE_PATH):
            df_log.to_csv(LOG_FILE_PATH, index=False)
            print(f"\nSe creó el registro en: {LOG_FILE_PATH}")
        else:
            df_log.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)
            print(f"\nPredicción añadida al registro: {LOG_FILE_PATH}")
            
    except Exception as e:
        print(f"\nError al guardar el registro CSV: {e}")

    print("\n==============================================")
    print("    SCRIPT DE PREDICCIÓN COMPLETADO    ")
    print("==============================================")

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Punto de Entrada ---
if __name__ == "__main__":
    ejecutar_predicciones()
    # ¡CAMBIO CLAVE! El input() solo se ejecuta si corremos el script directo
    input("\nPresiona Enter para salir.")