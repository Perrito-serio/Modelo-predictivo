import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # <--- ¡ESTA ES LA LÍNEA QUE FALTABA!
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# --- Configuración de Ruta (Path) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- Fin de Configuración ---

try:
    from modelo.procesador_features import crear_features_y_target
    from modelo.gestor_modelo import entrenar_nuevo_modelo
except ImportError as e:
    print("Error: No se pudieron importar los módulos de 'modelo/'.")
    print(f"Detalle: {e}")
    sys.exit(1)

def graficar_importancia_features(modelo):
    """
    Usa la función nativa de XGBoost para graficar la importancia (por Ganancia).
    """
    print("[Graficador] Creando gráfico de Importancia de Features (por Ganancia)...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(
            modelo, 
            ax=ax, 
            importance_type='gain', # Usamos 'gain' (impacto)
            title='Importancia de Features (por Ganancia)'
        )
        plt.tight_layout()
        plt.show() 
    except Exception as e:
        print(f"Error al graficar importancia de features: {e}")

def graficar_matriz_confusion(y_true, y_pred):
    """
    Crea un mapa de calor (heatmap) con la Matriz de Confusión.
    """
    print("[Graficador] Creando Matriz de Confusión...")
    try:
        cm = confusion_matrix(y_true, y_pred)
        labels = ['Predijo No Sube', 'Predijo Sube']
        categories = ['Real No Sube', 'Real Sube']
        
        sum_rows = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = cm.astype('float') / sum_rows
            cm_percent = np.nan_to_num(cm_percent) # Convertir nan a 0

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_percent, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues', 
            xticklabels=labels, 
            yticklabels=categories
        )
        
        plt.title('Matriz de Confusión (Datos de Prueba)')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predicho')
        plt.tight_layout()
        plt.show() 
        
    except Exception as e:
        print(f"Error al graficar la matriz de confusión: {e}")

# --- BLOQUE PRINCIPAL (Para ejecutar) ---
if __name__ == "__main__":
    print("--- INICIANDO PRUEBA RÁPIDA DE plot_modelo.py ---")
    
    print("[Paso 1] Entrenando modelo para obtener resultados...")
    modelo = entrenar_nuevo_modelo()
    
    if modelo is None:
        print("El entrenamiento falló. No se pueden generar gráficos.")
    else:
        print("\n[Paso 2] Modelo entrenado. Generando gráficos...")
        
        # Graficar Importancia
        graficar_importancia_features(modelo)
        
        print("\n[Paso 3] Recalculando datos de prueba para Matriz de Confusión...")
        X, y = crear_features_y_target()
        
        if X is not None and not X.empty:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            y_pred = modelo.predict(X_test)
            graficar_matriz_confusion(y_test, y_pred) # Ahora esto funcionará
        else:
            print("No se pudieron cargar datos para la matriz de confusión.")
        
        print("\n--- Gráficos del modelo finalizados ---")