import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb # El modelo
from sklearn.model_selection import train_test_split # Para dividir los datos
from sklearn.metrics import accuracy_score, classification_report # Para medir


# Importamos nuestro propio procesador
try:
    from modelo.procesador_features import crear_features_y_target
except ImportError:
    print("Error: No se pudo importar 'procesador_features.py'.")
    sys.exit(1)

def entrenar_nuevo_modelo(dias_a_predecir=5):
    """
    Función principal de este módulo.
    1. Carga y procesa los features (usando procesador_features).
    2. Divide los datos en entrenamiento y prueba.
    3. Entrena un modelo XGBoost.
    4. Evalúa el modelo y muestra los resultados.
    5. Devuelve el modelo entrenado.
    
    Args:
        dias_a_predecir (int): El horizonte (1, 5, 21) para el que se entrenará.
    """
    print(f"[Gestor Modelo] Iniciando entrenamiento para {dias_a_predecir} días...")

    # --- 1. Obtener Datos ---
    # ¡CAMBIO CLAVE! Pasamos el parámetro a la función de creación
    X, y = crear_features_y_target(dias_a_predecir=dias_a_predecir)
    
    if X is None or y is None:
        print("[Gestor Modelo] Falló la obtención de datos. Abortando.")
        return None
    
    if X.empty or y.empty:
        print(f"[Gestor Modelo] No hay datos suficientes para entrenar (Filas: {len(X)}).")
        return None

    # --- 2. Dividir Datos (Train/Test Split) ---
    # NUNCA barajamos (shuffle=False) en series de tiempo.
    # Usamos el 20% más reciente de los datos para probar.
    
    print("[Gestor Modelo] Dividiendo datos en 80% entrenamiento y 20% prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        shuffle=False 
    )
    
    if X_train.empty or y_train.empty:
        print("[Gestor Modelo] Error: No hay suficientes datos después de la división. Prueba un período más largo.")
        return None

    # --- 3. Definir y Entrenar el Modelo (XGBoost) ---
    print("[Gestor Modelo] Definiendo modelo XGBClassifier...")
    modelo = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("[Gestor Modelo] Entrenando el modelo...")
    modelo.fit(X_train, y_train)
    print("[Gestor Modelo] ¡Entrenamiento completado!")

    # --- 4. Evaluar el Modelo ---
    print(f"\n--- Evaluación (para {dias_a_predecir} días) sobre datos de prueba ---")
    
    y_pred = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión (Accuracy): {accuracy * 100:.2f}%")
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['No Sube (0)', 'Sube (1)'], zero_division=0))
    
    # --- 5. Devolver el modelo ---
    return modelo
