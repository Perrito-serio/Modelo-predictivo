import sys
import os
import time

# --- Configuración de Ruta (Path) ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Imports de Módulos del Proyecto ---
try:
    from datos import gestor_aapl, gestor_smh
    from graficos.plot_velas import graficar_velas_ventana
    from modelo.gestor_modelo import entrenar_nuevo_modelo
    from modelo.procesador_features import crear_features_y_target
    from graficos.plot_modelo import graficar_importancia_features, graficar_matriz_confusion
    from graficos.plot_prediccion import graficar_predicciones_vs_realidad
    from sklearn.model_selection import train_test_split
    
    # --- ¡NUEVA LÍNEA! ---
    from predicciones.ejecutar_prediccion import ejecutar_predicciones

except ImportError as e:
    print("--- ERROR FATAL AL IMPORTAR ---")
    print(f"Error: {e}")
    input("Presiona Enter para salir.")
    sys.exit(1)


# --- Variables Globales para el Menú ---
modelo_entrenado_5d = None # Lo haremos específico para 5 días
X_test_cache = None
y_test_cache = None
y_pred_cache = None

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

def pausar():
    input("\nPresiona Enter para continuar...")

# --- Sub-Menús de Gráficos (Sin cambios) ---

def menu_graficos_velas():
    # ... (Sin cambios)...
    limpiar_pantalla()
    print("--- Menú de Gráficos: Velas ---")
    print("1. Graficar Velas de AAPL (Últimos 6 meses)")
    print("2. Graficar Velas de SMH (Últimos 6 meses)")
    print("0. Volver al menú de gráficos")
    opcion = input("\nSelecciona una opción: ")
    if opcion == '1':
        try:
            print("Cargando datos de AAPL...")
            datos_aapl = gestor_aapl.cargar_datos_modelo()
            graficar_velas_ventana(datos_aapl, "AAPL")
            print("Gráfico de AAPL cerrado.")
        except Exception as e:
            print(f"Error al graficar AAPL: {e}")
        pausar()
    elif opcion == '2':
        try:
            print("Cargando datos de SMH...")
            datos_smh = gestor_smh.cargar_datos_modelo()
            graficar_velas_ventana(datos_smh, "SMH")
            print("Gráfico de SMH cerrado.")
        except Exception as e:
            print(f"Error al graficar SMH: {e}")
        pausar()
    elif opcion == '0':
        return
    else:
        print("Opción no válida.")
        pausar()


def menu_graficos_modelo():
    global modelo_entrenado_5d, X_test_cache, y_test_cache, y_pred_cache
    
    if modelo_entrenado_5d is None:
        print("\n¡Error! Primero debes entrenar un modelo (Opción 2).")
        pausar()
        return

    if X_test_cache is None:
        print("\nPreparando datos de prueba (X_test, y_test) para los gráficos (para 5 días)...")
        # ¡CAMBIO! Especificamos 5 días
        X, y = crear_features_y_target(dias_a_predecir=5)
        if X is None or X.empty:
            print("Error al crear features.")
            pausar()
            return
        _, X_test_cache, _, y_test_cache = train_test_split(X, y, test_size=0.2, shuffle=False)
        y_pred_cache = modelo_entrenado_5d.predict(X_test_cache)
        print("¡Datos de prueba listos!")

    limpiar_pantalla()
    print("--- Menú de Gráficos: Modelo (Evaluación 5 Días) ---")
    print("1. Gráfico de Importancia de Features")
    print("2. Matriz de Confusión")
    print("3. Gráfico de Predicción vs. Realidad")
    print("0. Volver al menú de gráficos")
    opcion = input("\nSelecciona una opción: ")
    if opcion == '1':
        graficar_importancia_features(modelo_entrenado_5d)
        pausar()
    elif opcion == '2':
        graficar_matriz_confusion(y_test_cache, y_pred_cache)
        pausar()
    elif opcion == '3':
        graficar_predicciones_vs_realidad(modelo_entrenado_5d, X_test_cache, y_test_cache, y_pred_cache)
        pausar()
    elif opcion == '0':
        return
    else:
        print("Opción no válida.")
        pausar()

def menu_graficos_principal():
    while True:
        limpiar_pantalla()
        print("--- Menú Principal de Gráficos ---")
        print("1. Gráficos de Velas (Datos Crudos)")
        print("2. Gráficos de Evaluación (Modelo de 5 Días)")
        print("\n0. Volver al Menú Principal")
        opcion = input("\nSelecciona una opción: ")
        if opcion == '1':
            menu_graficos_velas()
        elif opcion == '2':
            menu_graficos_modelo()
        elif opcion == '0':
            break
        else:
            print("Opción no válida.")
            pausar()

# --- Funciones del Menú Principal (Actualizadas) ---

def ejecutar_entrenamiento_menu():
    global modelo_entrenado_5d, X_test_cache, y_test_cache, y_pred_cache
    
    limpiar_pantalla()
    print("--- Entrenando Modelo de 5 Días (para Gráficos) ---")
    print("Esto es solo para evaluar el modelo de 5 días...")
    try:
        # ¡CAMBIO! Entrenamos solo el de 5 días para esta opción
        modelo_entrenado_5d = entrenar_nuevo_modelo(dias_a_predecir=5)
        
        # Reseteamos el caché
        X_test_cache, y_test_cache, y_pred_cache = None, None, None
        
        if modelo_entrenado_5d is not None:
            print("\n¡Modelo de 5 días entrenado y evaluado!")
            print("Ya puedes ver los 'Gráficos de Evaluación del Modelo'.")
        else:
            print("\nError: El entrenamiento falló.")
    
    except Exception as e:
        print(f"\nSe produjo un error inesperado: {e}")
    
    pausar()

def ejecutar_prediccion_en_vivo_menu():
    """
    Esta función es solo un "envoltorio" para llamar a la
    función que importamos de 'predicciones/'.
    """
    global modelo_entrenado_5d # Reseteamos el modelo de evaluación
    
    limpiar_pantalla()
    print("--- Ejecutando Script de Predicción Completo ---")
    print("Esto entrenará los 3 modelos (1d, 5d, 21d) y guardará un registro.")
    print("Puede tardar varios segundos...")
    pausar()
    
    try:
        # --- ¡AQUÍ ESTÁ LA LLAMADA! ---
        ejecutar_predicciones()
        
        # Como esto ya entrenó los modelos, reseteamos el de evaluación
        # para forzar que los gráficos carguen los datos correctos si se usan después.
        modelo_entrenado_5d = None
        
    except Exception as e:
        print(f"\nError al ejecutar el script de predicciones: {e}")
    
    pausar()


# --- Bucle Principal del Menú ---

def main():
    while True:
        limpiar_pantalla()
        print("==============================================")
        print("    Sistema de Predicción de AAPL (v0.2)    ")
        print("==============================================")
        # ¡CAMBIO! El estado ahora es solo para el modelo de evaluación
        print(f"Estado Modelo Evaluación (5d): {'[ ENTRENADO ]' if modelo_entrenado_5d else '[ NO ENTRENADO ]'}")
        print("----------------------------------------------")
        print("\nMenú Principal:")
        print("1. Menú de Gráficos")
        print("2. Entrenar Modelo de Evaluación (5 Días)")
        print("3. Ejecutar Predicciones y Guardar Log (1d, 5d, 21d)")
        print("\n0. Salir")
        
        opcion = input("\nSelecciona una opción: ")
        
        if opcion == '1':
            menu_graficos_principal()
        
        elif opcion == '2':
            ejecutar_entrenamiento_menu()
        
        elif opcion == '3':
            # --- ¡NUEVA OPCIÓN! ---
            ejecutar_prediccion_en_vivo_menu()
        
        elif opcion == '0':
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida, por favor intenta de nuevo.")
            time.sleep(1)

# --- Punto de Entrada ---
if __name__ == "__main__":
    main()