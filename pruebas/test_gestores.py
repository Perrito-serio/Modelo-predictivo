import sys
import os

# --- Configuración de Ruta (Path) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- Fin de Configuración ---


def probar_gestores():
    """Función principal para probar los gestores."""
    
    print("--- Iniciando prueba de gestores (AAPL y SMH) ---")
    
    try:
        from datos import gestor_aapl
        from datos import gestor_smh
    except ImportError as e:
        print("\n*** ERROR FATAL: No se pudieron importar los gestores. ***")
        print(f"Detalle del error: {e}")
        print("Asegúrate de tener el archivo 'modelo predictivo/datos/__init__.py'.")
        return

    # =================================================================
    # --- 1. PRUEBA CON GESTOR AAPL (Stock) ---
    # =================================================================
    print("\n" + "="*40)
    print("      PRUEBA GESTOR: AAPL")
    print("="*40)

    try:
        print("\n--- A. Probando gestor_aapl.cargar_datos_modelo() ---")
        datos_modelo_aapl = gestor_aapl.cargar_datos_modelo()
        if datos_modelo_aapl is not None and not datos_modelo_aapl.empty:
            print(f"Datos (precios) de AAPL ({len(datos_modelo_aapl)} filas):")
            print(datos_modelo_aapl.tail(3))
        else:
            print("No se recibieron datos del modelo para AAPL.")

        print("\n--- B. Probando gestor_aapl.cargar_datos_inspector() ---")
        datos_insp_aapl = gestor_aapl.cargar_datos_inspector()
        if datos_insp_aapl:
            info_aapl = datos_insp_aapl.get('info', {})
            print("Datos del inspector de AAPL (muestra):")
            if isinstance(info_aapl, dict):
                # Para un Stock, "sector" es relevante
                print(f"  - Nombre: {info_aapl.get('longName', 'N/A')}")
                print(f"  - Sector: {info_aapl.get('sector', 'N/A')}")
            else:
                print(f"  - Info: {info_aapl}")
            
    except Exception as e:
        print(f"\n*** ERROR INESPERADO probando AAPL: {e} ***")

    # =================================================================
    # --- 2. PRUEBA CON GESTOR SMH (ETF) ---
    # =================================================================
    print("\n" + "="*40)
    print("      PRUEBA GESTOR: SMH")
    print("="*40)
    
    try:
        print("\n--- A. Probando gestor_smh.cargar_datos_modelo() ---")
        datos_modelo_smh = gestor_smh.cargar_datos_modelo()
        if datos_modelo_smh is not None and not datos_modelo_smh.empty:
            print(f"Datos (precios) de SMH ({len(datos_modelo_smh)} filas):")
            print(datos_modelo_smh.tail(3))
        else:
            print("No se recibieron datos del modelo para SMH.")

        print("\n--- B. Probando gestor_smh.cargar_datos_inspector() ---")
        datos_insp_smh = gestor_smh.cargar_datos_inspector()
        if datos_insp_smh:
            info_smh = datos_insp_smh.get('info', {})
            print("Datos del inspector de SMH (muestra):")
            if isinstance(info_smh, dict):
                # Para un ETF, "sector" no aplica.
                # Preguntamos por "legalType" o "fundFamily"
                print(f"  - Nombre: {info_smh.get('longName', 'N/A')}")
                print(f"  - Tipo Legal: {info_smh.get('legalType', 'N/A')}")
                print(f"  - Familia de Fondos: {info_smh.get('fundFamily', 'N/A')}")
            else:
                print(f"  - Info: {info_smh}")

    except Exception as e:
        print(f"\n*** ERROR INESPERADO probando SMH: {e} ***")

    print("\n" + "="*40)
    print("--- Prueba de gestores finalizada ---")


if __name__ == "__main__":
    probar_gestores()