# --- ¡CAMBIO AQUÍ! ---
# Antes decía: from .fuente_yfinance import ...
# Ahora dice:
from datos.fuente_yfinance import obtener_datos_precios, obtener_datos_relevantes

# Ticker específico para este gestor
TICKER_GESTOR = "SMH"

def cargar_datos_modelo():
    """
    Función que llamará 'gestor_modelo.py'.
    Obtiene los datos de precios para el Ticker de ESTE gestor (SMH).
    """
    print(f"[gestor_smh] Pidiendo datos del modelo para {TICKER_GESTOR}")
    return obtener_datos_precios(ticker=TICKER_GESTOR, periodo="10y")

def cargar_datos_inspector():
    """
    Función que llamará 'main.py' para el menú.
    Obtiene TODOS los datos relevantes de ESTE gestor (SMH).
    """
    print(f"[gestor_smh] Pidiendo datos del inspector para {TICKER_GESTOR}")
    return obtener_datos_relevantes(ticker_symbol=TICKER_GESTOR)