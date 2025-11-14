import sys
import os
import pandas as pd
import mplfinance as mpf # ¡NUEVA BIBLIOTECA!

# --- Configuración de Ruta (Path) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- Fin de Configuración ---

try:
    from datos import gestor_aapl
    from datos import gestor_smh
except ImportError:
    print("Error: No se pudieron importar los gestores desde 'datos/'.")
    sys.exit(1)

def graficar_velas_ventana(df_precios: pd.DataFrame, ticker_nombre: str):
    """
    Toma un DataFrame y genera un gráfico de velas en una
    ventana emergente de Matplotlib.
    """
    if df_precios is None or df_precios.empty:
        print(f"No hay datos para graficar para {ticker_nombre}")
        return

    print(f"[Graficador] Creando gráfico de velas para {ticker_nombre}...")
    
    # Acortamos el DataFrame a los últimos 6 meses (180 días)
    # mplfinance se ve mejor con menos datos que plotly
    df_reciente = df_precios.last('180D')
    
    # ¡Mostrar el gráfico!
    # Esto abrirá una ventana emergente de matplotlib
    mpf.plot(
        df_reciente,
        type='candle',         # Tipo de gráfico: velas
        style='yahoo',         # Estilo (colores)
        title=f'Gráfico de Velas: {ticker_nombre} (Últimos 6 Meses)',
        ylabel='Precio (USD)',
        volume=True,           # Mostrar el volumen en un panel separado
        mav=(20, 50)           # Añadir medias móviles de 20 y 50 días
    )

