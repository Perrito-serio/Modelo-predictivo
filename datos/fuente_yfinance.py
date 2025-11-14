import yfinance as yf
import pandas as pd
import sys

# Este es el archivo que se conecta a Yahoo Finance
# para descargar los datos.

def obtener_datos_precios(ticker: str, periodo: str = "10y"):
    """
    Descarga datos históricos de precios (OHLCV) para un ticker.
    Llamado por los gestores (ej. gestor_aapl.py).
    """
    print(f"[fuente_yfinance] Descargando datos de precios para {ticker} ({periodo})...")
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=periodo)
        
        if df.empty:
            print(f"*** [fuente_yfinance] No se encontraron datos de precios para {ticker} ***")
            return None
            
        # Asegurarse de que el índice es solo Fecha (sin hora)
        df.index = df.index.tz_convert(None).normalize()
        
        print(f"[fuente_yfinance] Datos de precios para {ticker} cargados.")
        return df
        
    except Exception as e:
        print(f"*** [fuente_yfinance] Error fatal al descargar precios para {ticker}: {e} ***")
        # sys.exit(1) # Podríamos salir, pero es mejor retornar None y dejar que el gestor falle
        return None

def obtener_datos_relevantes(ticker_symbol: str):
    """
    Obtiene información 'info' y otros datos relevantes 
    para el menú 'inspector' en main.py.
    """
    print(f"[fuente_yfinance] Descargando 'info' (inspector) para {ticker_symbol}...")
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info
        
        if not info or info.get('trailingPE') is None: # 'trailingPE' es una comprobación de que 'info' está poblado
            print(f"*** [fuente_yfinance] No se encontró 'info' completa para {ticker_symbol} ***")
            return None
            
        print(f"[fuente_yfinance] Datos 'info' para {ticker_symbol} cargados.")
        
        # El script 'pruebas/test_gestores.py' espera un diccionario
        # que contenga al menos la clave 'info'.
        return {
            "info": info
            # Aquí se podrían añadir más datos si el inspector los usara:
            # "news": t.news,
            # "recommendations": t.recommendations
        }
        
    except Exception as e:
        print(f"*** [fuente_yfinance] Error fatal al descargar 'info' para {ticker_symbol}: {e} ***")
        return None