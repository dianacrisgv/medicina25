import pandas as pd

def clean_data(df):
    """
     # Eliminar filas con valores faltantes representados como '?'
    df_clean = df.replace('?', pd.NA).dropna()
    
    # Convertir columnas a tipo numérico si es necesario
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
    """
    df_clean = df
    return df_clean