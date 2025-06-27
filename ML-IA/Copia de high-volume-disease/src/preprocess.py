import pandas as pd
import numpy as np

def clean_data(df):
    
    # Copia del DataFrame original para aplicar limpieza
    df_clean = df.copy()

    # Eliminar espacios en los encabezados de las columnas
    df_clean.columns = df_clean.columns.str.strip()

    # Mostrar los nombres de las columnas actualizados
    print(df_clean.columns.tolist())

    df_clean

    # Función para limpiar y convertir a valores numéricos con NaN
    def clean_numeric_column(col):
        return pd.to_numeric(
            df_clean[col].astype(str)
            .str.replace('>', '', regex=False)
            .str.replace('SD', '', regex=False)
            .str.strip(),
            errors='coerce'
        )

    # Aplicamos la limpieza a las columnas CD4 y CD8
    df_clean['CD4'] = clean_numeric_column('CD4')
    df_clean['CD8'] = clean_numeric_column('CD8')

    # Mostrar resumen de valores nulos y estadísticos básicos
    print(df_clean[['CD4', 'CD8']].describe().T)
    print('\n')
    print(df_clean[['CD4', 'CD8']].isna().sum())

    # Columnas categóricas a limpiar
    categorical_ = ['DXprincipal', 'Des_Dx', 'Sifilis', 'proctitis']

    # Reemplazar "SD" (y variantes con espacios) por NaN
    df_clean[categorical_] = df_clean[categorical_].replace('SD', np.nan)

    # Verificar cantidad de valores faltantes después de la limpieza
    print(df_clean[categorical_].isna().sum())
    print('\n')

    # Verificar valores únicos antes de codificar
    print(df_clean['Genero'].unique())

    # Codificar 'Genero': 1 para masculino (M), 0 para femenino (F)
    df_clean['Genero'] = df_clean['Genero'].map({'M': 1, 'F': 0})

    # Verificamos los valores codificados
    print(df_clean['Genero'].value_counts())

    # Convertir las columnas 'Fecha Nacimiento', 'Anoscopia' al formato datetime con día/mes/año
    df_clean['Fecha Nacimiento'] = pd.to_datetime(df_clean['Fecha Nacimiento'], format='%d/%m/%Y', errors='coerce')
    df_clean['Anoscopia'] = pd.to_datetime(df_clean['Anoscopia'], format='%d/%m/%Y', errors='coerce')

    # Verificar valores únicos en la columna objetivo
    print(df_clean['Enfermedad_gran_volumen'].unique())
    print('\n')
    # Mostrar las primeras filas de la columna original y la codificada
    print(df_clean['Enfermedad_gran_volumen'])
    print('\n')

    # Codificar: 1 si 'Sí', 0 si 'No'
    df_clean['Enfermedad_gran_volumen'] = df_clean['Enfermedad_gran_volumen'].map({'SI': 1, 'No': 0})

    print(df_clean['Enfermedad_gran_volumen'])
    print('\n')
    # Verificar la codificación
    print(df_clean['Enfermedad_gran_volumen'].value_counts(dropna=False))
    
    # Verificar los valores únicos en la columna 'Sifilis' antes de la conversión
    print(df_clean['Sifilis'].dropna().unique())
    print('\n')

    # Mostrar las primeras filas de la columna original
    print(df_clean['Sifilis'])

    print('\n')

    # Normalizar texto de la columna 'Sifilis'
    df_clean['Sifilis'] = (
        df_clean['Sifilis']
        .str.strip()
        .str.lower()
        .replace({'reactivo': 1, 'no reactivo': 0})
    )

    # Mostrar los valores únicos resultantes
    print(df_clean['Sifilis'].value_counts(dropna=False))
    print('\n')

    # Verificar los valores únicos en la columna 'proctitis' antes de la conversión
    print(df_clean['proctitis'].dropna().unique())
    print('\n')

    # Mostrar las primeras filas de la columna original
    print(df_clean['proctitis'])

    # Normalizar y codificar: 1 si 'Sí' (en cualquiera de sus formas), 0 si 'No'
    df_clean['proctitis'] = (
        df_clean['proctitis']
        .str.strip()
        .str.lower()
        .replace({'si': 1, 'sí': 1, 'no': 0})
    )

    # Mostrar los valores únicos resultantes
    print(df_clean['proctitis'].value_counts(dropna=False))
    print('\n')

    # Verificar los valores únicos en la columna 'proctitis' antes de la conversión
    print(df_clean['Des_Dx'].dropna().unique())
    print('\n')
    # Mostrar los valores únicos resultantes
    print(df_clean['Des_Dx'].value_counts(dropna=False))

    # Reemplazar valores NaN con 'Sin dato' para incluirlos en la codificación
    df_clean['Des_Dx_clean'] = df_clean['Des_Dx'].fillna('Sin dato')

    # Generar codificación one-hot con nombre prefijo 'Dx'
    Dx = pd.get_dummies(df_clean['Des_Dx_clean'])

    # Renombrar las columnas con nombres más cortos
    Dx.rename(columns={
        'Tumor benigno del conducto anal y del ano': 'Dx_BenignoAnal',
        'Tumor benigno del tejido cunjuntivo y de otros tejidos': 'Dx_BenignoOtro',
        'Tumor maligno del ano parte no especificada': 'Dx_Maligno',
        'Verrugas (venereas) anogenitales': 'Dx_Verrugas',
        'Sin dato': 'Dx_SinDato'
    }, inplace=True)

    # Concatenar con el DataFrame original
    df_clean = pd.concat([df_clean, Dx], axis=1)

    df_clean = df_clean.drop(columns=['Des_Dx_clean'])
    """print(40*'-')
    print(df_clean[df_clean ['CD4'] > 40000])
    print(40*'-')

    df_clean = df_clean.drop([993, 999])
    print(40*'-')
    print(df_clean[df_clean ['CD4'] > 40000])
    print(40*'-') """

    """ print(40*'-')
    print(df_clean[df_clean ['Edad'] > 120])
    print(40*'-')

    filas_a_imputar = [874, 876, 882, 887, 898, 974]

    df_clean.loc[filas_a_imputar, 'Edad'] = df_clean['Edad'].mean()
    print(40*'-')
    print(df_clean[df_clean ['Edad'] > 120])
    print(40*'-') """


    #exportar a csv
    df_clean.to_csv('Enfermedad_gran_volumen_clean.csv', index=False)


    
    return df_clean



import pandas as pd


url = 'https://raw.githubusercontent.com/John624-web/curso-IA-John/refs/heads/main/IA%20Coloproctologia.csv'


df = pd.read_csv(url)

print(df.info())

df