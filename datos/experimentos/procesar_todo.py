import pandas as pd
import os
from scipy.stats import linregress
import numpy as np

# --- 1. FUNCIONES (Las mismas que ya probaste) ---
def limpiar_y_recortar(ruta_archivo, umbral_salto=200):
    try:
        # Intenta leer con tabulaciones o espacios
        try:
            df = pd.read_csv(ruta_archivo, sep='\t', header=None, engine='python')
        except:
            df = pd.read_csv(ruta_archivo, sep='\s+', header=None, engine='python')

        if df.shape[1] < 2: return None # Archivo vacío o mal formato

        df.columns = ['hora', 'valor']
        df['hora_dt'] = pd.to_datetime(df['hora'], format='%H:%M:%S', errors='coerce')
        df = df.dropna() # Eliminar errores de hora

        # Detectar salto
        df['diferencia'] = df['valor'].diff()
        indices_salto = df[df['diferencia'] > umbral_salto].index

        idx_inicio = indices_salto[0] if len(indices_salto) > 0 else 0

        df_limpio = df.iloc[idx_inicio:].copy()
        hora_cero = df_limpio['hora_dt'].iloc[0]
        df_limpio['minutos'] = (df_limpio['hora_dt'] - hora_cero).dt.total_seconds() / 60.0

        return df_limpio
    except Exception as e:
        print(f"   [!] Error leyendo {ruta_archivo}: {e}")
        return None

def obtener_tasa(df):
    if df is None or len(df) < 5: return 0 # Si hay muy pocos datos, tasa 0
    slope, _, _, _, _ = linregress(df['minutos'], df['valor'])
    return slope

# --- 2. EL BUCLE PRINCIPAL (Tu Fábrica) ---

dias = ['9', '11', '13', '17'] # Tus carpetas de días
dietas = ['D1', 'D4']          # Tus tratamientos
gases = ['co2', 'ch4']         # Tus variables

resultados = []

print("Iniciando procesamiento masivo...")

for dia in dias:
    for dieta in dietas:
        for gas in gases:
            # Construir rutas basadas en tu estructura de carpetas
            ruta_exp = f"{dia}/experimento_{dieta}/experimento_{gas}.csv"
            ruta_ctrl = f"{dia}/experimento_{dieta}_alimento/experimento_{gas}.csv"

            # Verificar si existen (ej. Día 17 D1 puede no existir)
            if os.path.exists(ruta_exp) and os.path.exists(ruta_ctrl):
                print(f"Procesando: Día {dia} | {dieta} | {gas} ...")

                # Ajustar umbral: CH4 suele necesitar un umbral más alto o más bajo según tus datos
                umbral = 200 if gas == 'co2' else 100

                # 1. Calcular Tasa Experimento
                df_exp = limpiar_y_recortar(ruta_exp, umbral_salto=umbral)
                tasa_exp = obtener_tasa(df_exp)

                # 2. Calcular Tasa Control (Ruido)
                df_ctrl = limpiar_y_recortar(ruta_ctrl, umbral_salto=50) # Umbral bajo para control
                tasa_ctrl = obtener_tasa(df_ctrl)

                # 3. Calcular Tasa Neta
                tasa_neta = tasa_exp - tasa_ctrl

                # Guardar en la lista
                resultados.append({
                    'dia': int(dia),
                    'dieta': 0 if dieta == 'D1' else 1, # Codificamos D1=0, D4=1 para la IA
                    'gas_tipo': 0 if gas == 'co2' else 1, # CO2=0, CH4=1
                    'tasa_neta': tasa_neta if tasa_neta > 0 else 0 # Evitamos tasas negativas
                })
            else:
                print(f"   [i] Saltando Día {dia} {dieta} (Archivo no encontrado)")

# --- 3. GUARDAR EL RESULTADO FINAL ---
df_final = pd.DataFrame(resultados)
nombre_salida = 'datos_entrenamiento_PINN.csv'
df_final.to_csv(nombre_salida, index=False)

print("\n" + "="*50)
print(f"¡LISTO! Se ha creado el archivo: {nombre_salida}")
print(df_final.head(10)) # Mostrar las primeras filas
print("="*50)
