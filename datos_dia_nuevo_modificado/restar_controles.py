import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar datos previos
df = pd.read_csv("datos_finales_PINN_corregidos.csv")

def aplicar_resta(df_total, tag_tratamiento, tag_control):
    # Separar tratamiento y su control específico
    tratam = df_total[df_total['Etiqueta'] == tag_tratamiento].copy()
    control = df_total[df_total['Etiqueta'] == tag_control].copy()

    # Calcular promedio del control por día y gas (para suavizar ruido)
    promedios_control = control.groupby(['Dia_Experimento', 'Gas'])['Tasa_Produccion'].mean().reset_index()
    promedios_control.rename(columns={'Tasa_Produccion': 'Tasa_Base'}, inplace=True)

    # Unir (Merge) para tener el valor del control al lado del tratamiento
    merged = pd.merge(tratam, promedios_control, on=['Dia_Experimento', 'Gas'], how='left')

    # Si no hay control ese día, asumimos 0 (o mantenemos el valor original)
    merged['Tasa_Base'] = merged['Tasa_Base'].fillna(0)

    # --- LA RESTA MAGISTRAL ---
    merged['Tasa_Neta'] = merged['Tasa_Produccion'] - merged['Tasa_Base']

    # Limpieza física: La producción no puede ser negativa (si el control fue > tratamiento por ruido)
    merged['Tasa_Neta'] = merged['Tasa_Neta'].clip(lower=0)

    return merged

# 2. Ejecutar para D1 y D4
df_d1 = aplicar_resta(df, "D1_Tratamiento", "Control_Alimento_D1")
df_d4 = aplicar_resta(df, "D4_Tratamiento", "Control_Alimento_D4")

# 3. Unir y Guardar
df_final = pd.concat([df_d1, df_d4], ignore_index=True)
csv_name = "datos_PINN_netos.csv"
df_final.to_csv(csv_name, index=False)

print(f"✅ ¡Hecho! Datos limpios guardados en: {csv_name}")
print(df_final[['Dia_Experimento', 'Etiqueta', 'Gas', 'Tasa_Produccion', 'Tasa_Base', 'Tasa_Neta']].head())

# 4. Visualización Comparativa (Bruto vs Neto)
plt.figure(figsize=(12, 6))

# Filtrar solo CO2 para la gráfica
data_plot = df_final[df_final['Gas'] == 'co2']

plt.subplot(1, 2, 1)
sns.lineplot(data=data_plot, x='Dia_Experimento', y='Tasa_Produccion', hue='Etiqueta', marker='o')
plt.title("Datos Brutos (Incluye Alimento)")
plt.ylabel("ppm/min")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
sns.lineplot(data=data_plot, x='Dia_Experimento', y='Tasa_Neta', hue='Etiqueta', marker='o', linestyle='--')
plt.title("Datos Netos (Solo Larvas)")
plt.ylabel("ppm/min (Limpio)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
