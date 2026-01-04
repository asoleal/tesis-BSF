import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el archivo generado
try:
    df = pd.read_csv('datos_entrenamiento_PINN.csv')
    print("Datos cargados correctamente.")
    print(df.head())
except FileNotFoundError:
    print("Error: No encuentro 'datos_entrenamiento_PINN.csv'")
    exit()

# 2. Configurar etiquetas para que la gráfica sea legible
# Mapeamos los números 0/1 a texto real
df['Nombre_Dieta'] = df['dieta'].map({0: 'D1 (Rápido)', 1: 'D4 (Lento)'})
df['Nombre_Gas'] = df['gas_tipo'].map({0: 'CO2 (Respiración)', 1: 'CH4 (Fermentación)'})

# 3. Graficar
plt.figure(figsize=(12, 6))

# Usamos Seaborn para graficar líneas de tendencia automáticamente
# Hue=Dieta (Colores diferentes para D1 y D4)
# Style=Gas (Línea sólida para CO2, punteada para CH4)
sns.lineplot(data=df, x='dia', y='tasa_neta', hue='Nombre_Dieta', style='Nombre_Gas', markers=True, dashes=True)

plt.title("Evolución Metabólica: Tasa de Producción de Gas vs Tiempo")
plt.xlabel("Días de Desarrollo")
plt.ylabel("Tasa de Producción (ppm/min)")
plt.grid(True, alpha=0.3)
plt.legend(title='Condiciones')
plt.show()
