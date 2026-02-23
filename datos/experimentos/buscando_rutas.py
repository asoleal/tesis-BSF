import os

print("--- DIAGNÓSTICO DE RUTAS ---")
cwd = os.getcwd()
print(f"📂 Carpeta actual de trabajo: {cwd}")
print("\n📋 Contenido de esta carpeta:")

elementos = os.listdir(cwd)
carpetas_encontradas = [d for d in elementos if os.path.isdir(d)]

for item in elementos:
    tipo = "📁" if os.path.isdir(item) else "📄"
    print(f"  {tipo} {item}")

print("\n--- BUSCANDO TUS EXPERIMENTOS ---")
# Nombres que el script anterior estaba buscando
carpetas_buscadas = [
    "experimento1_D1_voraz_tenian_hambre",
    "experimento2_D4",
    "experimento3_D1_mas_real",
    "experimento4-alimento_D1",
    "experimento4-alimento-D4"
]

for buscada in carpetas_buscadas:
    if buscada in carpetas_encontradas:
        print(f"✅ ENCONTRADA: {buscada}")
        # Verificar si tiene los CSV dentro
        archivos_dentro = os.listdir(os.path.join(cwd, buscada))
        if "experimento_co2.csv" in archivos_dentro:
            print(f"   └── ✅ Tiene experimento_co2.csv")
        else:
            print(f"   └── ❌ FALTA experimento_co2.csv (Veo: {archivos_dentro})")
    else:
        print(f"❌ NO ENCONTRADA: {buscada}")
        # Intentar buscar parecidos
        for real in carpetas_encontradas:
            if buscada[:10] in real: # Si los primeros 10 caracteres coinciden
                print(f"   💡 ¿Quizás quisiste decir '{real}'?")

