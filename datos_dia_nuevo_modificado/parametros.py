def extraer_parametros_biologicos(model, nombre_tratamiento):
    # Obtener los parámetros del modelo (se pasan por softplus para ser positivos)
    params_tensor = model.get_physics_params()
    
    # Convertir a números normales de Python
    params = params_tensor.detach().numpy()
    Y_g = params[0] # Yield Growth (Costo de crear estructura)
    Y_l = params[1] # Yield Lipids (Costo de crear/quemar grasa)
    m   = params[2] # Maintenance (Costo de mantenimiento basal)
    
    print(f"--- 🧬 PARÁMETROS DESCUBIERTOS: {nombre_tratamiento} ---")
    print(f"  [Y_g] Costo Energético de Crecer:      {Y_g:.5f}")
    print(f"  [Y_l] Costo Energético de Grasas:      {Y_l:.5f}")
    print(f"  [m]   Tasa de Mantenimiento Basal:     {m:.5f}")
    print("-" * 50)

# Ejecutar para ambos modelos
extraer_parametros_biologicos(model_d1, "TRATAMIENTO D1 (Voraz/Estrés)")
extraer_parametros_biologicos(model_d4, "TRATAMIENTO D4 (Crecimiento)")
