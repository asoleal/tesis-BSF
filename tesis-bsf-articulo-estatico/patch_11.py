# patch_11.py - agregar w_p (ancho del interruptor) al ajuste conjunto
from pathlib import Path
p = Path("simulacion/validacion_estatica.py"); s = p.read_text()

old1 = '''def curva_lar(dias, B0, tp, f, kref, YCO2):
    pd = dict(P); pd["tp"] = tp; pd["f"] = f'''
new1 = '''def curva_lar(dias, B0, tp, f, kref, YCO2, wp=1.0):
    pd = dict(P); pd["tp"] = tp; pd["f"] = f; pd["wp"] = wp'''
assert s.count(old1) == 1; s = s.replace(old1, new1)

old2 = '''        B0 = 10.0**par[0]; f4, tp1, tp4 = par[1], par[2], par[3]
        r = []
        for dieta, tp, f in [("D1", tp1, 1.0), ("D4", tp4, f4)]:
            d = datos[dieta].dropna(subset=["neto"])
            _, _, mod = curva_lar(d["dia"].to_numpy(float), B0, tp, f,
                                  kmic[dieta], ymic[dieta])'''
new2 = '''        B0 = 10.0**par[0]; f4, tp1, tp4, wp = par[1], par[2], par[3], par[4]
        r = []
        for dieta, tp, f in [("D1", tp1, 1.0), ("D4", tp4, f4)]:
            d = datos[dieta].dropna(subset=["neto"])
            _, _, mod = curva_lar(d["dia"].to_numpy(float), B0, tp, f,
                                  kmic[dieta], ymic[dieta], wp)'''
assert s.count(old2) == 1; s = s.replace(old2, new2)

old3 = '''    mejor = None
    for x0 in [[np.log10(0.012), 0.8, 11.0, 11.0],
               [np.log10(0.05), 0.5, 10.0, 12.0],
               [np.log10(0.1), 0.7, 12.0, 10.5]]:
        sol = least_squares(resid, x0, bounds=([np.log10(0.008), 0.2, 9.0, 9.0],
                                               [np.log10(0.5), 1.2, 14.0, 14.0]))
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol
    return 10.0**mejor.x[0], mejor.x[1], mejor.x[2], mejor.x[3]'''
new3 = '''    mejor = None
    for x0 in [[np.log10(0.012), 0.8, 11.0, 11.0, 1.0],
               [np.log10(0.05), 0.5, 10.0, 12.0, 0.5],
               [np.log10(0.1), 0.7, 12.0, 10.5, 0.7]]:
        sol = least_squares(resid, x0, bounds=([np.log10(0.008), 0.2, 9.0, 9.0, 0.3],
                                               [np.log10(0.5), 1.2, 14.0, 14.0, 2.0]))
        if mejor is None or sol.cost < mejor.cost:
            mejor = sol
    return 10.0**mejor.x[0], mejor.x[1], mejor.x[2], mejor.x[3], mejor.x[4]'''
assert s.count(old3) == 1; s = s.replace(old3, new3)

old4 = '''def evaluar(dieta, d, B0, tp, f, kref, YCO2, ayuno):'''
new4 = '''def evaluar(dieta, d, B0, tp, f, wp, kref, YCO2, ayuno):'''
assert s.count(old4) == 1; s = s.replace(old4, new4)

old5 = '''    t, lar, mod_en_dias = curva_lar(d["dia"].to_numpy(float), B0, tp, f, kref, YCO2)
    pd = dict(P); pd["tp"] = tp; pd["f"] = f'''
new5 = '''    t, lar, mod_en_dias = curva_lar(d["dia"].to_numpy(float), B0, tp, f, kref, YCO2, wp)
    pd = dict(P); pd["tp"] = tp; pd["f"] = f; pd["wp"] = wp'''
assert s.count(old5) == 1; s = s.replace(old5, new5)

old6 = '''f"B0={B0:.3g} mg, f={f:.3g}, t_p={tp:.2f} d ===")'''
new6 = '''f"B0={B0:.3g} mg, f={f:.3g}, t_p={tp:.2f} d, w_p={wp:.2f} d ===")'''
assert s.count(old6) == 1; s = s.replace(old6, new6)

old7 = '''    d["dieta"] = dieta; d["B0"] = B0; d["f"] = f; d["tp"] = tp'''
new7 = '''    d["dieta"] = dieta; d["B0"] = B0; d["f"] = f; d["tp"] = tp; d["wp"] = wp'''
assert s.count(old7) == 1; s = s.replace(old7, new7)

old8 = '''    B0, f4, tp1, tp4 = calibrar_conjunto(datos, kmic, ymic)
    print(f"\\n*** Ajuste conjunto: B0={B0:.3g} mg/larva (postura de huevos), "
          f"f_D4={f4:.3g} (D1=1 referencia), t_p D1={tp1:.2f} d, t_p D4={tp4:.2f} d ***")
    filas = []
    for dieta, tp, f in [("D1", tp1, 1.0), ("D4", tp4, f4)]:
        filas.append(evaluar(dieta, datos[dieta], B0, tp, f, kmic[dieta], ymic[dieta], ayuno))'''
new8 = '''    B0, f4, tp1, tp4, wp = calibrar_conjunto(datos, kmic, ymic)
    print(f"\\n*** Ajuste conjunto: B0={B0:.3g} mg/larva (postura de huevos), "
          f"f_D4={f4:.3g} (D1=1 referencia), t_p D1={tp1:.2f} d, t_p D4={tp4:.2f} d, "
          f"w_p={wp:.2f} d ***")
    filas = []
    for dieta, tp, f in [("D1", tp1, 1.0), ("D4", tp4, f4)]:
        filas.append(evaluar(dieta, datos[dieta], B0, tp, f, wp, kmic[dieta], ymic[dieta], ayuno))'''
assert s.count(old8) == 1; s = s.replace(old8, new8)

p.write_text(s)
print("patch_11 aplicado: w_p calibrable")
