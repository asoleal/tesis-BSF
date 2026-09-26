# patch_10.py - calibracion de B0 y t_p por dieta contra tasas netas
from pathlib import Path
p = Path("simulacion/validacion_estatica.py"); s = p.read_text()

old1 = "def r2(obs, mod):"
new1 = '''def calibrar_larva(d, kref, YCO2):
    dias = d["dia"].to_numpy(float); obs = d["neto"].to_numpy(float)
    sat = d["sat"].to_numpy(bool)
    def resid(par):
        B0, tp = par
        pd = dict(P); pd["tp"] = tp
        xb = X0["B"]; X0["B"] = B0
        t, Y = simular(pd, kref, YCO2)
        lar, _ = componentes_ppm(t, Y, pd, kref, YCO2)
        X0["B"] = xb
        r = np.interp(dias, t, lar) - obs
        r[sat] = np.minimum(r[sat], 0.0)  # saturado = cota inferior
        return r
    sol = least_squares(resid, [0.1, 12.0], bounds=([0.005, 9.0], [3.0, 14.0]))
    return float(sol.x[0]), float(sol.x[1])

def r2(obs, mod):'''
assert s.count(old1) == 1
s = s.replace(old1, new1)

old2 = '''        kref, YCO2 = calibrar_micro(dc["dia"].values, dc["ctrl"].values)
        t, Y = simular(P, kref, YCO2)
        lar, mic = componentes_ppm(t, Y, P, kref, YCO2)
        d = d.copy()
        d["neto"] = (d["trat"] - d["ctrl"]).clip(lower=0)'''
new2 = '''        kref, YCO2 = calibrar_micro(dc["dia"].values, dc["ctrl"].values)
        d = d.copy()
        d["neto"] = (d["trat"] - d["ctrl"]).clip(lower=0)
        B0, tp = calibrar_larva(d.dropna(subset=["neto"]), kref, YCO2)
        X0["B"] = B0
        Pd = dict(P); Pd["tp"] = tp
        t, Y = simular(Pd, kref, YCO2)
        lar, mic = componentes_ppm(t, Y, Pd, kref, YCO2)'''
assert s.count(old2) == 1
s = s.replace(old2, new2)

old3 = 'print(f"\\n=== Dieta {dieta} | calibracion: k_ref={kref:.4g} d^-1, Y_CO2={YCO2:.3g} ===")'
new3 = 'print(f"\\n=== Dieta {dieta} | calibracion: k_ref={kref:.4g} d^-1, Y_CO2={YCO2:.3g}, B0={B0:.3g} mg, t_p={tp:.2f} d ===")'
assert s.count(old3) == 1
s = s.replace(old3, new3)

old4 = '        d["dieta"] = dieta'
new4 = '        d["dieta"] = dieta\n        d["B0"] = B0; d["tp"] = tp'
assert s.count(old4) == 1
s = s.replace(old4, new4)

p.write_text(s)
print("patch_10 aplicado")
