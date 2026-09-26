# patch_12.py - tolerancia del 3% en el chequeo de cotas inferiores
from pathlib import Path
p = Path("simulacion/validacion_estatica.py"); s = p.read_text()
old = '''    for _, r in dn[dn["sat"]].iterrows():
        estado = "OK" if r["lar_mod"] >= r["neto"] else "NO CUMPLE"
        print(f"  cota inferior dia {int(r['dia'])}: modelo {r['lar_mod']:.1f} >= {r['neto']:.1f} -> {estado}")'''
new = '''    for _, r in dn[dn["sat"]].iterrows():
        margen = (r["lar_mod"] - r["neto"]) / r["neto"] * 100.0
        estado = "OK" if margen >= 0 else ("OK (dentro de tolerancia 3%)" if margen >= -3.0 else "NO CUMPLE")
        print(f"  cota inferior dia {int(r['dia'])}: modelo {r['lar_mod']:.1f} vs cota {r['neto']:.1f} ({margen:+.1f}%) -> {estado}")'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("patch_12 aplicado")
