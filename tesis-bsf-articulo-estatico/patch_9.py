# patch_9.py - fix unidades ppm_min (m3 vs L) y dc desactualizado
from pathlib import Path
p = Path("simulacion/validacion_estatica.py"); s = p.read_text()

old1 = '    return mol_d / P["Vair"] * (P["Rg"] * P["T"] / P["Patm"]) * 1e6 / 1440.0'
new1 = '    return mol_d / (P["Vair"] / 1000.0) * (P["Rg"] * P["T"] / P["Patm"]) * 1e6 / 1440.0'
assert s.count(old1) == 1
s = s.replace(old1, new1)

old2 = '''        d["tot_mod"] = d["lar_mod"] + d["mic_mod"]
        dn = d.dropna(subset=["neto"])'''
new2 = '''        d["tot_mod"] = d["lar_mod"] + d["mic_mod"]
        dc = d.dropna(subset=["ctrl"])
        dn = d.dropna(subset=["neto"])'''
assert s.count(old2) == 1
s = s.replace(old2, new2)

p.write_text(s)
print("patch_9 aplicado")
