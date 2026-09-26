# patch_paso6.py — cosmeticos: ancho Fig2 + caveat Fig3b al caption
from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()

# 1) Figura 2 (escenarios): 0.92\textwidth
old = ("\\includegraphics[width=\\textwidth]{imagenes/sim_E2.pdf}\\\\[2mm]\n"
       "  \\includegraphics[width=\\textwidth]{imagenes/sim_E5.pdf}")
new = ("\\includegraphics[width=0.92\\textwidth]{imagenes/sim_E2.pdf}\\\\[2mm]\n"
       "  \\includegraphics[width=0.92\\textwidth]{imagenes/sim_E5.pdf}")
assert s.count(old) == 1
s = s.replace(old, new)

# 2) caption fig:validacion — agregar caveat del panel (b)
old_cap = ("prepupa $S_{pr}(t)$; las lineas punteadas marcan la reposicion de\n"
           "  alimento.}")
new_cap = ("prepupa $S_{pr}(t)$; las lineas punteadas marcan la reposicion de\n"
           "  alimento. Nota: el panel (b) es una trayectoria del modelo, no\n"
           "  una medicion --- en este experimento no se peso biomasa.}")
assert s.count(old_cap) == 1
s = s.replace(old_cap, new_cap)

f.write_text(s)
print("OK: paso 6 (tex)")
