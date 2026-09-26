# patch_dietas.py — define D1 con ref. Eriksen y D4 con composicion S12
from pathlib import Path

# 1) contenido_art4.tex
f = Path("contenido_art4.tex")
s = f.read_text()

old = ("Se evaluaron dos dietas: D1, dieta estandar de sustrato de pollo\n"
       "reportada en la literatura, y D4, dieta de residuos agroindustriales\n"
       "(frutas, cascaras de naranja y residuos similares) procesada con\n"
       "\\emph{Bacillus}. En ambos casos")
assert s.count(old) == 1
new = ("Se evaluaron dos dietas. D1 es alimento comercial para pollo de\n"
       "engorde ($\\sim 20\\%$ de proteina cruda en base seca), sustrato de\n"
       "referencia en la literatura BSF y en la calibracion del modelo DEB\n"
       "\\parencite{eriksenDynamicModellingFeed2022,\n"
       "eriksenMetabolicPerformanceFeed2024}. D4 es una mezcla de residuos\n"
       "agroindustriales --- cascaras de naranja y platano, pulpas y frutas\n"
       "de descarte --- procesada por fermentacion con \\emph{Bacillus}\n"
       "(composicion aproximada 40/30/30 en peso humedo, supuesto S12).\n"
       "En ambos casos")
s = s.replace(old, new)
f.write_text(s)

# 2) supuestos.md: agregar S12
f2 = Path("supuestos.md")
t = f2.read_text()
assert "S12" not in t
t = t.rstrip() + ("\n\n- S12 Dietas: D1 = alimento comercial pollo de engorde (~20% PC en MS),\n"
                  "  referencia en eriksenDynamicModellingFeed2022/2024; D4 = mezcla de\n"
                  "  cascaras de naranja y platano, pulpas y frutas de descarte fermentada\n"
                  "  con Bacillus. Proporciones D4 40/30/30 peso humedo = SUPUESTAS, ajus-\n"
                  "  tar al registro real de la campana sep 2025 antes de envio.\n")
f2.write_text(t)
print("OK: dietas definidas (D1 con ref, D4 con S12)")
