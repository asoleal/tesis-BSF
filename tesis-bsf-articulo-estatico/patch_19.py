# patch_19.py - llave faltante en \author
from pathlib import Path
p = Path("main.tex"); s = p.read_text()
old = "{\\small ${}^{1}$[Afiliacion: departamento, universidad, ciudad, pais]}"
assert s.count(old) == 1
p.write_text(s.replace(old, old + "}"))
print("patch_19 aplicado: llave de cierre agregada")
