# patch_18.py - autores del articulo
from pathlib import Path
p = Path("main.tex"); s = p.read_text()
lineas = s.split("\n")
for i, l in enumerate(lineas):
    if l.strip().startswith("\\author"):
        lineas[i] = ("\\author{[Tu nombre completo]${}^{1}$, "
                     "Sorany Milena Barrientos${}^{1}$ y Luis Octavio Gonz\\'alez${}^{1}$\\\\[4pt]\n"
                     "{\\small ${}^{1}$[Afiliacion: departamento, universidad, ciudad, pais]}")
        break
else:
    raise SystemExit("no se encontro \\author")
p.write_text("\n".join(lineas))
print("patch_18 aplicado: autores agregados")
