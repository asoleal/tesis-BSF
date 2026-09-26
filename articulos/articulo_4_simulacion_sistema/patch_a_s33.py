from pathlib import Path
f = Path("contenido_art4.tex")
s = f.read_text()
ini = "\\subsection{Cierres adaptativos}"
fin = "\\subsection{Validacion contra datos preliminares}"
assert s.count(ini) == 1 and s.count(fin) == 1, (s.count(ini), s.count(fin))
i0, i1 = s.index(ini), s.index(fin)
nuevo = r'''\subsection{Cierres adaptativos}

El calendario aplicado en las simulaciones es el adaptativo de dos pasadas
descrito en la Seccion 2.2: una primera pasada con cierres nominales de
$\Delta = 15$ min y una segunda con duraciones ajustadas al margen del NDIR
con la tasa evaluada en el punto medio del cierre. En la practica los
cierres resultantes se acortan a pocos minutos en el pico de emision del
ciclo y se alargan hasta 30 min al inicio y en la fase de ayuno. Se evaluo
ademas la sensibilidad al volumen de aire efectivo $V_{aire}$: una
incertidumbre de $\pm 20\%$ en $V_{aire}$ se traduce directamente en igual
incertidumbre en las tasas estimadas, lo que justifica su calibracion
experimental por dilucion de un trazador.

'''
s = s[:i0] + nuevo + s[i1:]
f.write_text(s)
print("OK: S3.3 alineada con S2.2")
