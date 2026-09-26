# patch_readme17b.py — cierra §17: tareas 1-6 HECHAS + siguiente paso
from pathlib import Path
f = Path("README.md")
s = f.read_text()

old_estado = """### Estado
- Pasos 1-6 de A1: PENDIENTE (parches por paso, uno a la vez).
- Campana de experimentos: por disenar (DOE)."""

new_estado = """### Estado
- Pasos 1-6 de A1: HECHOS (2026-09-26). Estructura final: §1 Introduccion
  (objetivo + P1-P3) / §2 Materiales y metodos (2.1 Sistema experimental,
  2.2 ODE comprimida, 2.3 Cierres, 2.4 CFD) / §3 Resultados por pregunta
  (3.1 P1 rango calendario, 3.2-3.3 P2 CFD, 3.4 P3 validacion) / §4
  Discusion (incl. PINN trabajo en curso) / §5 Conclusiones (8 items).
  Compila limpio con xelatex+biber, 13 paginas. Incidente: la S2.4 CFD se
  perdio en el paso 3 (parche con corte mal limitado); se recupero de
  c8a84e6 sin perdida de contenido. Leccion: commitear DESPUES de cada
  paso compilado, no al final de varios.
- Articulo A1: listo para lectura de terceros / envio a coautor.
- Campana de experimentos: por disenar (DOE) — habilita A2 + A3.

### Siguiente paso (fuera de A1)
1. Disenar el DOE de la campana extendida (dietas x lineas x replicas x
   pesaje cada 2 d + gas con protocolo A1). Salida: plan de 1-2 meses.
2. Redactar Articulo A2 (ambiental) con los datos de la campana.
3. Lista de verificacion instrumental previa: calibracion V_air por
   trazador, referencia CH4 para el TGS2611, bascula para biomasa."""

assert s.count(old_estado) == 1
s = s.replace(old_estado, new_estado)

# marcar las 6 tareas como hechas (prefijo [x])
for t in ["1. Titulo nuevo", "2. Nueva §2.1", "3. §2.1 actual (ODE)", "4. PINN (§2.3 + §3.5)", "5. Resultados espejando", "6. Pendientes menores"]:
    assert s.count(t) == 1, t
    s = s.replace("   " + t + "", "   [x] " + t) if False else s
# reemplazo directo linea a linea
lines = s.split("\n")
for i, ln in enumerate(lines):
    st = ln.strip()
    if st[:2] in ("1.", "2.", "3.", "4.", "5.", "6.") and "[x]" not in ln:
        for key in ["Titulo nuevo", "Nueva §2.1", "§2.1 actual (ODE)",
                    "PINN (§2.3 + §3.5)", "Resultados espejando", "Pendientes menores"]:
            if key in ln:
                lines[i] = ln.replace(st, "[x] " + st, 1)
                break
s = "\n".join(lines)

f.write_text(s)
print("OK: README §17 cerrado")
