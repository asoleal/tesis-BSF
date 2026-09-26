from pathlib import Path

# --- 1) script: deduplicar filas identicas (mismo dia+etiqueta+tasa) ---
p = Path("simulacion/validacion_experimento.py")
s = p.read_text()
old = """    import csv as _csv
    filas = [r for r in _csv.DictReader(open(csv))
             if r['Gas'] == 'co2' and float(r['R2']) >= 0.5]
"""
new = """    import csv as _csv
    filas = [r for r in _csv.DictReader(open(csv))
             if r['Gas'] == 'co2' and float(r['R2']) >= 0.5]
    # deduplicar filas identicas (mismo dia + etiqueta + tasa): una medicion
    # no debe contar dos veces en el promedio de la jornada
    _vistos, _fs = set(), []
    for _r in filas:
        _k = (_r['Dia_Experimento'], _r['Etiqueta'],
              round(float(_r['Tasa_Produccion']), 6))
        if _k not in _vistos:
            _vistos.add(_k)
            _fs.append(_r)
    filas = _fs
"""
assert s.count(old) == 1, "script"
s = s.replace(old, new)
p.write_text(s)

# --- 2) articulo: n = 3 replicas + rango entre jornadas + cuadro ---
p = Path("contenido_art4.tex")
s = p.read_text()

old = "mismo dia (supuesto S7). Las tasas extraidas, con su $R^2$, se\nconsolidan en"
new = ("mismo dia (supuesto S7). El protocolo repitio el par cierre/control "
       "para cada dieta en las tres jornadas con alimento (dias 9, 11 y 13) "
       "del mismo lote de cria, de modo que cada dieta aporta $n = 3$ cierres "
       "independientes y el valor de cada jornada es el promedio de sus "
       "cierres. Las tasas extraidas, con su $R^2$, se\nconsolidan en")
assert s.count(old) == 1, "2.1 n=3"
s = s.replace(old, new)

old = "las resume por dia del ciclo. El panel (a) muestra"
new = ("las resume por dia del ciclo, con el rango entre las tres replicas "
       "de cada dia. El panel (a) muestra")
assert s.count(old) == 1, "3.4 rango"
s = s.replace(old, new)

old = "9  & 210 & 89--309 \\\\"
new = "9  & 210 & 89--268 \\\\"
assert s.count(old) == 1, "cuadro"
s = s.replace(old, new)

p.write_text(s)
print("OK: replicas — dedup en script + n=3 en 2.1/3.4 + cuadro 89--268")
