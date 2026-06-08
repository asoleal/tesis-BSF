#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reemplaza_citas_cap1_v2.py

Uso:
    python3 reemplaza_citas_cap1_v2.py --bib /ruta/absoluta/al.bib
    python3 reemplaza_citas_cap1_v2.py  # busca automaticamente

Reemplaza claves cortas tipo Allan2023 -> clave_bib completa.
"""
import os, re, sys, shutil, argparse
from collections import defaultdict

ARCHIVOS_OBJETIVO = [
    'cap1_sec01_introduccion.tex',
    'cap1_sec02_antecedentes.tex',
    'cap1_sec03_planteamiento.tex',
    'cap1_sec04_pregunta.tex',
    'cap1_sec05_hipotesis.tex',
    'cap1_sec06_objetivos.tex',
    'cap1_sec07_alcance.tex',
]
BACKUP_SUFFIX = '.backup_refs'

def log(msg):
    print(msg)

def buscar_bib_automaticamente():
    """Busca bibliografia_tesis_leal.bib hacia arriba y lados"""
    start = os.getcwd()
    # Buscar en padres
    for _ in range(4):
        for root, dirs, files in os.walk(start):
            for f in files:
                if f == 'bibliografia_tesis_leal.bib':
                    return os.path.join(root, f)
        start = os.path.dirname(start)
        if start == '/':
            break
    return None

def extraer_mapeo_bib(path):
    if not os.path.exists(path):
        log(f"[ERROR] No encontrado: {path}")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        contenido = f.read()

    entradas = re.findall(r'@\w+\{([^,\s]+)\s*,', contenido)
    author_year_map = defaultdict(list)

    for key in entradas:
        idx = contenido.find(key + ',')
        if idx == -1:
            continue
        fin = contenido.find('\n@', idx + len(key))
        if fin == -1:
            fin = len(contenido)
        bloque = contenido[idx:fin]

        author_match = re.search(r'author\s*=\s*\{([^}]+)\}', bloque)
        date_match = re.search(r'date\s*=\s*\{([^}]+)\}', bloque)
        year_match = re.search(r'year\s*=\s*\{([^}]+)\}', bloque)

        author = author_match.group(1) if author_match else ""
        date = date_match.group(1) if date_match else (year_match.group(1) if year_match else "")

        if author:
            partes = author.split(',')[0].split()
            if partes:
                apellido = re.sub(r'[^a-zA-Z]', '', partes[-1])
            else:
                apellido = ""
        else:
            apellido = ""

        year = date[:4] if date and date[:4].isdigit() else ""

        if apellido and year:
            short = apellido + year
            author_year_map[short].append(key)

    mapeo = {}
    ambiguedades = {}
    for short, claves in author_year_map.items():
        if len(claves) == 1:
            mapeo[short] = claves[0]
        else:
            elegido = sorted(claves)[0]
            mapeo[short] = elegido
            ambiguedades[short] = claves

    return mapeo, ambiguedades

def reemplazar_en_archivo(path, mapeo, ambiguedades):
    if not os.path.exists(path):
        log(f"[SKIP] No existe: {path}")
        return 0, set()

    with open(path, 'r', encoding='utf-8') as f:
        texto = f.read()

    cambios = 0
    citas_no_mapeadas = set()

    def replacer(match):
        nonlocal cambios
        comando = match.group(1)
        contenido = match.group(2)
        claves = [k.strip() for k in contenido.split(',')]
        nuevas_claves = []
        for k in claves:
            if k in mapeo:
                if mapeo[k] != k:
                    cambios += 1
                nuevas_claves.append(mapeo[k])
            else:
                nuevas_claves.append(k)
                citas_no_mapeadas.add(k)
        return f'\\{comando}{{' + ', '.join(nuevas_claves) + '}'

    nuevo_texto = re.sub(r'\\(cite|citep|citet|citeauthor|citeyear|nocite)\*?\{([^}]+)\}', replacer, texto)

    backup_path = path + BACKUP_SUFFIX
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        log(f"[BACKUP] {path}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(nuevo_texto)

    return cambios, citas_no_mapeadas

def main():
    parser = argparse.ArgumentParser(description='Reemplaza citas cortas por claves .bib completas')
    parser.add_argument('--bib', type=str, help='Ruta al archivo .bib (absoluta o relativa)')
    args = parser.parse_args()

    bib_path = args.bib
    if not bib_path:
        bib_path = buscar_bib_automaticamente()
        if bib_path:
            log(f"[AUTO] .bib encontrado: {bib_path}")
        else:
            log("[ERROR] No se encontro bibliografia_tesis_leal.bib automaticamente.")
            log("  Usa: python3 reemplaza_citas_cap1_v2.py --bib /ruta/al/archivo.bib")
            sys.exit(1)

    log("="*60)
    log("REEMPLAZO DE CITAS EN CAPITULO 1")
    log(f"BIB: {bib_path}")
    log("="*60)

    mapeo, ambiguedades = extraer_mapeo_bib(bib_path)
    log(f"[INFO] Total claves cortas mapeadas: {len(mapeo)}")
    log(f"[INFO] Ambiguedades detectadas: {len(ambiguedades)}")

    if ambiguedades:
        log("\n[AMBIGUEDADES] (se uso la primera alfabeticamente)")
        for short, claves in sorted(ambiguedades.items())[:10]:
            log(f"   {short} -> {claves[0]}  (alt: {claves[1:]})")
        if len(ambiguedades) > 10:
            log(f"   ... y {len(ambiguedades)-10} mas")

    total_cambios = 0
    total_no_mapeadas = set()

    for arch in ARCHIVOS_OBJETIVO:
        log(f"\n[PROCESANDO] {arch}")
        c, no_map = reemplazar_en_archivo(arch, mapeo, ambiguedades)
        total_cambios += c
        total_no_mapeadas.update(no_map)
        log(f"   Cambios: {c}")
        if no_map:
            log(f"   No mapeadas: {', '.join(no_map)}")

    log("\n" + "="*60)
    log("RESUMEN")
    log("="*60)
    log(f"Total reemplazos: {total_cambios}")
    if total_no_mapeadas:
        log(f"Citas no encontradas en .bib: {len(total_no_mapeadas)}")
        for c in sorted(total_no_mapeadas):
            log(f"   - {c}")
    else:
        log("Todas las citas fueron mapeadas.")
    log(f"\nBackups: *{BACKUP_SUFFIX}")
    log('Restaurar: for f in *.backup_refs; do mv "$f" "${f%.backup_refs}"; done')

if __name__ == '__main__':
    main()
