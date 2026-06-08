#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_refs_tesis.py

Script Python para verificar referencias cruzadas rotas en un proyecto LaTeX.
Detecta:
  1. ?? explícitos en \ref, \eqref, \cite, \label, etc.
  2. Referencias huérfanas (usadas en \ref pero sin \label definido).
  3. Labels definidas pero nunca referenciadas.
  4. Claves bibliográficas citadas pero ausentes en .bib.

Uso:
    python verificar_refs_tesis.py /ruta/a/tesis-BSF/tesis/Capitulos

Autor: Generado para tesis doctoral BSF
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict


def main(tex_dir: str, bib_file: str = None):
    tex_path = Path(tex_dir)
    if not tex_path.exists():
        print(f"ERROR: Directorio no encontrado: {tex_dir}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Indexar archivos .tex
    # -------------------------------------------------------------------------
    tex_files = sorted(tex_path.rglob("*.tex"))
    print(f"[INFO] Archivos .tex encontrados: {len(tex_files)}")

    labels = defaultdict(list)      # label -> [(archivo, linea, texto)]
    refs = defaultdict(list)       # ref -> [(archivo, linea, texto)]
    cites = defaultdict(list)       # cite_key -> [(archivo, linea, texto)]
    explicit_q = []                  # lista de tuplas (archivo, linea, texto) con ??

    label_pattern = re.compile(r'\\label\{([^}]+)\}')
    ref_pattern = re.compile(r'\\(ref|eqref|autoref|cref|Cref|pageref)\{([^}]+)\}')
    cite_pattern = re.compile(r'\\(cite|citet|citep|citeauthor|citeyear)[^{]*\{([^}]+)\}')
    explicit_q_pattern = re.compile(r'\\(ref|eqref|cite|label|autoref|cref|Cref|pageref)\{\?\?\}')
    any_q_pattern = re.compile(r'\?\?')

    for fpath in tex_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[WARN] No se pudo leer {fpath}: {e}")
            continue

        for lineno, line in enumerate(lines, start=1):
            # Labels definidas
            for m in label_pattern.finditer(line):
                lbl = m.group(1)
                labels[lbl].append((fpath.name, lineno, line.strip()[:80]))

            # Referencias usadas
            for m in ref_pattern.finditer(line):
                ref_key = m.group(2)
                refs[ref_key].append((fpath.name, lineno, line.strip()[:80]))

            # Citas bibliográficas
            for m in cite_pattern.finditer(line):
                cite_keys_raw = m.group(2)
                for key in [k.strip() for k in cite_keys_raw.split(',')]:
                    if key:
                        cites[key].append((fpath.name, lineno, line.strip()[:80]))

            # ?? explícitos en comandos LaTeX
            if explicit_q_pattern.search(line):
                explicit_q.append((fpath.name, lineno, line.strip()[:80]))

    # -------------------------------------------------------------------------
    # 2. Reporte: ?? explícitos
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(" 1. ?? EXPLÍCITOS EN COMANDOS LaTeX (referencias rotas visibles)")
    print("="*60)
    if explicit_q:
        for fname, lineno, text in explicit_q:
            print(f"   {fname}:{lineno} | {text}")
    else:
        print("   [OK] No se encontraron ?? explícitos en comandos.")

    # -------------------------------------------------------------------------
    # 3. Referencias huérfanas (usadas pero sin label)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(" 2. REFERENCIAS HUÉRFANAS (\\ref usado pero \\label no definido)")
    print("="*60)
    orphan_refs = []
    for ref_key, occurrences in refs.items():
        if ref_key not in labels:
            orphan_refs.append((ref_key, occurrences))
    if orphan_refs:
        for ref_key, occurrences in sorted(orphan_refs, key=lambda x: x[0]):
            print(f"\n   \\ref{{{ref_key}}} usado en:")
            for fname, lineno, text in occurrences:
                print(f"      {fname}:{lineno}")
    else:
        print("   [OK] Todas las referencias tienen su \\label correspondiente.")

    # -------------------------------------------------------------------------
    # 4. Labels sin uso (definidas pero nunca referenciadas)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(" 3. LABELS SIN USO (definidas pero nunca referenciadas) [ADVERTENCIA]")
    print("="*60)
    unused_labels = []
    for lbl, occurrences in labels.items():
        if lbl not in refs:
            unused_labels.append((lbl, occurrences))
    if unused_labels:
        for lbl, occurrences in sorted(unused_labels, key=lambda x: x[0]):
            print(f"\n   \\label{{{lbl}}} definida en:")
            for fname, lineno, text in occurrences:
                print(f"      {fname}:{lineno}")
    else:
        print("   [OK] Todas las labels están referenciadas.")

    # -------------------------------------------------------------------------
    # 5. Verificar bibliografía
    # -------------------------------------------------------------------------
    if bib_file and Path(bib_file).exists():
        print("\n" + "="*60)
        print(" 4. CITAS BIBLIOGRÁFICAS vs ARCHIVO .bib")
        print("="*60)
        with open(bib_file, 'r', encoding='utf-8') as bf:
            bib_content = bf.read()
        # Extraer claves @article{clave, ...
        bib_keys = set(re.findall(r'^@[a-zA-Z]+\{([^,\s]+)', bib_content, re.MULTILINE))
        print(f"   Claves en .bib: {len(bib_keys)}")

        orphan_cites = []
        for cite_key, occurrences in cites.items():
            if cite_key not in bib_keys:
                orphan_cites.append((cite_key, occurrences))
        if orphan_cites:
            for cite_key, occurrences in sorted(orphan_cites, key=lambda x: x[0]):
                print(f"\n   \\cite{{{cite_key}}} NO ENCONTRADO en .bib:")
                for fname, lineno, text in occurrences:
                    print(f"      {fname}:{lineno}")
        else:
            print("   [OK] Todas las citas existen en el archivo .bib.")
    else:
        print(f"\n[INFO] Archivo .bib no especificado o no encontrado: {bib_file}")

    # -------------------------------------------------------------------------
    # 6. Resumen
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(" RESUMEN EJECUTIVO")
    print("="*60)
    print(f"   Archivos .tex analizados : {len(tex_files)}")
    print(f"   Labels definidas         : {len(labels)}")
    print(f"   Referencias usadas       : {len(refs)}")
    print(f"   ?? explícitos            : {len(explicit_q)}")
    print(f"   Referencias huérfanas    : {len(orphan_refs)}")
    print(f"   Labels sin uso           : {len(unused_labels)}")
    if bib_file and Path(bib_file).exists():
        print(f"   Citas huérfanas          : {len(orphan_cites)}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python verificar_refs_tesis.py <ruta_capitulos> [ruta_bib]")
        print("Ejemplo:")
        print("  python verificar_refs_tesis.py /mnt/Compartida/Descargas_HDD/tesis-doctorado/tesis/Capitulos /mnt/Compartida/Descargas_HDD/tesis-doctorado/tesis/bibliografia_tesis_leal.bib")
        sys.exit(1)

    tex_dir = sys.argv[1]
    bib_file = sys.argv[2] if len(sys.argv) > 2 else None
    main(tex_dir, bib_file)
