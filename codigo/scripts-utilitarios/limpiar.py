import os

def remove_zwsp(directory):
    count = 0
    # Recorre todas las subcarpetas (contenido, preambulo, etc.)
    for root, _, files in os.walk(directory):
        for file in files:
            # Solo revisamos archivos de texto relevantes
            if file.endswith(".tex") or file.endswith(".bib") or file.endswith(".sty"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # El carácter culpable es \u200b
                    if '\u200b' in content:
                        print(f"¡ENCONTRADO! Limpiando carácter invisible en: {filepath}")
                        new_content = content.replace('\u200b', '')
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                except Exception as e:
                    print(f"Error leyendo {filepath}: {e}")

    if count == 0:
        print("No se encontraron caracteres invisibles (U+200B).")
    else:
        print(f"--> ¡Listo! Se limpiaron {count} archivos.")

if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"Escaneando carpeta: {current_dir}")
    remove_zwsp(current_dir)
