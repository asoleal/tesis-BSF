# --- Variables ---
# Nombres de archivos principales
TESIS_MAIN = tesis/tesis.tex
PRES_MAIN  = presentacion/presentacion.tex
# Comando base de latexmk
LATEXMK = latexmk -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error

# --- Reglas Principales ---

.PHONY: all tesis presentacion articulos clean

# Por defecto, si escribes solo 'make', compila todo
all: tesis presentacion articulos

# Compilar la Tesis
tesis: $(TESIS_MAIN)
	@echo "==> Compilando Tesis..."
	(cd tesis && $(LATEXMK) tesis.tex)

# Compilar la Presentación (Beamer)
presentacion: $(PRES_MAIN)
	@echo "==> Compilando Presentación..."
	(cd presentacion && $(LATEXMK) presentacion.tex)

# Compilar todos los artículos encontrados en las subcarpetas
articulos:
	@echo "==> Compilando artículos uno por uno..."
	@find articulos -name "main.tex" | while read -r main_file; do \
		dir=$$(dirname "$$main_file"); \
		echo "--------------------------------------------------"; \
		echo "Procesando: $$dir"; \
		echo "--------------------------------------------------"; \
		(cd "$$dir" && $(LATEXMK) main.tex); \
	done

# Compilar un artículo específico
# Uso: make articulo dir=articulos/nombre-del-articulo
articulo:
	@if [ -z "$(dir)" ]; then \
		echo "Error: Indica la carpeta. Ejemplo: make articulo dir=articulos/articulo-01"; \
		exit 1; \
	fi
	@echo "==> Compilando artículo individual en: $(dir)..."
	(cd $(dir) && $(LATEXMK) main.tex)
    
# Limpiar archivos temporales de LaTeX
clean:
	@echo "==> Limpiando archivos auxiliares..."
	$(LATEXMK) -C
	find articulos -name "main.tex" -execdir latexmk -C \;
	rm -rf _minted* *.nav *.snm *.vrb