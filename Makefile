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


# --- Regla corregida para evitar el error .tex.tex ---
articulo:
	@if [ -z "$(dir)" ]; then \
		echo "Error: Indica la carpeta. Ejemplo: make articulo dir=articulos/01-nombre"; \
		exit 1; \
	fi
	@echo "==> Detectando archivo en $(dir)..."
	@FILE=$$(ls $(dir)/*.tex | head -n 1); \
	FILENAME=$$(basename $$FILE .tex); \
	echo "==> Compilando $$FILENAME.tex..."; \
	(cd $(dir) && $(LATEXMK) $$FILENAME)
        
# Limpiar archivos temporales de LaTeX
clean:
	@echo "==> Limpiando archivos auxiliares..."
	$(LATEXMK) -C
	find articulos -name "main.tex" -execdir latexmk -C \;
	rm -rf _minted* *.nav *.snm *.vrb