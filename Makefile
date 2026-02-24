# Variables de configuración
DOCKER_IMG  := tesis-bsf
CURRENT_DIR := $(shell pwd)
UID         := $(shell id -u)
GID         := $(shell id -g)


# Base Docker (sin -t para evitar errores de TTY en procesos automáticos)
DOCKER_BASE := docker run --rm -i --user $(UID):$(GID) \
    -v "$(CURRENT_DIR):/src" -w /src


# Colores (opcional)
YELLOW := \033[0;33m
NC     := \033[0m


# Permite recetas con prefijo '>' en vez de TAB (para conservar espacios)
.RECIPEPREFIX := >


.PHONY: all tesis slides articulos clean help watch-tesis watch-art


help:
>    @echo "Uso del Makefile:"
>    @echo "  make tesis         - Compila la tesis principal"
>    @echo "  make slides        - Compila las presentaciones"
>    @echo "  make articulos     - Compila todos los artículos en articulos/"
>    @echo "  make watch-tesis   - Auto-compilar al guardar (latexmk -pvc)"
>    @echo "  make watch-art DIR=nombre-carpeta - Watch de un artículo"
>    @echo "  make clean         - Elimina temporales"


all: tesis slides articulos


tesis:
>    @echo "$(YELLOW)Compilando tesis...$(NC)"
>    $(DOCKER_BASE) -w /src/tesis $(DOCKER_IMG) \
>        latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=. tesis.tex


slides:
>    @echo "$(YELLOW)Compilando presentación...$(NC)"
>    $(DOCKER_BASE) -w /src/presentacion $(DOCKER_IMG) \
>        latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=. presentacion.tex

articulos:
>	@echo "Compilando artículos uno por uno..."
>	@find articulos -name "main.tex" | while read -r main_file; do \
>		dir=$$(dirname "$$main_file"); \
>		echo "--------------------------------------------------"; \
>		echo "Compilando artículo en: $$dir"; \
>		echo "--------------------------------------------------"; \
>		(cd "$$dir" && latexmk -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error main.tex); \
>	done

solo-art:
>	@if [ -z "$(DIR)" ]; then \
>		echo "Error: Debes especificar la carpeta. Ejemplo: make solo-art DIR=00-red-neuronal"; \
>		exit 1; \
>	fi
>	@echo "Compilando únicamente: articulos/$(DIR)/"
>	@(cd articulos/$(DIR) && latexmk -pdfxe -interaction=nonstopmode -file-line-error -halt-on-error main.tex)

watch-tesis:
>    @echo "$(YELLOW)Watch tesis (Ctrl+C para detener).$(NC)"
>    $(DOCKER_BASE) -w /src/tesis $(DOCKER_IMG) \
>        latexmk -xelatex -pvc -view=none -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=. tesis.tex


watch-art:
>    @if [ -z "$(DIR)" ]; then \
>        echo "Error: usa DIR=. Ej: make watch-art DIR=04-hardware-dashboard"; \
>        exit 1; \
>    fi
>    @echo "$(YELLOW)Watch artículo $(DIR) (Ctrl+C para detener).$(NC)"
>    $(DOCKER_BASE) -w /src/articulos/$(DIR) $(DOCKER_IMG) \
>        latexmk -xelatex -pvc -view=none -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=. main.tex


clean:
>    @echo "Limpiando archivos temporales..."
>    find . -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o \
>        -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o \
>        -name "*.xdv" -o -name "*.bcf" -o -name "*.run.xml" -o -name "*.bbl" -o \
>        -name "*.blg" -o -name "*.snm" -o -name "*.nav" -o -name "*.vrb" \) -delete
>    @echo "Limpieza completada."
