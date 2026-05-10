# Tesis BSF — Compilación reproducible

Repositorio de la tesis doctoral **“Hybrid System for CO₂/CH₄ Prediction in *Hermetia illucens* Bioconversion”**.  
La compilación está soportada de dos maneras: localmente en Arch Linux y dentro de una imagen reproducible publicada en GitHub Container Registry [file:8].

## Requisitos

### Opción 1: compilación local en Arch Linux

Instala TeX Live y las dependencias necesarias:

```bash
sudo pacman -S texlive texlive-bibtexextra texlive-science texlive-latexextra texlive-plaingeneric make
```

Registra las fuentes OTF/TTF de TeX Live en `fontconfig` para que `fontspec` + XeLaTeX puedan verlas:

```bash
echo '<fontconfig><dir>/usr/share/texmf-dist/fonts/opentype</dir><dir>/usr/share/texmf-dist/fonts/truetype</dir></fontconfig>' \
  | sudo tee /etc/fonts/conf.d/09-texlive-fonts.conf
sudo fc-cache -fv
```

### Opción 2: compilación reproducible en contenedor

La imagen base del proyecto está definida en el `Dockerfile` y publicada como paquete en GitHub Container Registry [file:7].

Imagen publicada:

```text
ghcr.io/asoleal/tesis-bsf-base:latest
```

En Arch Linux se recomienda usar **Podman**:

```bash
sudo pacman -S podman
```

## Comandos de compilación

El `Makefile` centraliza toda la compilación de la tesis, la presentación y los artículos [file:8].

### Compilar localmente

```bash
make tesis
```

Otros targets útiles:

```bash
make presentacion
make candidatura
make articulos
make all
```

### Compilar en el contenedor

```bash
make docker-tesis
```

Otros targets en contenedor:

```bash
make docker-presentacion
make docker-candidatura
make docker-all
```

> En este proyecto, el contenedor se ejecuta sin red (`--network=none`) para evitar problemas del kernel local con `tun`/`veth` y porque la compilación LaTeX no necesita acceso a internet.

## Limpiar auxiliares

```bash
make clean
make clean-tesis
```

## Compilar un artículo específico

Para compilar un artículo dentro de `articulos/`:

```bash
make articulo dir=articulos/01-nombre-del-articulo
```

## Fuentes usadas

La tesis debe usar nombres de fuente compatibles tanto en local como en el contenedor:

```latex
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{Latin Modern Mono Light}
```

No usar `TeX Gyre TermesX`, porque puede no estar disponible de forma consistente entre Arch y la imagen del contenedor.

## Notas sobre ecuaciones y unidades

El proyecto usa `siunitx`. En ecuaciones matemáticas, no conviene escribir porcentajes con `\%` dentro del modo matemático si esto entra en conflicto con paquetes de unidades.

Ejemplo recomendado:

```latex
\begin{equation}
    \delta_{\text{lipid}} = \frac{L_{\text{DW}}}{X_{\text{DW}}} \times 100\,\si{\percent}
\end{equation}
```

Evitar expresiones como:

```latex
\times 100\;\%
```

porque pueden producir errores como `Incompatible glue units`.

## Imagen reproducible

La imagen del proyecto parte de `texlive/texlive:latest`, instala `make`, `fontconfig`, `locales`, `fonts-texgyre` y `fonts-lmodern`, y actualiza el caché de fuentes para que `fontspec` encuentre correctamente las familias tipográficas usadas por la tesis [file:7].

## Reconstruir y publicar la imagen

Si cambias el `Dockerfile`, puedes reconstruir y publicar la imagen así:

```bash
podman build -t tesis-bsf-base .
podman tag localhost/tesis-bsf-base ghcr.io/asoleal/tesis-bsf-base:latest
podman push ghcr.io/asoleal/tesis-bsf-base:latest
```

## Estructura relevante

```text
.
├── Dockerfile
├── Makefile
├── tesis/
│   └── tesis.tex
├── presentacion/
│   ├── presentacion.tex
│   └── presentacion-candidatura.tex
├── articulos/
└── archivos_bib/
```
