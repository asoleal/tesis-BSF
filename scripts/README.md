# tesis-BSF

Entorno reproducible para compilar la tesis, la presentación y los artículos usando Docker y una imagen fija publicada en GHCR.

## Imagen usada

Este proyecto compila localmente y en CI con la imagen:

```bash
ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab
```

## Requisitos

- Docker instalado
- Acceso a GHCR
- Repositorio clonado localmente

## Descargar la imagen

Si la imagen es privada, primero inicia sesión en GHCR con un PAT de GitHub con permiso `read:packages`:

```bash
echo TU_GITHUB_PAT | docker login ghcr.io -u asoleal --password-stdin
```

## Reconstruir la imagen localmente (Respaldo)

Si la imagen ya no está disponible en GitHub Container Registry o necesitas reconstruirla desde cero localmente, debes usar la red del host (`--network host`) durante la construcción. 

Esto se debe a un problema del host con la creación de interfaces `veth` en la red `bridge` por defecto de Docker, lo que impide que el contenedor temporal tenga internet para descargar paquetes como TeX Live.

Para reconstruirla con el mismo tag, ejecuta en la raíz del proyecto:

```bash
docker build --network host -t ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab .



Luego descarga la imagen:

```bash
docker pull ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab
```

## Compilación local manual

Debido a un problema de red `bridge` en este equipo, la ejecución local usa `--network none`. Para compilar no hace falta acceso a internet.

Entrar al contenedor:

```bash
docker run --rm -it \
  --network none \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd):/work" \
  -w /work \
  ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab \
  bash
```

Ya dentro del contenedor:

```bash
make tesis
make presentacion
make articulo dir=articulos/01-pinn-biomasa
```

## Script de ayuda

Se recomienda usar el script:

```bash
scripts/compilar-local.sh
```

Dar permisos una vez:

```bash
chmod +x scripts/compilar-local.sh
```

## Uso del script

Abrir shell dentro del contenedor:

```bash
scripts/compilar-local.sh shell
```

Compilar tesis:

```bash
scripts/compilar-local.sh tesis
```

Compilar presentación:

```bash
scripts/compilar-local.sh presentacion
```

Compilar artículo:

```bash
scripts/compilar-local.sh articulo articulos/01-pinn-biomasa
```

Si tu script acepta ruta directa, también puede usarse así:

```bash
scripts/compilar-local.sh articulos/01-pinn-biomasa
```

Limpiar archivos auxiliares:

```bash
scripts/compilar-local.sh clean
```

## Script sugerido

Contenido recomendado para `scripts/compilar-local.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE}")/.." && pwd)"

run_in_container() {
  docker run --rm -it \
    --network none \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,src="${ROOT_DIR}",target=/work \
    -w /work \
    "${IMAGE}" \
    bash -lc "$1"
}

usage() {
  cat <<'EOH'
Uso:
  scripts/compilar-local.sh shell
  scripts/compilar-local.sh tesis
  scripts/compilar-local.sh presentacion
  scripts/compilar-local.sh articulo <ruta>
  scripts/compilar-local.sh <ruta-articulo>
  scripts/compilar-local.sh clean

Ejemplos:
  scripts/compilar-local.sh shell
  scripts/compilar-local.sh tesis
  scripts/compilar-local.sh presentacion
  scripts/compilar-local.sh articulo articulos/01-pinn-biomasa
  scripts/compilar-local.sh articulos/01-pinn-biomasa
  scripts/compilar-local.sh clean
EOH
}

cmd="${1:-}"

case "${cmd}" in
  shell)
    docker run --rm -it \
      --network none \
      --user "$(id -u):$(id -g)" \
      --mount type=bind,src="${ROOT_DIR}",target=/work \
      -w /work \
      "${IMAGE}" \
      bash
    ;;
  tesis)
    run_in_container "make tesis"
    ;;
  presentacion)
    run_in_container "make presentacion"
    ;;
  articulo)
    ART_DIR="${2:-}"
    if [[ -z "${ART_DIR}" ]]; then
      echo "Error: debes indicar la ruta del artículo."
      echo "Ejemplo: scripts/compilar-local.sh articulo articulos/01-pinn-biomasa"
      exit 1
    fi
    run_in_container "make articulo dir='${ART_DIR}'"
    ;;
  clean)
    run_in_container "make clean"
    ;;
  articulos/*)
    run_in_container "make articulo dir='${cmd}'"
    ;;
  *)
    usage
    exit 1
    ;;
esac
```

## Nota sobre red de Docker

En este equipo, Docker falla con la red `bridge` por un problema del host relacionado con interfaces `veth`, por eso se usa `--network none` como solución práctica para compilar localmente. Eso no afecta la compilación de LaTeX porque el proceso no necesita red.

## Flujo recomendado

1. Editar archivos `.tex`, `.bib` o figuras.
2. Ejecutar compilación local con el script.
3. Verificar PDFs generados.
4. Hacer commit y push.
5. Dejar que GitHub Actions compile con la misma imagen base.

