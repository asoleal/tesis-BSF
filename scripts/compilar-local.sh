#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/asoleal/tesis-bsf-base:sha-7c75eab"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
  cat <<'UsoEOF'
Uso:
  scripts/compilar-local.sh shell
  scripts/compilar-local.sh tesis
  scripts/compilar-local.sh presentacion
  scripts/compilar-local.sh candidatura
  scripts/compilar-local.sh articulo <ruta>
  scripts/compilar-local.sh <ruta-articulo>
  scripts/compilar-local.sh clean

Ejemplos:
  scripts/compilar-local.sh shell
  scripts/compilar-local.sh tesis
  scripts/compilar-local.sh presentacion
  scripts/compilar-local.sh candidatura
  scripts/compilar-local.sh articulo articulos/01-pinn-biomasa
  scripts/compilar-local.sh articulos/01-pinn-biomasa
  scripts/compilar-local.sh clean
UsoEOF
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
  candidatura)
    run_in_container "make candidatura"
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
