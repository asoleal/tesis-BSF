#!/bin/bash
set -euo pipefail

OWNER="asoleal"
REPO="tesis-BSF"
PROJECT_NUM="1"
PROJECT_ID="PVT_kwHOA8nWbM4BXN46"
FIELD_START="PVTF_lAHOA8nWbM4BXN46zhSfDCw"
FIELD_END="PVTF_lAHOA8nWbM4BXN46zhSfDJY"
CSV_FILE="${1:-cronograma.tsv}"

if ! command -v gh &>/dev/null; then
    echo "❌ gh CLI no instalado"
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "❌ jq no instalado"
    exit 1
fi
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ No se encontró '$CSV_FILE'"
    exit 1
fi

crear_milestone_si_no_existe() {
    local title="$1"
    local desc="$2"
    local due="$3"
    local exists
    exists=$(gh api "repos/$OWNER/$REPO/milestones" --jq ".[] | select(.title == \"$title\") | .number" 2>/dev/null || true)
    if [ -n "$exists" ]; then
        echo "   ℹ️  Milestone '$title' ya existe (#$exists)"
        return
    fi
    echo "   🏗️  Creando milestone '$title'..."
    gh api "repos/$OWNER/$REPO/milestones" -f title="$title" -f description="$desc" -f due_on="${due}T23:59:59Z" -f state="open" >/dev/null 2>&1
    echo "   ✅ Milestone creado"
}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PREPARACIÓN: Creando milestones base si no existen"
echo "═══════════════════════════════════════════════════════════════"
echo ""

crear_milestone_si_no_existe "B1: Experimentos (Sem 5-12)"     "Recepción larvas, sustratos, experimentos T1/C1/T2/C2"              "2026-08-01"
crear_milestone_si_no_existe "B2a: Modelado (Sem 13-18)"       "Ajuste DEB, sensor virtual, PINN, validación"                         "2026-09-12"
crear_milestone_si_no_existe "B2b: Métricas (Sem 19-24)"       "Balance carbono, PCG, MRV, sensibilidad"                              "2026-10-10"
crear_milestone_si_no_existe "B3a: Redacción (Sem 25-30)"      "Capítulos 3, 4, 5 y anexos"                                           "2026-11-21"
crear_milestone_si_no_existe "B3b: Defensa (Sem 31-36)"        "Revisión, presentación, defensa"                                      "2026-12-26"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  CARGANDO TAREAS DESDE: $CSV_FILE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

COUNT=0
SUCCESS=0
FAILED=0

while IFS=$'\t' read -r title body milestone label fecha_inicio fecha_fin; do
    title="${title%$'\r'}"
    body="${body%$'\r'}"
    milestone="${milestone%$'\r'}"
    label="${label%$'\r'}"
    fecha_inicio="${fecha_inicio%$'\r'}"
    fecha_fin="${fecha_fin%$'\r'}"
    
    [ -z "$title" ] && continue
    
    COUNT=$((COUNT + 1))
    echo "[$COUNT] ⏳ $title"
    
    issue_url=$(gh issue create --repo "$OWNER/$REPO" --title "$title" --body "$body" --milestone "$milestone" --label "$label" 2>/dev/null) || {
        echo "      ❌ ERROR creando issue"
        FAILED=$((FAILED + 1))
        continue
    }
    echo "      📝 Issue: $issue_url"
    
    issue_num=$(echo "$issue_url" | grep -oE '[0-9]+$')
    
    gh project item-add "$PROJECT_NUM" --owner "$OWNER" --url "$issue_url" >/dev/null 2>&1 || {
        echo "      ❌ ERROR añadiendo al proyecto"
        FAILED=$((FAILED + 1))
        continue
    }
    
    sleep 1
    item_id=$(gh project item-list "$PROJECT_NUM" --owner "$OWNER" --format json 2>/dev/null | jq -r ".items[] | select(.content.number == $issue_num) | .id") || true
    
    if [ -z "$item_id" ]; then
        echo "      ⚠️  Reintentando obtener item ID..."
        sleep 2
        item_id=$(gh project item-list "$PROJECT_NUM" --owner "$OWNER" --format json 2>/dev/null | jq -r ".items[] | select(.content.number == $issue_num) | .id") || true
    fi
    
    if [ -z "$item_id" ]; then
        echo "      ❌ ERROR: No se pudo obtener item ID para issue #$issue_num"
        FAILED=$((FAILED + 1))
        continue
    fi
    echo "      🔗 Item ID: $item_id"
    
    gh project item-edit --project-id "$PROJECT_ID" --id "$item_id" --field-id "$FIELD_START" --date "$fecha_inicio" >/dev/null 2>&1 || {
        echo "      ❌ ERROR asignando fecha inicio"
        FAILED=$((FAILED + 1))
        continue
    }
    echo "      📅 Inicio: $fecha_inicio"
    
    gh project item-edit --project-id "$PROJECT_ID" --id "$item_id" --field-id "$FIELD_END" --date "$fecha_fin" >/dev/null 2>&1 || {
        echo "      ❌ ERROR asignando fecha fin"
        FAILED=$((FAILED + 1))
        continue
    }
    echo "      📅 Fin: $fecha_fin"
    echo "      ✅ Completado"
    SUCCESS=$((SUCCESS + 1))
    echo ""
done < <(tail -n +2 "$CSV_FILE")

echo "═══════════════════════════════════════════════════════════════"
echo "  RESUMEN"
echo "═══════════════════════════════════════════════════════════════"
echo "  Procesadas: $COUNT"
echo "  Exitosas:   $SUCCESS"
echo "  Fallidas:   $FAILED"
echo ""
echo "  Ver tu cronograma en:"
echo "  https://github.com/users/$OWNER/projects/$PROJECT_NUM"
echo "═══════════════════════════════════════════════════════════════"
