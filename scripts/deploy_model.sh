#!/usr/bin/env bash
# Deploy the MLflow champion model to VM 2 when its version changes.
#
#   deploy_model.sh                 check once and deploy if needed
#   deploy_model.sh --install-cron  run that check every 5 minutes
#
# Keep this script next to compose.vm2.yml and a .env file with IMAGE_TAG and
# MLFLOW_URL. A model that fails its health check is rolled back and not retried.
set -euo pipefail

DEPLOY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT="$DEPLOY_DIR/$(basename "${BASH_SOURCE[0]}")"
COMPOSE_FILE=compose.vm2.yml
MODEL_DIR=${MODEL_DIR:-/opt/flight-delay/models}
REGISTERED_MODEL=${REGISTERED_MODEL:-flight-delay-severity}
HEALTH_URL=${HEALTH_URL:-http://localhost:8000/health}
HEALTH_RETRIES=${HEALTH_RETRIES:-30}
HEALTH_DELAY=${HEALTH_DELAY:-2}

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

install_cron() {
    local line="*/5 * * * * bash $SCRIPT >> $DEPLOY_DIR/deploy_model.log 2>&1"
    { crontab -l 2>/dev/null | grep -vF "$SCRIPT" || true; echo "$line"; } | crontab -
    log "Installed: $line"
}

# The container reads MODEL_VERSION from .env through Compose.
start_api() {
    { grep -v '^MODEL_VERSION=' .env || true; echo "MODEL_VERSION=$1"; } > .env.tmp
    mv .env.tmp .env
    docker compose -f "$COMPOSE_FILE" up -d --force-recreate
}

is_healthy() {
    local attempt reported
    for ((attempt = 1; attempt <= HEALTH_RETRIES; attempt++)); do
        reported=$(curl -fsS "$HEALTH_URL" 2>/dev/null | jq -r .model.version || true)
        [[ "$reported" == "$1" ]] && return 0
        sleep "$HEALTH_DELAY"
    done
    return 1
}

deploy() {
    cd "$DEPLOY_DIR"
    exec 9> .deploy.lock
    flock -n 9 || exit 0

    # shellcheck disable=SC1091
    source .env
    : "${MLFLOW_URL:?Set MLFLOW_URL in .env}"
    : "${IMAGE_TAG:?Set IMAGE_TAG in .env}"

    local info version run_id installed failed
    local model="$MODEL_DIR/model.joblib"
    info=$(curl -fsS -G "$MLFLOW_URL/api/2.0/mlflow/registered-models/alias" \
        --data-urlencode "name=$REGISTERED_MODEL" --data-urlencode "alias=champion")
    version=$(jq -r .model_version.version <<< "$info")
    run_id=$(jq -r .model_version.run_id <<< "$info")
    installed=$(cat "$MODEL_DIR/VERSION" 2> /dev/null || true)
    failed=$(cat "$MODEL_DIR/FAILED_VERSION" 2> /dev/null || true)
    if [[ "$version" == "$installed" || "$version" == "$failed" ]]; then
        exit 0
    fi

    log "Deploying champion version $version (run $run_id)"
    curl -fsS -o "$model.tmp" "$MLFLOW_URL/get-artifact?path=pipeline.joblib&run_uuid=$run_id"
    chmod 644 "$model.tmp"
    if [[ -f "$model" ]]; then
        cp -p "$model" "$model.previous"
    fi
    mv "$model.tmp" "$model"

    if start_api "$version" && is_healthy "$version"; then
        echo "$version" > "$MODEL_DIR/VERSION"
        rm -f "$MODEL_DIR/FAILED_VERSION"
        log "Deployed champion version $version"
        exit 0
    fi

    echo "$version" > "$MODEL_DIR/FAILED_VERSION"
    if [[ -n "$installed" && -f "$model.previous" ]]; then
        mv "$model.previous" "$model"
        start_api "$installed"
        log "Version $version failed; rolled back to version $installed"
    else
        log "Version $version failed; no previous model to restore"
    fi
    exit 1
}

case "${1:-}" in
    --install-cron) install_cron ;;
    "") deploy ;;
    *) echo "Usage: $0 [--install-cron]" >&2; exit 2 ;;
esac
