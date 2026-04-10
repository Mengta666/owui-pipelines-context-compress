#!/usr/bin/env bash
set -Eeuo pipefail

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

log() {
  printf '%s [chat-cache-gc] %s\n' "$(timestamp)" "$*"
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

WORKSPACE_ROOT="${MINI_AGENT_WORKSPACE_ROOT:-/app/pipelines/agent_workspace}"
SESSION_ROOT="${CHAT_CACHE_GC_SESSION_ROOT:-${WORKSPACE_ROOT%/}/sessions}"
RUN_ONCE="${CHAT_CACHE_GC_RUN_ONCE:-false}"
INTERVAL_SEC="${CHAT_CACHE_GC_INTERVAL_SEC:-1800}"
MIN_AGE_SEC="${CHAT_CACHE_GC_MIN_AGE_SEC:-600}"
REQUEST_TIMEOUT_SEC="${CHAT_CACHE_GC_REQUEST_TIMEOUT_SEC:-10}"
DRY_RUN="${CHAT_CACHE_GC_DRY_RUN:-false}"

OPENWEBUI_API_BASE_URL="${OPENWEBUI_API_BASE_URL:-}"
OPENWEBUI_CHAT_CHECK_URL_TEMPLATE="${OPENWEBUI_CHAT_CHECK_URL_TEMPLATE:-}"
OPENWEBUI_API_TOKEN="${OPENWEBUI_API_TOKEN:-}"
OPENWEBUI_API_AUTH_HEADER="${OPENWEBUI_API_AUTH_HEADER:-Authorization}"
OPENWEBUI_API_TOKEN_SCHEME="${OPENWEBUI_API_TOKEN_SCHEME:-Bearer}"

if [[ -z "$OPENWEBUI_CHAT_CHECK_URL_TEMPLATE" ]]; then
  if [[ -z "$OPENWEBUI_API_BASE_URL" ]]; then
    log "missing OPENWEBUI_API_BASE_URL or OPENWEBUI_CHAT_CHECK_URL_TEMPLATE"
    exit 1
  fi
  OPENWEBUI_CHAT_CHECK_URL_TEMPLATE="${OPENWEBUI_API_BASE_URL%/}/api/v1/chats/{chat_id}"
fi

resolve_chat_url() {
  local chat_id="$1"
  printf '%s' "${OPENWEBUI_CHAT_CHECK_URL_TEMPLATE//\{chat_id\}/$chat_id}"
}

extract_last_active() {
  local state_path="$1"
  if [[ ! -f "$state_path" ]]; then
    return 0
  fi
  grep -oE '"last_active_at"[[:space:]]*:[[:space:]]*[0-9]+' "$state_path" \
    | grep -oE '[0-9]+' \
    | tail -n 1 \
    || true
}

file_mtime() {
  local target="$1"
  stat -c %Y "$target" 2>/dev/null || date +%s
}

is_old_enough() {
  local session_dir="$1"
  local state_path="$session_dir/state.json"
  local now_ts
  now_ts="$(date +%s)"

  local last_active
  last_active="$(extract_last_active "$state_path")"
  if [[ -z "$last_active" ]]; then
    last_active="$(file_mtime "$session_dir")"
  fi

  local age_sec=$(( now_ts - last_active ))
  if (( age_sec < MIN_AGE_SEC )); then
    log "skip recent session chat_id=$(basename "$session_dir") age_sec=$age_sec min_age_sec=$MIN_AGE_SEC"
    return 1
  fi
  return 0
}

check_chat_status() {
  local chat_id="$1"
  local url
  url="$(resolve_chat_url "$chat_id")"

  local tmp
  tmp="$(mktemp)"
  local http_code="000"
  local -a curl_args=(
    --silent
    --show-error
    --location
    --output "$tmp"
    --write-out '%{http_code}'
    --max-time "$REQUEST_TIMEOUT_SEC"
    -H 'Accept: application/json'
  )

  if [[ -n "$OPENWEBUI_API_TOKEN" ]]; then
    curl_args+=(
      -H "${OPENWEBUI_API_AUTH_HEADER}: ${OPENWEBUI_API_TOKEN_SCHEME} ${OPENWEBUI_API_TOKEN}"
    )
  fi

  if ! http_code="$(curl "${curl_args[@]}" "$url")"; then
    local body
    body="$(head -c 200 "$tmp" 2>/dev/null | tr '\n' ' ' || true)"
    rm -f "$tmp"
    log "request failed chat_id=$chat_id url=$url body=${body:-<empty>}"
    return 2
  fi

  local body
  body="$(head -c 200 "$tmp" 2>/dev/null | tr '\n' ' ' || true)"
  rm -f "$tmp"

  # Match middleware behavior:
  # - 200 => exists
  # - 404 => missing
  # - 401/403 => missing only if detail indicates not found
  # - others => unknown/error (do not delete)
  local detail
  detail="$(_extract_response_detail "$body" "$http_code")"

  case "$http_code" in
    200)
      return 0
      ;;
    404)
      return 1
      ;;
    401|403)
      if _detail_indicates_missing "$detail"; then
        log "treat as missing by detail chat_id=$chat_id http_code=$http_code detail=$detail url=$url"
        return 1
      fi
      log "auth/forbidden but not missing chat_id=$chat_id http_code=$http_code detail=$detail url=$url body=${body:-<empty>}"
      return 2
      ;;
    *)
      log "unexpected response chat_id=$chat_id http_code=$http_code detail=$detail url=$url body=${body:-<empty>}"
      return 2
      ;;
  esac
}

_extract_response_detail() {
  local body="$1"
  local http_code="$2"
  if [[ "$http_code" == "200" ]]; then
    printf 'ok'
    return 0
  fi

  # Lightweight detail extraction without jq.
  local detail
  detail="$(printf '%s' "$body" | sed -nE 's/.*"detail"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' | head -n 1 | tr '[:upper:]' '[:lower:]')"
  if [[ -n "$detail" ]]; then
    printf '%s' "$detail"
  else
    printf '%s' "$http_code"
  fi
}

_detail_indicates_missing() {
  local detail="${1,,}"
  [[ "$detail" == *"could not find"* || "$detail" == *"not found"* ]]
}

delete_session_dir() {
  local session_dir="$1"
  local chat_id
  chat_id="$(basename "$session_dir")"

  if is_true "$DRY_RUN"; then
    log "dry-run delete chat_id=$chat_id dir=$session_dir"
    return 0
  fi

  rm -rf -- "$session_dir"
  log "deleted chat_id=$chat_id dir=$session_dir"
}

scan_once() {
  if [[ ! -d "$SESSION_ROOT" ]]; then
    log "session root does not exist: $SESSION_ROOT"
    return 0
  fi

  local scanned=0
  local kept=0
  local deleted=0
  local skipped=0
  local errors=0

  shopt -s nullglob
  for session_dir in "$SESSION_ROOT"/*; do
    [[ -d "$session_dir" ]] || continue
    scanned=$(( scanned + 1 ))

    local chat_id
    chat_id="$(basename "$session_dir")"

    if ! is_old_enough "$session_dir"; then
      skipped=$(( skipped + 1 ))
      continue
    fi

    if check_chat_status "$chat_id"; then
      kept=$(( kept + 1 ))
      log "keep chat_id=$chat_id"
      continue
    else
      local rc=$?
      if (( rc == 1 )); then
        delete_session_dir "$session_dir"
        deleted=$(( deleted + 1 ))
      else
        errors=$(( errors + 1 ))
      fi
    fi
  done

  log "scan complete session_root=$SESSION_ROOT scanned=$scanned kept=$kept deleted=$deleted skipped=$skipped errors=$errors"
}

log "started session_root=$SESSION_ROOT interval_sec=$INTERVAL_SEC min_age_sec=$MIN_AGE_SEC dry_run=$DRY_RUN url_template=$OPENWEBUI_CHAT_CHECK_URL_TEMPLATE"

while true; do
  scan_once
  if is_true "$RUN_ONCE"; then
    break
  fi
  sleep "$INTERVAL_SEC"
done
