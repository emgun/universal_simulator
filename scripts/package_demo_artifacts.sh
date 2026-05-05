#!/usr/bin/env bash
set -euo pipefail

# Package remote demo artifacts before tearing down a data-prep or experiment
# instance. Missing paths are recorded but do not fail the packaging step.

OUTPUT=${OUTPUT:-reports/demo/demo_artifacts.tar.gz}
MANIFEST=${MANIFEST:-${OUTPUT%.tar.gz}.manifest.txt}

default_paths=(
  "docs/demo_completion_audit.md"
  "docs/demo_runbook.md"
  "docs/demo_smoke_data_manifest.yaml"
  "docs/demo_data_manifest.yaml"
  "reports/demo/remote_smoke_pipeline"
  "reports/demo/latest"
  "reports/light_experiments_remote"
)

paths=("$@")
if [ "${#paths[@]}" -eq 0 ]; then
  paths=("${default_paths[@]}")
fi

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$MANIFEST")"

include_file="$(mktemp)"
trap 'rm -f "$include_file"' EXIT

{
  echo "# UPS demo artifact package"
  echo "output: ${OUTPUT}"
  echo "created_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "## Included"
} > "$MANIFEST"

for path in "${paths[@]}"; do
  if [ -e "$path" ]; then
    printf '%s\n' "$path" >> "$include_file"
    echo "- ${path}" >> "$MANIFEST"
  fi
done

{
  echo
  echo "## Missing"
} >> "$MANIFEST"

for path in "${paths[@]}"; do
  if [ ! -e "$path" ]; then
    echo "- ${path}" >> "$MANIFEST"
  fi
done

if [ ! -s "$include_file" ]; then
  echo "No artifact paths exist; wrote manifest only: ${MANIFEST}" >&2
  exit 1
fi

tar -czf "$OUTPUT" -T "$include_file"
echo "$OUTPUT"
echo "$MANIFEST"
