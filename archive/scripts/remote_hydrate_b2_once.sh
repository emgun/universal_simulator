#!/usr/bin/env bash
set -e

cd /workspace/universal_simulator || exit 1

# Normalize .env line endings if present
[ -f .env ] && sed -i 's/\r$//' .env || true

# Parse B2 vars from .env without sourcing
awk_val() {
  awk -F= -v key="$1" 'BEGIN{found=0} $0 ~ "^"key"=" { val=$2; sub(/^"\"|\"$/,"",val); print val; found=1 } END{ if(found==0) print "" }' .env 2>/dev/null || true
}

B2_KEY_ID_VAL=$(awk_val B2_KEY_ID)
B2_APP_KEY_VAL=$(awk_val B2_APP_KEY)
B2_BUCKET_VAL=$(awk_val B2_BUCKET)
B2_PREFIX_VAL=$(awk_val B2_PREFIX)
B2_S3_ENDPOINT_VAL=$(awk_val B2_S3_ENDPOINT)
B2_S3_REGION_VAL=$(awk_val B2_S3_REGION)

export B2_KEY_ID="$B2_KEY_ID_VAL"
export B2_APP_KEY="$B2_APP_KEY_VAL"
export B2_BUCKET="$B2_BUCKET_VAL"
export B2_PREFIX="$B2_PREFIX_VAL"
export B2_S3_ENDPOINT="$B2_S3_ENDPOINT_VAL"
export B2_S3_REGION="$B2_S3_REGION_VAL"

echo "B2_BUCKET=${B2_BUCKET:-unset} B2_KEY_ID=${B2_KEY_ID:+set} B2_APP_KEY=${B2_APP_KEY:+set}"

mkdir -p data/pdebench
CLEAN_OLD_SPLITS=1 bash scripts/fetch_datasets_b2.sh burgers1d_full_v1

# Create expected links
ln -sf burgers1d_full_v1/burgers1d_train.h5 data/pdebench/burgers1d_train.h5 || true
ln -sf burgers1d_full_v1/burgers1d_val.h5   data/pdebench/burgers1d_val.h5   || true
ln -sf burgers1d_full_v1/burgers1d_test.h5  data/pdebench/burgers1d_test.h5  || true

echo "--- hydrated listing"
ls -lh data/pdebench/burgers1d_full_v1 || true
ls -lh data/pdebench || true






