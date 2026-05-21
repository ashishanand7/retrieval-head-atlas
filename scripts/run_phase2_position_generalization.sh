#!/usr/bin/env bash
set -euo pipefail

NEEDLE_FRACS="${NEEDLE_FRACS:-0.5 0.9}"
N_PER_VARIANT="${N_PER_VARIANT:-8}"
TARGET_TOKENS="${TARGET_TOKENS:-8192}"
OUT_DIR="${OUT_DIR:-artifacts_phase2}"

FUNCTIONAL_GROUPS="answer_address,non_address_core,strong13,core19,first_token_sink,query_tail,answer_address_inactive_control,query_tail_inactive_control"
SINGLE_HEAD_GROUPS="address_L22H7,address_L21H11,address_L22H10,support_L20H7,support_L18H3,support_L17H10,support_L20H8,support_L22H0,support_L22H4,activation_control_L20H5,inactive_control_L21H2,inactive_control_L22H8,inactive_control_L22H3"

for NEEDLE_FRAC in ${NEEDLE_FRACS}; do
  LABEL="${NEEDLE_FRAC/./p}"
  SUFFIX="${TARGET_TOKENS}_n${N_PER_VARIANT}_frac${LABEL}"

  echo "== needle_frac=${NEEDLE_FRAC}: functional group ablation =="
  python scripts/run_semantic_group_ablation.py \
    --groups-csv configs/semantic_functional_groups.csv \
    --groups "${FUNCTIONAL_GROUPS}" \
    --n-per-variant "${N_PER_VARIANT}" \
    --target-tokens "${TARGET_TOKENS}" \
    --needle-frac "${NEEDLE_FRAC}" \
    --intervention-scope query \
    --out "${OUT_DIR}/semantic_group_ablation_functional_controls_${SUFFIX}_query.csv" \
    --summary-out "${OUT_DIR}/semantic_group_ablation_functional_controls_${SUFFIX}_query_summary.json"

  echo "== needle_frac=${NEEDLE_FRAC}: functional component patching =="
  python scripts/run_semantic_component_patching.py \
    --groups-csv configs/semantic_functional_groups.csv \
    --groups "${FUNCTIONAL_GROUPS}" \
    --n-per-variant "${N_PER_VARIANT}" \
    --target-tokens "${TARGET_TOKENS}" \
    --needle-frac "${NEEDLE_FRAC}" \
    --out "${OUT_DIR}/semantic_component_patching_functional_${SUFFIX}.csv" \
    --summary-out "${OUT_DIR}/semantic_component_patching_functional_${SUFFIX}_summary.json"

  echo "== needle_frac=${NEEDLE_FRAC}: single-head patching =="
  python scripts/run_semantic_component_patching.py \
    --groups-csv configs/semantic_single_head_patch_groups.csv \
    --groups "${SINGLE_HEAD_GROUPS}" \
    --n-per-variant "${N_PER_VARIANT}" \
    --target-tokens "${TARGET_TOKENS}" \
    --needle-frac "${NEEDLE_FRAC}" \
    --out "${OUT_DIR}/semantic_single_head_patching_${SUFFIX}.csv" \
    --summary-out "${OUT_DIR}/semantic_single_head_patching_${SUFFIX}_summary.json"

  echo "== needle_frac=${NEEDLE_FRAC}: attention trace =="
  python scripts/run_semantic_attention_trace.py \
    --heads-csv configs/semantic_core_heads.csv \
    --n-heads 19 \
    --n-per-variant "${N_PER_VARIANT}" \
    --target-tokens "${TARGET_TOKENS}" \
    --needle-frac "${NEEDLE_FRAC}" \
    --out "${OUT_DIR}/semantic_attention_trace_core19_${SUFFIX}.csv" \
    --summary-out "${OUT_DIR}/semantic_attention_trace_core19_${SUFFIX}_summary.json"

  echo "== needle_frac=${NEEDLE_FRAC}: activation delta =="
  python scripts/run_semantic_activation_delta.py \
    --groups-csv configs/semantic_functional_groups.csv \
    --groups "${FUNCTIONAL_GROUPS}" \
    --n-per-variant "${N_PER_VARIANT}" \
    --target-tokens "${TARGET_TOKENS}" \
    --needle-frac "${NEEDLE_FRAC}" \
    --include-head-rows \
    --out "${OUT_DIR}/semantic_activation_delta_functional_${SUFFIX}.csv" \
    --summary-out "${OUT_DIR}/semantic_activation_delta_functional_${SUFFIX}_summary.json"
done
