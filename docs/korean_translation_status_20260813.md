# 연구 문서 한국어화 범위 — 2026-08-13

## 처리 원칙

- 현재 연구자가 읽는 활성 Markdown 설명서와 연구 기록은 한국어를 기본 언어로 한다.
- Model ID, class/function/file/config 이름, command option, metric 이름, artifact path,
  protocol ID, SHA256은 재현성을 위해 원문 표기를 유지한다.
- 표 안의 짧은 architecture label과 공식 library 이름도 번역하지 않을 수 있다.
- PDF, JSON result, code snapshot, checkpoint, gzip prediction은 실행 증거이므로 번역하거나
  수정하지 않는다.
- 번역 전 Markdown 원본은
  `archive/pre_korean_translation_20260813.zip`에 보존했다.

## 한국어로 교정한 핵심 문서

- `README.md`, `RESEARCH_GUIDE.md`.
- `docs/research_log.md`.
- `docs/revised_research_roadmap_v3.md`.
- `docs/phase0_baseline.md`, `phase1_libero_state.md`, `phase2_graph_spec.md`.
- `docs/phase3_offline_probe.md`.
- `docs/phase3_reduced_crossfold_gate_result.md`.
- `docs/phase3_corrected_protocol_v2.md`.
- `docs/phase3_corrected_threefold_seed0_result.md`.
- `docs/phase3_action_semantics_gate_result.md`.
- `docs/phase3_holder_action_v2_smoke_result.md`.
- `docs/phase3_topology_action_followup_result.md`.
- `docs/phase3_pair_local_temporal_smoke_result.md`.
- `docs/phase3_pair_local_temporal_threefold_seed0_result.md`.
- `docs/phase3_weak_label_audit_v2.md`.
- `docs/colab_runtime_persistence.md`, repository/archive audit 문서.
- Pair-local 계획서와 공식 phase별 notebook 설명서.

기존에 이미 한국어 중심이던 계획·설계·분석 문서는 내용과 수치를 유지하고 영문 기술
용어만 필요한 범위에서 남겼다. 모든 활성 Markdown 파일은 한국어 문맥을 포함하며
영문만으로 된 설명 문서는 남기지 않았다.

## Phase 이름 교정

임시 순번형 notebook의 `phase_01`~`phase_05`는 공식 연구 단계처럼 오해될 수 있어
사용하지 않는다. 공식 순서는 Phase 0 → Phase 1A → Phase 2A → Phase 2R → Phase 2D →
Phase 3A → Phase 3B다. Phase 3C와 Phase 4 이후는 gate를 통과할 때까지 미작성 상태다.
