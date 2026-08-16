# Graph-CLaD 문서 색인

문서가 생성 시점과 역할에 따라 여러 파일로 나뉘어 있으므로, 다음 우선순위를
canonical navigation으로 사용한다. 과거 결과 문서는 삭제하거나 합쳐 쓰지 않고 근거
자료로 보존한다.

## 가장 먼저 읽을 문서

| 순서 | 문서 | 역할 |
|---:|---|---|
| 1 | `CURRENT_STATUS.md` | 현재 실행 상태, 서버 경로, 다음 gate와 즉시 할 일 |
| 2 | `01-plan/features/graph-clad-integrated-research-v4.plan.md` | 새 canonical 연구계획서: 질문, 비교, gate, 성공·중단 기준 |
| 3 | `RESEARCH_WORKFLOW_FOR_BEGINNERS.md` | 연구 질문, 데이터, Phase, 지표를 처음부터 설명 |
| 4 | `CODEBASE_GUIDE_FOR_BEGINNERS.md` | 폴더·Python 파일·입출력·실행 흐름을 상세 설명 |
| 5 | `NEXT_SESSION_PROMPT.md` | 새 Codex 세션에 그대로 붙여 넣는 인계 프롬프트 |
| 6 | `../RESEARCH_GUIDE.md` | 운영 중심의 전체 폴더와 실행 방법 설명 |
| 7 | `research_log.md` | 날짜순 실행·판단 기록 |
| 8 | `revised_research_roadmap_v3.md` | v4 이전 Phase 개정 근거와 이전 gate 보존 |

현재 상태가 다른 문서와 충돌하면 먼저 `CURRENT_STATUS.md`와 최신
`research_log.md`를 확인한다. 향후 연구 판단은 v4 계획서의 gate를 따르고, 결정은
계획서·현재 상태·research log에 함께 반영한다.

## 현재 Phase 3B 핵심 문서

| 문서 | 내용 |
|---|---|
| `phase3_corrected_protocol_v2.md` | natural validation, frozen threshold, metric 계약 |
| `phase3_pair_local_temporal_threefold_seed0_result.md` | H0–H3 12-run 결과와 gate |
| `phase3_weak_label_audit_v2.md` | 90-item weak-label audit 절차와 한계 |
| `01-plan/features/phase3_pair_local_temporal_encoder.plan.md` | pair-local history/action 설계 |
| `kcloudvpn_linux_ssh_runbook_ko.md` | RTX 3090 서버 실행 및 artifact 보존 방법 |

## 이전 Phase와 설계 근거

| 범주 | 문서 |
|---|---|
| Baseline/State/Graph | `phase0_baseline.md`, `phase1_libero_state.md`, `phase2_graph_spec.md` |
| 연구 재평가 | `literature_alignment_and_phase_reassessment.md`, `03-analysis/graph-clad-research-roadmap.analysis.md` |
| G1 설계 | `phase3_holder_object_action_graph_design.md`, `01-plan/features/phase3_target_centric_action_conditioned_gnn.plan.md` |
| 코드 구조 | `codebase_organization.md`, `repository_audit_20260813.md` |
| 미확인 사항 | `unknowns.md` |

## 과거 결과 문서

다음은 현재 결론을 만들기까지의 실험 근거다. 최신 실행 명령으로 사용하지 않는다.

- `phase3_holder_action_v2_smoke_result.md`
- `phase3_topology_action_followup_result.md`
- `phase3_action_semantics_gate_result.md`
- `phase3_corrected_threefold_seed0_result.md`
- `phase3_reduced_crossfold_gate_result.md`
- `phase3_pair_local_temporal_smoke_result.md`

## 운영 문서 상태

- `colab_mcp_next_session.md`는 Colab 중심이던 시기의 연결 메모다. 현재 실행 환경은
  KCloudVPN이므로 새 세션은 `NEXT_SESSION_PROMPT.md`를 사용한다.
- Notebook 실행 순서는 `../notebooks/README.md`가 기준이다.
- `archive/`는 현재 실행에서 import하지 않는 과거 bundle과 staging copy다.
- 루트에 있던 `.tmp_*` 전송 bundle은 삭제하지 않고
  `../archive/temporary_transfers_20260816/`로 보존 이동했다.
