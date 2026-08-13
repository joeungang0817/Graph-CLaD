# Phase 3A holding weak-label audit v2

날짜: 2026-08-12  
상태: trajectory-enriched audit bundle과 interactive reviewer 준비 완료, human decision 대기.

## 목적과 주장 경계

Heuristic holding label의 초기 screening audit다. 새 training split이 아니며 label이
정확하다는 증거도 아니다. Task 3개 × onset/release/hard-negative × 10개로 총 90개다.
각 cell 10개는 흔하거나 심각한 failure mode를 찾을 수 있지만 task/event별 error rate를
정밀 추정하기에는 작다.

## Trajectory evidence

Audit v1은 current/future graph와 six-action window를 저장했다. V2는 t~t+6의 모든 graph
frame을 재구성해 robot–object distance, relative xyz, contact/holding change,
object-following residual, arm/gripper action을 최종 heuristic flag에만 의존하지 않고
검사할 수 있다.

- Audit item 90/90.
- 9개 task/event cell마다 10개.
- Item마다 trajectory frame 7개.
- Overlap deduplication 후 unique step graph 592개.
- 592/592 available, missing 0, payload conflict 0.
- Review row 90개, 초기 상태는 모두 unreviewed.

Original dataset과 audit v1은 변경하지 않았다. 592는 90개 review window에 필요한
deduplicated evidence copy 수다.

## Interactive review

`scripts/phase3/weak_label_audit_viewer.py`는 task/event/status filter, navigation,
current/future evidence table, trajectory plot, decision/error control, note, atomic CSV save,
JSON summary update를 제공한다. Viewer가 verdict를 자동으로 정하지 않는다.

- `pass`: independent trajectory view와 weak label이 일치.
- `label_error`: trajectory evidence가 label과 모순. Error type 필수.
- `ambiguous`: binary decision에 evidence가 부족. Evidence/error type 필수.

Ambiguous는 pass로 세지 않는다. Required cell을 모두 검사하기 전에는 결과를 보고하지
않는다.

## 확대 규칙

90개는 사전 등록한 최소 screen이다. Label error 또는 substantial ambiguity가 있는
task/event cell은 최소 30개로 확대한다. 10개에서 error 0이어도 rule-of-three one-sided
95% upper bound는 약 30%이고 30개에서 0이면 약 10%다. Publication-quality claim은 특히
release에서 cell당 30개, 총 270개가 필요할 수 있다.

## Persistent artifact

Root:
`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/weak_label_audit_v2_trajectory`

- `holding_weak_label_audit_manifest_v1.json`
- `holding_weak_label_audit_config_v2.json`
- `holding_weak_label_audit_runtime_manifest_v2.json`
- `holding_weak_label_audit_evidence_v1.jsonl.gz`
- `holding_weak_label_audit_review_v1.csv`

Code snapshot은 sibling `code_snapshot_weak_label_audit_v2`에 있다.
