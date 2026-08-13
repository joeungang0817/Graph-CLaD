# Phase 2D — official demonstration dataset

1. `split_manifest.py`로 demo ID 단위 split을 label 생성 전에 고정한다.
2. `state_replay.py`로 HDF5 simulator state를 직접 복원한다.
3. `temporal_holding.py`로 contact/closed/stability/follow evidence와 상태 전이를 만든다.
4. `build_demo_dataset.py`가 task 0/1/2를 multi-horizon graph dynamics shard로 변환한다.
5. `input_clean.py`와 `audit_relations.py`가 model-input leakage와 relation coverage를
   검사한다.
6. `persistence.py`가 Drive backup/restore manifest와 checksum을 관리한다.
7. `build_holding_event_dataset.py`가 official-demo state-replay 결과에서 holding
   positive, contact-only hard-negative, background window를 deterministic하게 선택한다.
8. `build_holding_target_dataset.py`가 `graph_t`와 `graph_target`의 valid holding
   label을 기준으로 future-positive, changed, hard-negative, background strata를 만든다.
9. `audit_holding_target_dataset.py`가 실제 Phase 3 cap sampler를 적용한 뒤 task별
   category와 episode coverage를 검사한다.

Action은 label 생성을 위해 replay하지 않으며 transition-conditioning input으로만
저장한다. `build_demo_dataset.py`는 동일 config/checksum일 때만 완료 shard를 resume한다.
Event category와 event index는 sampler/audit metadata이며 model input graph에는 넣지 않는다.
Target-aligned selection은 challenge-set 구성용 target-conditioned sampling이므로 natural
test 분포와 구분해서 보고한다.
