# Graph-CLaD 연구 기록

이 문서는 연구 진행, 실험 결과, gate 판단을 날짜순으로 기록한다. 모델명, metric 이름,
config ID, 경로, SHA256은 재현성을 위해 영문 표기를 유지한다. 2026-08-13 이전 영문
원문은 `archive/pre_korean_translation_20260813.zip`에 보존했다.

## 연구 질문

CLaD의 두 시점 semantic transition을 object–relation graph transition으로 구조화했을 때
robot–object interaction과 object spatial transition이 더 명시적으로 보존되는지 검증한다.
Holding onset/release는 최종 목적이 아니라 representation과 architecture를 검사하는
중간 probe다. Loss만으로 가설을 판단하지 않고 event PR-AUC/F1, release/onset,
hard-negative FPR, calibration, action/edge control, non-graph baseline을 함께 본다.

## 2026-08-05 — Phase 0: 제공 CLaD baseline 실행 경계

- `baseline_code/`를 수정하지 않고 synthetic train/eval/EMA smoke 경로를 만들었다.
- Training forward의 loss 네 개, finite gradient, evaluation embedding shape `[B,H]`,
  EMA copy/update 계약을 검사하도록 했다.
- 실제 VLM preprocessing, action packing, Stage 2 diffusion policy는 확인되지 않았으므로
  `docs/unknowns.md`에 남겼다.
- 로컬 bundled Python에는 PyTorch가 없어 model smoke 자체는 project runtime에서만
  실행할 수 있었다. Phase 0 code/config 계약은 완료했다.

## 2026-08-05 — Phase 1A: LIBERO state와 API 조사

- LIBERO task 0의 live environment에서 observation, `env.sim`, robot/object pose,
  logical object identity, predicate interface를 조사했다.
- `logical_id`를 episode 간 graph identity로 사용하고 MuJoCo `body_id`는 runtime audit
  metadata로만 쓰기로 고정했다.
- Live capture는 snapshot 2개, snapshot당 object/fixture/site entry 23개,
  92-dimensional simulator state를 포함했다.
- 일부 `get_joint_state`, `is_open`, `is_close`가 object type에 따라 오류를 내므로
  unavailable predicate를 false로 채우지 않고 `valid=0`으로 유지하기로 했다.
- Colab runtime에서는 robosuite/MuJoCo mass-matrix compatibility와 신뢰한 init-state
  loading 옵션이 필요했다. Baseline model source에는 runtime patch를 적용하지 않았다.

## 2026-08-05 — Phase 2A: 정적 GraphSpec과 extractor

- Snapshot 하나를 deterministic directed graph로 바꾸는 `phase2.v1` 계약을 고정했다.
- Node는 robot/object/fixture/site, identity는 logical name이다.
- 공통 node numeric feature는 position, gripper/joint state와 validity mask를 포함한
  24차원이고 node type은 별도 one-hot이다.
- Valid position node 사이의 self-loop 없는 complete directed spatial edge를 사용했다.
- Colab live capture에서 graph 2개, graph당 node 24개와 edge 552개가 생성됐다.
- Predicate audit의 error는 unknown으로 유지했다. 이 graph는 controlled oracle graph이며
  RGB policy에서 simulator state 사용 가능성을 주장하지 않는다.

## 2026-08-05 — 문헌 재검토와 data 경로 개정

- 정적 graph snapshot만으로는 action-conditioned relational dynamics를 검증할 수 없음을
  확인했다.
- Episode 단위 split, action-bearing trajectory, temporal target, capability-aware label이
  필요하다고 판단했다.
- Phase 2A는 extractor regression baseline으로 유지하고 scripted Phase 2R과 official-demo
  Phase 2D를 분리했다.

## 2026-08-05 — Phase 2R scripted diagnostic

- Bounded/holding scripted probe, robot-base frame, semantic handler, transition dataset,
  scale-up validation을 구현했다.
- Robot-base coordinate와 containment semantic 처리에서 오류를 발견해 수정했다.
- Phase 2R은 extractor, contact/handler/frame, hard-negative 진단에는 유효하지만 scripted
  distribution이므로 main training source에서 제외하기로 했다.
- 관련 compact 결과는 `data/phase2r_*_summary.json`에 보존했다.

## 2026-08-05 — 초기 Phase 3 offline probe

- Graph pair와 action history로 future relation을 예측하는 offline probe 골격을 만들었다.
- Flat/non-graph baseline, graph encoder, action control, relation change metric을 비교했다.
- 초기 단일 split 결과는 낮은 label support와 dataset shortcut 가능성 때문에 architecture
  우월성 근거로 사용하지 않기로 했다.
- Task-family-held-out matched 재검증에서도 holding-positive와 changed-event support가
  부족한 문제가 드러났다.

## 2026-08-06 — Holding-positive 후속과 연구 roadmap v3

- Holding-positive 수집을 보완했지만 scripted source의 분포 한계를 해결하지 못했다.
- 연구 전체를 다시 조사해 main data source를 LIBERO official demonstration HDF5의 exact
  state replay로 변경했다.
- Phase 2D에서 full graph timeline, action alignment, holding state machine, event-centered
  sample을 만들고 Phase 3A QA 후 Phase 3B gate를 열도록 결정했다.
- `is_object_of_interest`와 BDDL-derived target relevance는 primary model input에서 제외했다.
- Phase 4는 Phase 3B와 Phase 3C gate 통과 전까지 차단했다.

## 2026-08-06 — Phase 2D official-demo replay

- `libero_spatial` task 0/1/2의 official demonstration 150개를 persistent Drive에서
  exact replay했다.
- Per-demo shard, merged task dataset, graph/action timing, provenance, fixed episode split을
  저장했다.
- 일시적으로 `holding=0`이 된 원인은 gripper contact mapping 문제였으며 runtime geom/body
  mapping을 바로잡아 전체 replay와 QA를 다시 수행했다.
- Sample/episode leakage 없이 task 0/1/2 natural graph dataset을 완성했다.
- Persistent root는 `/content/drive/MyDrive/Graph-CLaD/artifacts`로 고정했다.

## 2026-08-06 — Relation 및 event audit

- Contact, holding, on, inside와 temporal event support를 검사했다.
- Holding은 contact, gripper closure, 3-frame relative-pose stability, object following을
  조합한 heuristic weak label로 정의했다.
- `inside`는 valid label support가 없어 deferred했다.
- Holding-positive, onset, release, hard-negative index를 만들었다.
- Future event 정보를 이용해 선택한 challenge는 독립 test가 아니라 natural held-out
  episode의 event-enriched stress view로만 해석하기로 했다.

## 2026-08-07 — Phase 2D input-clean과 구조 정리

- Train input에서 target-derived relevance와 future-dependent field를 제거한 input-clean
  artifact를 만들고 structural QA를 통과했다.
- Colab 임시 `/content` source와 persistent Drive source의 역할을 분리했다.
- Phase별 source를 `scripts/phase*/`로 정리하고 과거 명령 호환을 위해 root wrapper를
  유지했다.
- Code snapshot, config, manifest, checkpoint, prediction, aggregate result를 함께
  보존하는 artifact 정책을 고정했다.

## 2026-08-07 — Legacy Phase 3 controlled experiment

- A100에서 smoke 후 task-family-held-out controlled experiment를 수행했다.
- 총 45 runs를 완료했다.
- 자연 prevalence 성능과 target-conditioned holding stress 성능이 크게 달라 하나의
  숫자로 합치지 않기로 했다.
- Holding을 primary relation으로 포함하고 target-aligned event dataset을 추가했다.
- Category-aware episode round-robin v2 sampler와 balanced-cap 비교를 수행했다.
- 이 결과는 모델/trainer 골격과 failure mode를 제공하지만 corrected protocol 이전
  결과이므로 최종 architecture 주장에는 사용하지 않는다.

## 2026-08-11 — Phase 3B-R1 계획 고정

- Complete graph보다 target-centric holder–object topology가 holding에 더 적합할 수 있다는
  가설을 세웠다.
- 비교 모델을 flat/pair MLP, complete/sparse GNN, late/global action,
  action-conditioned update의 통제 조합으로 정의했다.
- 평가 protocol을 먼저 고정하는 Gate E0, shortcut 진단 E1, architecture pilot E2,
  full comparison E3, action/edge control E4, history/context E5 순서를 정했다.
- Natural held-out test가 primary이고 challenge는 stress analysis임을 명시했다.

## 2026-08-11 — Holder–object smoke와 action 진단

- Sparse holder–object GNN G1 smoke에서 complete topology보다 유망한 task 0 신호를 봤다.
- G3 action-conditioned edge smoke와 action-signal diagnostic을 수행했지만 단일 task/seed
  근거에 불과했다.
- 이후 확인 결과 G1의 action은 message update가 아니라 prediction head에 들어가는
  late/global action이었다. 따라서 action-conditioned temporal edge 가설을 검증했다고
  주장하지 않기로 했다.
- B1-v2, G1-correct, S-0-G1, G1-train-shuffled의 near-parameter-matched 비교를 준비했다.

## 2026-08-11 — Holding metric과 parameter-count 교정

- 기존 display 일부가 holding metric의 nested path를 잘못 읽는 문제를 고쳤다.
- Near-matched parameter count는 다음과 같다.
  - B1-v2: hidden 55, 45,668 parameters.
  - G1-correct: hidden 48, 44,946 parameters.
  - S-0-G1: hidden 54, 44,784 parameters.
  - G1-train-shuffled: hidden 48, 44,946 parameters.
- Exact matching이 아니라 near matching임을 한계로 명시했다.

## 2026-08-12 — Reduced cross-fold gate 완료

- 3 task folds × 3 seeds × 4 conditions, 총 36/36 runs와 checkpoint 36개를 저장했다.
- Persistent artifact:
  - `phase3_reduced_crossfold_gate_v1.json`
  - `phase3B_R1_eval_manifest.json`
  - `checkpoints_reduced_crossfold_v1`
  - `code_snapshot_reduced_crossfold_v1`

9-run 평균은 다음과 같다.

| Model | Natural event F1 | Stress event F1 | Natural PR-AUC | Stress PR-AUC | Natural/Stress hard-neg FPR |
|---|---:|---:|---:|---:|---:|
| B1-v2 | 0.3285 | 0.7670 | 0.4047 | 0.4898 | 0.1075 / 0.1017 |
| G1 | 0.3400 | 0.7465 | 0.3872 | 0.4972 | 0.2146 / 0.1884 |
| S-0 | 0.3604 | 0.5927 | 0.3691 | 0.5159 | 0.2836 / 0.2613 |
| G1 train-shuffled | 0.3470 | 0.4746 | 0.2793 | 0.4093 | 0.3277 / 0.2931 |

- G1−B1 natural event F1은 +0.0115였지만 2/9 runs에서만 G1이 우세했다.
- Stress event F1 차이는 −0.0206이고 역시 2/9만 G1이 우세했다.
- B1 pair MLP가 가장 방어 가능한 baseline이며 graph의 일관된 추가 이점은 입증되지
  않았다.
- Natural release F1 약 0.21이 주요 병목이었다.
- 동일 task의 seed는 test episode를 공유하므로 9개 독립 표본으로 해석하지 않는다.

## 2026-08-12 — Corrected protocol v2 구현

Legacy 결과를 덮어쓰지 않고 별도 config/protocol/output root를 만들었다.

- Current auxiliary head의 action confound를 제거했다. 모든 비교 모델에 action-free
  current head를 사용하고 smoke에서는 동일한 loss 계약을 유지했다.
- Checkpoint criterion은 natural-validation conditional holding-event PR-AUC다.
- Future/current threshold는 natural validation에서 한 번 선택해 natural test, stress,
  perturbation control에 고정한다.
- Natural conditional/oracle-current event PR-AUC를 primary로 하고 thresholded F1,
  onset/release F1, hard-negative FPR, Brier, 10-bin ECE를 secondary로 저장한다.
- Predicted-current와 predicted-future를 쓰는 end-to-end event metric을 함께 저장한다.
- Sample probability, target, prediction, task, episode, sample ID, timestep, edge identity,
  event cluster를 gzip JSONL로 저장한다.
- Manifest QA에 task-local quota와 natural/stress overlap payload SHA256 검사를 추가했다.
- Train-shuffled donor는 task-local, episode-disjoint이며 action magnitude와 coarse state로
  matching하고 donor QA를 저장한다.

## 2026-08-12 — Corrected three-fold seed-0 gate

Task 1 seed 0 smoke 후 누락된 task 0/2만 실행해 12 runs를 결합했다. Task 1은 반복하지
않았다.

- G1−B1 conditional PR-AUC: task 0 −0.1161, task 1 +0.1322, task 2 +0.1717.
- Task-macro 차이 +0.0626, hierarchical bootstrap 95% CI [−0.0892, +0.1905].
- G1 event F1 0.3272, B1 0.4083.
- G1 release F1은 B1보다 +0.1626 [0.0701, 0.2967].
- G1 hard-negative FPR은 −0.3326 [−0.4201, −0.2429]로 개선됐다.
- 그러나 G1−train-shuffled G1 PR-AUC는 −0.0567 [−0.1943, +0.0572]이고 task 0에서
  부호가 뒤집혔다. Train-shuffled G1이 task-macro PR-AUC 0.4985와 event F1 0.4333으로
  가장 높았다.

결정은 `stop_gnn_three_seed_expansion_and_pivot_pair_local_temporal_encoder`다. 이는
graph가 쓸모없다는 뜻이 아니라 현재 late-action G1이 architecture/action gate를
통과하지 못했다는 뜻이다. Phase 4는 계속 차단했다.

## 2026-08-12 — Weak-label audit 준비

- Task 0/1/2 × onset/release/hard-negative × 10개, 총 90개 audit manifest를 만들었다.
- Trajectory-enriched v2는 각 item의 t~t+6 graph frame을 재구성했다.
- 요청 frame 592/592를 회수했고 missing/conflict가 없었다.
- Interactive viewer가 distance, relative xyz motion, object-following residual,
  contact/holding, arm/gripper action을 표시한다.
- 판정은 `pass`, `label_error`, `ambiguous` 중 사람이 명시적으로 입력해야 한다.
- 자동 판정을 human ground truth로 대체하지 않는다. 현재 review 상태는 0/90이다.

## 2026-08-12 — Pair-local temporal H0–H3 smoke

Task 1, seed 0에서 robot–object pair를 독립 처리하고 causal history/action의 2×2
factorial을 실행했다.

- H0: 현재 state, action 없음.
- H1: causal history, action 없음.
- H2: 현재 state + action.
- H3: causal history + action.

Natural conditional PR-AUC는 H0 0.3810, H1 0.4960, H2 0.5869, H3 0.5731이었다.
H2가 release F1 0.2143과 hard-negative FPR 0.0900으로 가장 좋았다. H3는 natural
end-to-end event PR-AUC 0.4227이 가장 높았지만 conditional PR-AUC와 release는 H2보다
낮았다. 한 task/seed 결과이므로 일반화 결론은 내리지 않았다.

## 2026-08-13 — Pair-local temporal three-fold seed-0 screen

H0–H3를 held-out task 0/1/2, seed 0에서 12/12 runs 완료했다.

- Result: `phase3_pair_local_temporal_threefold_seed0_v1.json`
- SHA256: `492d45521e6ccecbc4f0d89923f50d49642962d60c1c53daf093b5aec9b4d188`

Task-macro natural conditional PR-AUC:

| Model | PR-AUC | Event F1 | Release F1 | Hard-negative FPR |
|---|---:|---:|---:|---:|
| H0-state | 0.3626 | 0.3134 | 0.0931 | 0.3168 |
| H1-history | 0.4348 | 0.4920 | 0.0215 | 0.3358 |
| H2-action | 0.3941 | 0.3186 | 0.1729 | 0.2241 |
| H3-history-action | 0.4824 | 0.3617 | 0.1193 | 0.2339 |

- H3−H1 action increment는 PR-AUC +0.0476으로 3/3 task에서 양수였다.
- Release F1 +0.0978, hard-negative FPR −0.1019였지만 task 0 FPR은 악화됐다.
- H3−H0 PR-AUC +0.1198로 3/3 task에서 양수였으나 task 0 hard-negative FPR은
  0.1429에서 0.4762로 크게 악화됐다.
- Full three-seed factorial은 즉시 실행하지 않고 H3 aligned vs matched
  episode-disjoint train-action-shuffled control을 먼저 수행하기로 했다.

## 2026-08-13 — Action-alignment control 실행

- Protocol ID: `phase3-pair-local-temporal-action-alignment-seed0-v1`.
- Output:
  `/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/pair_local_temporal_action_alignment_seed0_v1`
- Episode-disjoint matched action donor를 사용하며 aligned H3와 train-shuffled H3를
  same fold/seed로 비교한다.
- Colab PID 12616으로 실행을 시작했지만 runtime이 종료되기 전에 중단되어,
  aligned H3와 episode-disjoint matched train-shuffled H3의 결과 artifact는 생성되지
  않았다. 따라서 이 control의 성능이나 gate 통과 여부는 아직 기록하지 않는다.

## 2026-08-13 — 저장소와 Colab source 비파괴 정리

- 실행 중 process, checkpoint, Drive output을 읽기 전용으로 유지했다.
- Colab live stage와 local source를 SHA256으로 비교했다. Manifest builder, probe,
  pair-local model, helper가 일치했고 action-alignment config를 로컬에 반영했다.
- `scripts/phase*/`를 source of truth로 유지하고 위험한 `src/` migration은 하지 않았다.
- `scripts/research_paths.py`로 local/Colab artifact path를 공통화했다.
- 대용량 dataset/checkpoint/prediction은 복사하지 않고 경로와 checksum만 문서화했다.
- 임시 bundle/stage/cache는 삭제하지 않고 보관 후보로 분류했고, 2026-08-16에
  `archive/temporary_transfers_20260816/`로 보존 이동했다.

## 2026-08-13 — 공식 Phase 이름 교정과 문서 한국어화

- 처음 추가한 순번형 notebook 이름 `phase_01`~`phase_05`가 공식 roadmap Phase와
  충돌함을 확인했다. 특히 architecture training/evaluation은 Phase 4/5가 아니라 모두
  Phase 3B였다.
- Notebook을 공식 단계에 맞게 다음처럼 교정했다.
  - 환경 준비
  - Phase 0 baseline
  - Phase 1A state/API audit
  - Phase 2A static graph contract
  - Phase 2R scripted diagnostic
  - Phase 2D official-demo dataset
  - Phase 3A dataset/label QA
  - Phase 3B architecture gate와 evaluation/control
- Phase 3C와 Phase 4 이후 notebook은 gate 통과 전까지 만들지 않았다.
- 기존 영문 Markdown 원본은 `archive/pre_korean_translation_20260813.zip`에 보존하고
  활성 설명서와 연구 기록을 한국어로 정리했다. Code identifier, metric, path, SHA는
  재현성을 위해 원문 표기를 유지했다.

## 현재 gate와 다음 단계

1. 실행 중 action-alignment control의 완료 여부, stderr, expected run count, checkpoint,
   prediction, runtime manifest, code snapshot을 확인한다.
2. Aligned H3가 shuffled control보다 natural PR-AUC에서 최소 2개 task 우세한지 본다.
3. Release와 hard-negative 안전성을 함께 판단한다.
4. 90-item weak-label 수동 review를 완료한다.
5. Gate를 통과할 때만 H3/H1/H3-shuffled의 seeds 1/2 확대를 검토한다.
6. 이후 Phase 3C CLaD-aligned foresight bridge를 검토한다.
7. Phase 4에서는 semantic CLaD, pair-local temporal, graph-transition representation을
   같은 데이터와 동일 capacity probe로 비교한다.

현재까지 graph의 일반적 우월성, causal action 효과, 최종 Graph-CLaD representation의
우월성은 입증되지 않았다.

## 2026-08-14 — 현재 runtime의 T4 상태 기록

현재 연결된 Colab runtime에서만 NVIDIA T4를 사용할 수 있다. 이는 영구적인 GPU 제약이
아니며 다음 runtime에서는 실제 GPU를 다시 확인해야 한다. 현재 pair-local/G1 model의
config batch size 64는 우선 유지한다. T4에서 OOM이 실제로 발생할 때만 새 config version에서
batch 32를 사용하며, 이 protocol 변경을 A100/기존 결과와 동일 run으로 합치지 않는다.
GPU 이름, VRAM, CUDA, batch size는 각 launcher/result metadata에 기록한다.

## 2026-08-15 — KCloudVPN Linux 실행 전환 준비

- 이후 학습 실행 환경을 Colab에서 KCloudVPN Linux SSH 서버
  `ubuntu@172.10.5.118`로 전환하기 위한 경로·config·runbook을 추가했다.
- 기존 Colab process, Drive artifact, checkpoint, prediction은 중단·삭제·덮어쓰지
  않는다. KCloudVPN output은 `GRAPH_CLAD_ARTIFACT_ROOT` 아래 별도 version으로
  저장한다.
- `${GRAPH_CLAD_ARTIFACT_ROOT}` 확장을 지원하는 KCloudVPN용 pair-local
  three-fold와 action-alignment config를 추가했다. `require_persistent_output`와
  `persistent_output_roots`로 서버 영구 디스크 밖의 output을 거부한다.
- Colab manifest의 `/content/drive` 절대경로 문제를 피하기 위해
  `configs/phase3_kcloudvpn_linux_eval_manifest_v2.json`으로 서버에서 manifest를
  재생성하도록 했다. 자연 dataset, target-aligned dataset, demo split manifest가
  실행 필수 입력이다.
- KCloudVPN의 GPU 종류는 아직 확인되지 않았으므로 T4를 영구 제약으로 기록하지
  않는다. 실제 GPU/VRAM/CUDA 여부는 매 실행의 runtime manifest에 기록한다.

## 2026-08-16 — KCloudVPN GPU driver 확인

- KCloudVPN VM의 PCI passthrough에서 NVIDIA GeForce RTX 3090 24GB가 확인됐다.
- NVIDIA driver `595.71.05`와 CUDA compatibility version `13.2`가 `nvidia-smi`에서
  정상 표시됐고, 확인 시점에는 GPU 사용 프로세스가 없었다.
- TITAN RTX로 전달받은 사양과 실제 장치가 다르므로, 실험 기록에는 실제
  `nvidia-smi` 결과인 RTX 3090을 사용한다. GPU 종류는 영구 연구 조건이 아니라
  runtime metadata로 보존한다.

## 2026-08-16 — Phase2D 입력 이전과 corrected manifest 복구

- KCloudVPN의 `/home/ubuntu/graphclad-artifacts/phase2d/data` 아래에 다음 입력을
  복사하고 압축 해제했다.
  - natural dataset: 836 MB.
  - holding target-aligned dataset: 357 MB.
  - demo split manifest: 76 KB.
- `configs/phase3_kcloudvpn_linux_eval_manifest_v2.json`으로 manifest를 재생성하려 했으나
  process가 `Killed`로 종료됐다. 현재 builder가 gzip JSONL에서 읽은 natural/target
  graph payload 전체를 동시에 list에 보관하므로 CPU memory OOM으로 판단했다.
- 같은 작업을 반복하지 않고 기존 Colab의 corrected `status=pass` manifest를 서버로
  복사했다. Colab 원본은 별도 파일로 보존하고 source root만
  `/home/ubuntu/graphclad-artifacts`로 바꾼 portable copy를 runner 입력으로 사용한다.
- Portable manifest 검증은 `status=pass`, `folds=3`이다. Sample key, split, payload
  hash, quota/leakage QA 결과는 변경하지 않았다. Manifest SHA는 경로 변환 때문에
  Colab 원본과 다르므로 두 파일을 함께 보존한다.

## 2026-08-16 — KCloudVPN action-alignment 재실행 준비

- Config:
  `configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json`.
- Expected scope: H3 train-shuffled × 3 task folds × seed 0 = 3 runs.
- Output:
  `/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1`.
- `tmux` session 이름은 `graphclad-align`로 안내했다. 이후 사용자가 H3 action-alignment
  실행을 시작했다고 확인했다. 현재 process의 실행/완료 상태와 artifact 무결성은 server
  process와 output artifact로 다시 확인해야 하며 시작 명령을 반복하지 않는다.
- 현재 runner는 shuffled H3만 학습한다. Aligned H3와의 paired 비교는 기존 Colab
  three-fold result와 H3 per-sample prediction을 별도로 읽어야 한다.

## 2026-08-16 — 제출용 전체 CLaD 연결 방향

- Phase 3 gate 이후 no-future-action foresight bridge와 CLaD Stage 1 통합을 먼저 수행한다.
- Stage 2는 원 논문 설명에 가까운 canonical DDPM Diffusion Policy로 통제 구현한다.
  Stage 1은 freeze하고 current observation과 predicted foresight를 modality별 FiLM으로
  conditioning하며 action horizon `tau=6`, epsilon-prediction loss를 우선 사용한다.
- Policy-only, semantic CLaD foresight, 선택 pair-local/graph foresight를 같은 policy
  capacity와 training/rollout budget으로 비교한다.
- 공식 Stage 2 code와 LIBERO-LONG 전체 protocol이 없으므로 결과는
  CLaD-compatible controlled reimplementation으로 표현한다. 논문의 94.7%와 직접
  비교하거나 공식 재현이라고 주장하지 않는다.
- RTX 3090과 제출 일정에서는 one-task smoke와 reduced budget을 먼저 실행하고,
  정상 동작과 비교 가능성을 확인한 뒤 최적 후보만 확대한다.

## 2026-08-16 — 문서와 임시 전송 파일 정리

- `docs/CURRENT_STATUS.md`를 현재 실행 상태의 단일 진입점으로 추가했다.
- `docs/README.md`에서 설계·결과·운영·legacy 문서를 목적별로 분류했다.
- `docs/NEXT_SESSION_PROMPT.md`에 KCloud 경로, gate, claim limit, Stage 2 결정을 포함한
  새 세션용 인계 프롬프트를 저장했다.
- `docs/RESEARCH_WORKFLOW_FOR_BEGINNERS.md`에 연구 질문, 데이터 생성, Phase 0–8,
  metric, control, 통계와 최종 CLaD 연결을 신규 연구자 관점에서 설명했다.
- `docs/CODEBASE_GUIDE_FOR_BEGINNERS.md`에 폴더, 주요 Python 파일, 입출력, data schema,
  model/runner/analysis 흐름과 향후 Stage 1/2 ownership을 상세히 기록했다.
- 활성 source/config/notebook은 이동하지 않았다. 루트의 `.tmp_*` Colab transfer
  bundle과 stage만 삭제 없이 `archive/temporary_transfers_20260816/`로 이동해 보존한다.

## 2026-08-16 — 통합 연구계획서 v4 작성

- 초기 `Graph_CLaD_Stage2_최종목표_연구실행계획서.pdf`, 수정 roadmap v3, Phase 3
  corrected GNN 결과, pair-local H0–H3 결과와 제출용 Stage 2 결정을 통합했다.
- 새 canonical 계획서는
  `docs/01-plan/features/graph-clad-integrated-research-v4.plan.md`다.
- 연구 목표를 GNN 우월성 입증으로 고정하지 않고 semantic, pair-local temporal,
  object–relation graph representation의 same-data/same-capacity 비교로 명확히 했다.
- Action-alignment, human weak-label audit, no-future-action Phase 3C, Stage 1 adapter,
  canonical DDPM Stage 2, paired rollout의 단계별 통과·중단 기준을 추가했다.
- v3 roadmap은 삭제하거나 덮어쓰지 않고 이전 Phase 개정 근거로 보존한다.
- 후속 v4.1에서는 Phase 0–2D 기반 검증, legacy 45-run 탐색, task-0 topology/action
  smoke, 36-run reduced cross-fold, corrected hierarchical bootstrap, pair-local H0–H3,
  weak-label audit 준비 결과를 protocol별로 분리해 계획서 본문에 추가했다.
- v4.2에서는 연구기록과 `unknowns.md`를 다시 대조해 metric display path, same-episode
  legacy shuffle, validation threshold, 통계 독립 단위, late/global G1 action 명명 교정과
  authoritative artifact 위치를 추가했다. Stage 2의 미공개 network/noise/rollout 설정은
  baseline smoke 전 versioned config로 고정할 미확정 항목으로 명시했다.
- `docs/NEXT_SESSION_PROMPT.md`를 v4.2 기준으로 다시 작성했다. 새 세션은 local git diff를
  보존하고 KCloud action-alignment 상태를 artifact로 재확인하며, running/completed run을
  중단하거나 반복하지 않는다. Paired 분석에 Colab aligned H3 prediction이 별도로
  필요하다는 점과 제출 직전 최소 실행 순서도 인계한다.

## 2026-08-16 — H3 train-shuffled action-alignment 3-fold 완료와 gate 실패

- Protocol: `phase3-pair-local-temporal-action-alignment-seed0-v1`.
- Config: `configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json`.
- KCloudVPN에서 `H3-train-shuffled` × task folds 0/1/2 × seed 0의 3/3 runs를 완료했다.
- 각 run의 trainable parameter 수는 60,476이고 natural-validation에서 선택된 holding
  threshold는 세 fold 모두 0.95였다.
- Runner 최종 stdout은 `status=completed`, `runs=3`을 기록했다. Runtime manifest는
  `/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1/runtime_manifest.json`이다.

Natural conditional/oracle-current event stdout과 기존 aligned H3 기준값의 비교는 다음과
같다.

| Task | Aligned H3 PR-AUC | Shuffled H3 PR-AUC | Aligned−Shuffled | Aligned F1 | Shuffled F1 | Aligned release F1 | Shuffled release F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4009 | 0.4245499224 | 약 −0.0236 | 0.4074 | 0.4523809524 | 0.2353 | 0.4000000000 |
| 1 | 0.5731 | 0.5059178137 | 약 +0.0672 | 0.3357 | 0.7085714286 | 0.1075 | 0.5882352941 |
| 2 | 0.4734 | 0.4913168425 | 약 −0.0179 | 0.3419 | 0.6033519553 | 0.0150 | 0.3733333333 |
| Task macro | 0.4824 | 0.4739281929 | 약 +0.0085 | 0.3617 | 0.5881014454 | 0.1193 | 0.4538562092 |

- 사전 gate는 aligned H3의 natural PR-AUC가 shuffled H3보다 최소 2/3 tasks에서 높을
  것을 요구한다. 실제 우세는 task 1의 1/3뿐이므로 primary gate는 실패했다.
- Aligned task-macro PR-AUC가 약 +0.0085 높다는 사실은 사전 정의한 task별 우세 기준을
  대체하지 않는다. Outer evaluation unit은 task fold다.
- Aligned release F1은 shuffled보다 3/3 tasks에서 낮았다. F1 계열은 fold별 validation
  threshold에 민감하지만, 이 결과는 action alignment의 secondary support도 제공하지 않는다.
- 따라서 올바르게 정렬된 action의 causal/semantic 효과를 입증했다고 주장하지 않는다.
  계획대로 H3/H1/H3-train-shuffled seeds 1/2 확대를 중단하고 H3는 architecture-screen
  finding으로만 보존한다. Phase 3C에서는 H1 또는 다른 action-free pair-local
  representation을 우선 검토한다.
- 위 표에 없는 hard-negative, end-to-end, challenge와 calibration 값은 result JSON 및
  per-sample prediction을 이용한 후속 상세 분석에서 기록한다. 현재 stdout에 없는 값을
  추정해 채우지 않는다.

## 2026-08-16 — 지속적인 연구기록 운영 원칙 확정

- 이후 모든 주요 실험 완료·중단·실패, gate 판정과 해석을 발생한 세션에서 바로
  `docs/research_log.md`에 기록한다.
- Source·config·분석 기준을 수정할 때도 변경 파일, 수정 이유, 검증 결과와 기존
  artifact/결론에 미치는 영향을 기록한다.
- 기록은 확인된 artifact와 출력에 근거하며, 확인하지 않은 metric이나 상태는 추정하지
  않고 명시적인 미완료 항목으로 남긴다.

## 2026-08-16 — 분리 result JSON paired 분석기 보강

- Aligned H3와 KCloudVPN shuffled H3가 서로 다른 result JSON과 artifact root에 있어
  기존 `analyze_corrected_predictions.py`의 단일 result 입력 계약으로는 직접 비교할 수
  없었다.
- 분석기에 `--left-result`, `--right-result`, `--left-prediction-root`,
  `--right-prediction-root`를 추가했다. 원본 result JSON의 절대경로를 수정하지 않고,
  prediction 파일명으로 지정 root를 재지정한다. 기존 `--result` 단일 파일 방식은
  하위 호환으로 유지한다.
- 분리 result JSON과 aligned prediction root 재지정 경로를 검증하는 테스트를 추가했다.
  `python -m unittest tests.test_phase3_corrected_analysis` 결과는 2 tests, OK다.

## 2026-08-16 — H3 aligned-vs-shuffled paired bootstrap 완료

- Analysis comparison은 `H3-history-action_minus_H3-train-shuffled`이다.
- Output은 KCloudVPN의
  `corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1/aligned_vs_shuffled_bootstrap_v1.json`에 저장했다.
- Task fold → episode → event cluster 계층으로 2,000회 재표집했고 bootstrap seed는
  `20260816`이다.

| Metric | Aligned−shuffled estimate | 95% CI |
|---|---:|---:|
| Event PR-AUC | +0.0085088954 | [−0.0506334643, +0.0771584896] |
| Event F1 | −0.2264341004 | [−0.3955016732, −0.0459489249] |
| Release F1 | −0.3345700114 | [−0.5339250228, −0.1016183067] |
| Hard-negative FPR | +0.0056898657 | [−0.0795714496, +0.1007211270] |

- Event PR-AUC와 hard-negative FPR interval은 0을 포함해 aligned action의 뚜렷한 이점을
  지지하지 않는다.
- Event F1과 release F1 interval은 모두 0 아래이며 aligned가 낮다. F1 계열은 각
  model/fold의 natural-validation threshold를 적용한 thresholded metric이라는 제한을
  함께 둔다.
- 이 결과는 사전 2/3-task primary 기준 실패와 일치한다. Phase 3B action-alignment gate
  실패를 확정하고 seeds 1/2 확대를 중단한다.
- 상세 결과와 claim limit은
  `docs/phase3_pair_local_temporal_action_alignment_seed0_result.md`에 고정했다.

## 2026-08-16 — Human weak-label audit 일시 보류와 Phase 3C smoke 선행

- Interactive audit viewer를 열었지만 현재 화면은 weak label의 `holding`, `state`,
  `followed`, `confidence`와 label 생성에 사용된 contact/closure/follow 신호를 함께
  보여준다. 같은 규칙을 보고 판정하면 독립 human validation보다 rule-consistency
  confirmation에 가까워진다는 한계를 확인했다.
- 사용자 결정으로 90-item human review는 0/90 상태에서 일시 보류한다. Audit package와
  빈 review CSV는 그대로 보존하며 pass로 처리하지 않는다.
- Phase 3C one-task/seed-0 technical smoke는 shape, leakage, metric과 artifact 계약 검증
  목적으로 먼저 진행한다.
- Human audit 완료 전에는 holding weak label의 정확성, 작은 metric gain, 최종
  representation 우월성을 주장하지 않으며 multi-seed/full-scale 확대도 하지 않는다.

## 2026-08-16 — 원 CLaD 방법 재확인과 Phase 3C graph 후보 구분

- `CLaD_CVPR_2026.pdf`의 Section 4와 Figure 2, 제공된
  `baseline_code/LatentDynamics.py`를 대조했다. 원 CLaD Stage 1은 현재 상태만 보는
  action-free encoder가 아니다. `t-tau`와 `t`의 semantic/proprioceptive state 및 그
  사이에 이미 실행된 과거 action `a[t-tau:t]`로 두 transition token을 만든 뒤,
  proprioceptive transition이 semantic transition을 query하는 비대칭 cross-attention으로
  shared dynamics를 구성하고 `t+tau` latent를 예측한다. 미래 action은 입력하지 않는다.
- 따라서 Phase 3C에서는 `strict no-action`과 `no-future-action but causal past-action`을
  구분해야 한다. 전자는 안전한 ablation이고 후자가 원 논문 입력 계약에 더 가깝다.
- 과거 P3 geometric GNN은 한 시점의 complete directed graph에서 상대 xyz, 거리,
  geometry-valid edge를 사용하고, action은 message passing 뒤 prediction head에 붙여
  미래 relation을 직접 분류했다. 이는 semantic relation token과 graph history를 담는
  CLaD-aligned object-relation transition encoder와 동일하지 않다.
- Weak-label audit 완료는 새 encoder를 자동으로 추가하는 절차가 아니다. Audit가 holding
  label을 지지하면 동일한 frozen representation에 holding onset/release probe를 정식
  secondary target으로 추가한다. 오류가 발견되면 label version을 수정하고 영향받은
  실험만 재평가한다.
- 권장 Phase 3C 비교안은 semantic CLaD baseline, H1 action-free pair-local baseline,
  action-free object-relation graph, 그리고 원 논문 정렬을 위한 causal-past-action graph
  variant다. 새 graph의 효과를 relation/history 효과와 혼동하지 않도록 geometric-only
  graph ablation을 유지한다. 이 구성은 아직 최종 확정 전 설계 권고로 기록한다.

## 2026-08-16 — Phase 3C two-tier 비교계획 확정

- 사용자 확인에 따라 `no-future-action`과 `strict action-free`를 분리했다. Action-free
  graph만 원 CLaD와 다른 것이 아니라 action을 제거한 semantic/pair/graph 모델 모두가
  원 CLaD 입력 계약과 다르다. 원 CLaD는 future action 없이 causal past action
  `a[t-tau:t]`을 사용한다.
- Representation과 action 효과를 혼동하지 않도록 Phase 3C를 두 tier로 확정했다.
  - Tier A strict action-free: `C3-Sem-AF`, `C3-Pair-AF`, `C3-GeomGraph-AF`,
    `C3-RelGraph-AF`.
  - Tier B CLaD-causal: `C3-Sem-PastAct`, `C3-Pair-PastAct`,
    `C3-RelGraph-PastAct`.
- Tier A에서는 모든 모델이 같은 `<=t` state/history만 보고 action은 전혀 보지 않는다.
  `C3-GeomGraph-AF`와 `C3-RelGraph-AF`는 topology, history, capacity, objective를 맞추고
  valid current relation token의 유무를 핵심 차이로 둔다. 이전 P3 geometric GNN은
  single-snapshot/future-interval-action/direct-classifier 계약이라 그대로 재사용하지 않는다.
- Tier B는 Tier A에서 relation graph가 object displacement 또는 valid spatial transition의
  natural held-out metric으로 semantic 및 matched geometric graph보다 이점을 보일 때만
  실행한다. 세 모델 모두 같은 causal past-action availability를 사용하고 future action과
  future graph는 금지한다.
- Human weak-label audit은 모델을 추가하는 gate가 아니다. Audit가 label을 지지하면 같은
  frozen representations의 holding onset/release probe를 정식 secondary target으로
  승격한다. Audit 전 technical smoke의 primary target은 object displacement와 valid
  spatial transition이다.
- 이 결정을 canonical v4 plan, `CURRENT_STATUS.md`, `NEXT_SESSION_PROMPT.md`, beginner
  workflow 및 root research guide에 반영했다.

## 2026-08-16 — Phase 3C causal-past-action main comparison으로 직행

- 제출 속도를 우선한 사용자 결정으로 strict action-free Tier A의 4-model 선행 실행을
  제거했다. Phase 3C는 원 CLaD 입력 계약에 가까운 causal past-action main comparison으로
  바로 시작한다.
- Main comparison은 `C3-Sem-PastAct`, `C3-Pair-PastAct`,
  `C3-GeomGraph-PastAct`, `C3-RelGraph-PastAct` 네 모델이다. 모든 모델은 같은 `<=t`
  state/history, 같은 `a[t-tau:t]` past action, 같은 split/capacity/probe/update budget을
  사용한다. Future action `a[t:t+tau]`과 future graph는 금지한다.
- Past-action record는 Phase 2D를 다시 replay하지 않고 같은 episode/tau의 연속 sample을
  join해 만든다. `(t-tau -> t)` sample은 graph history와 past action을 제공하고
  `(t -> t+tau)` sample은 future target만 제공한다. 공유 `graph[t]` hash, episode, split,
  tau가 모두 일치해야 하며 두 번째 sample의 future action window는 encoder에서 버린다.
- `C3-GeomGraph-PastAct`는 relation graph의 추가 가치를 분리하기 위한 필수 matched
  control이다. 과거 P3는 single snapshot/future-interval-action/direct-classifier여서
  그대로 재사용하지 않는다.
- `C3-RelGraph-PastAct`가 semantic 및 matched geometric graph보다 natural object
  displacement 또는 valid spatial transition에서 유망할 때만 `C3-RelGraph-AF`와
  `C3-RelGraph-ShuffledPastAct`를 추가한다. 따라서 action-free는 main 선행 단계가 아니라
  유망 graph의 action-use ablation으로 축소됐다.
- Human audit의 역할은 이전 결정과 동일하다. 새 모델을 추가하지 않으며, 완료 후 holding
  onset/release를 정식 secondary target으로 승격할 수 있다.
- Canonical plan version을 4.4로 올리고 `CURRENT_STATUS.md`, `NEXT_SESSION_PROMPT.md`,
  beginner workflow, root guide와 README를 같은 순서로 갱신했다.

## 2026-08-16 — Phase 3C graph encoder 2×2와 6-model main comparison 확정

- 사용자가 Transformer가 아닌 graph encoder 두 개도 같은 Phase 3C smoke에 포함하는
  방안을 제안했다. 이는 임의 model 추가가 아니라 edge content와 encoder family를
  분리하는 2×2 factorial이므로 main comparison에 포함하기로 결정했다.
- 최종 six-model main set은 `C3-Sem-PastAct`, `C3-Pair-PastAct`,
  `C3-GeomMPNN-PastAct`, `C3-RelMPNN-PastAct`, `C3-GeomTx-PastAct`,
  `C3-RelTx-PastAct`다.
- Graph 2×2의 row는 temporal MPNN과 edge-token Transformer, column은 geometry-only와
  geometry+relation이다. Relation effect는 `RelMPNN−GeomMPNN`과 `RelTx−GeomTx`, encoder
  effect는 `GeomTx−GeomMPNN`과 `RelTx−RelMPNN`으로 분리 보고한다.
- 네 graph cell은 같은 node/edge set, topology, causal graph history, past-action embedding,
  output/probe와 training budget을 사용한다. Base edge token에 같은 action embedding을
  제공하고 MPNN은 message input, Transformer는 edge-token input으로 처리한다. Parameter는
  near-match하고 실제 차이를 공개한다.
- 과거 P3/G1은 temporal/action/target 계약이 다르므로 새 MPNN cell로 재사용하지 않는다.
- Relation model이 유망할 때만 `C3-RelPool-PastAct`를 추가해 relation feature와
  scene-level interaction을 분리하고, 선택된 relation encoder에 no-action 및
  shuffled-past-action controls를 추가한다.
- Canonical plan version을 4.5로 올리고 current status, next-session prompt, beginner
  workflow, root guide와 README를 같은 모델 집합으로 갱신했다.

## 2026-08-17 — Graph-CLaD improvement 목표와 Phase 3C primary protocol 확정

- 최종 연구 목표를 oracle representation screen 자체가 아니라 controlled semantic CLaD
  대비 Graph-CLaD의 개선으로 확정했다. 최종 성능 주장은 같은 Stage 2 policy와 rollout
  budget의 paired task success 결과가 있을 때만 허용한다.
- Phase 3C의 유일한 primary offline contrast는 사전 지정한
  `C3-RelMPNN-PastAct − C3-Sem-PastAct`이고, target/metric은 `tau=6` valid
  spatial-relation-change task-macro PR-AUC로 고정했다. Object displacement와
  source→destination은 secondary로 이동했다.
- `C3-Sem-PastAct`는 제공된 CLaD Stage 1 core와 replayed visual/language/proprioception,
  causal past action을 사용하는 controlled semantic baseline이어야 한다. 이를 충족하지
  못하면 `C3-SemProxy-PastAct`로 이름을 바꾸며 primary CLaD 비교를 대체하지 않는다.
- Graph와 같은 full-scene oracle state를 받되 edge/message passing이 없는
  `C3-SceneSet-PastAct`를 information-matched non-graph baseline으로 추가했다. Relation
  change no-change와 future-state copy-current trivial baseline도 필수 보고한다.
- Holding human audit은 0/90 상태로 보류할 수 있으나, Phase 3C에서는 holding을 loss,
  checkpoint, threshold, model selection과 gate에서 완전히 제외한다. Holding은 audit 전
  diagnostic으로만 저장한다.
- Core main comparison은 semantic, SceneSet, pair, GeomMPNN, RelMPNN으로 두고 RelMPNN을
  primary graph candidate로 사전 지정했다. Geometry/relation × Transformer cell은 core
  결과 후 secondary backbone robustness comparison으로 둔다.
- Simulator-state graph를 계속 사용하는 최종 variant는 `Oracle Graph-CLaD`로 표기하며
  RGB에서 graph를 추출하거나 deployable perception을 검증한 것으로 주장하지 않는다.

## 2026-08-17 — Phase 3C 코드 생성·실행 계획과 기술 설계 확정

- 사용자 요청에 따라 다음 작업이 코드 생성 단계임을 확인하고, 바로 full training code부터
  만들지 않고 data contract → semantic feature → controlled CLaD → six-model core → runner
  순서의 구현 계획과 gate를 문서화했다.
- 구현 계획은 `docs/01-plan/features/phase3c-oracle-graph-clad-core.plan.md`, 파일·schema·tensor·
  model·loss·test·실행 interface의 상세 설계는
  `docs/02-design/features/phase3c-oracle-graph-clad-core.design.md`에 고정했다.
- 기존 v4.6 계획의 공정성 문제를 수정했다. `SceneSet`은 full-scene state control이지만
  explicit relation edge token을 받지 않아 RelMPNN과 exact information match가 아니다.
  따라서 RelMPNN과 동일 edge token을 받고 message passing만 제거한
  `C3-RelPool-PastAct`를 conditional experiment에서 필수 core control로 승격했다.
- Core는 `Sem`, `SceneSet`, `Pair`, `GeomMPNN`, `RelPool`, `RelMPNN`의 6개로 확정했다.
  Primary는 `RelMPNN−Sem`, exact-token 구조 guard는 `RelMPNN−RelPool`, broader scene-state
  guard는 `RelMPNN−SceneSet`이다. Transformer는 core 후 secondary로 유지했다.
- Primary 출력은 global semantic model과 graph model이 같은 head로 비교될 수 있도록
  `tau=6` sample-level relation any-change 8-vector로 구체화했다. Candidate edge는
  object→object/fixture이고, relation은 현재 handler에 실제 존재하는 `left`, `right`,
  `front`, `behind`, `above`, `below`, `contact`, `on`만 쓴다. 구현되지 않은 `near`와
  `support`, support 부족 `inside`, audit 미완료 `holding`, unary `open/close`는 제외했다.
- Relation eligibility는 test를 보지 않고 train/validation positive·negative minimum
  support로 고정한다. Fold당 evaluable relation이 2개 미만이면 학습 전에 unsupported로
  중단하도록 했다.
- 기존 Phase 2D artifact에는 RGB/semantic embedding이 없음을 확인했다. Controlled semantic
  CLaD를 위해 official HDF5 state의 joined-manifest unique frame만 render해 DecisionNCE
  embedding store를 만드는 단계를 필수로 추가했다. CLaD 논문은 DecisionNCE P/T variant와
  optimizer를 명시하지 않으므로 `DecisionNCE-P`, AdamW 설정은 controlled assumption으로
  표기하고 repository commit/checkpoint/config SHA를 저장한다.
- Phase 3C screen은 fold별 base CLaD를 원 latent/reconstruction objective로 학습한 뒤 freeze하고,
  six candidate adapter와 동일 head를 relation change/motion target으로 학습하는 matched
  architecture screen으로 정의했다. 따라서 이를 순수 self-supervised frozen-probe 결과로
  부르지 않는다. 최종 Graph-CLaD는 Phase 4에서 선택 구조를 CLaD foresight residual로 다시
  통합하고 adapter-off equivalence를 통과한 뒤 Stage 2로 넘긴다.
- RelMPNN을 최종 후보로 선택할 경우 Stage 2에서도 semantic+RelPool을 유지한다. 최종 policy
  비교는 policy-only, semantic, semantic+SceneSet, semantic+RelPool, semantic+RelMPNN으로
  구성해 relation token의 효과와 message passing의 효과를 policy 수준에서도 분리한다.
- 아직 Phase 3C model 코드는 생성하거나 실행하지 않았다. 다음 구현은 위 plan/design의
  Milestone 1인 contract, streaming join, target/support report와 unit test부터 시작한다.

## 2026-08-17 — Phase 3C Milestone 1 구현 시작

- `scripts/phase3c/` package를 생성하고 torch/LIBERO와 독립적인 causal data contract를
  구현했다. `contracts.py`는 canonical graph hash, action shape `[6,7]`, forbidden input
  field, object→object/fixture candidate edge, 8개 relation any-change target, displacement와
  train/validation support report를 정의한다.
- `build_joined_manifest.py`는 Phase 2D의 `(t-6 → t)`와 `(t → t+6)` sample을 episode/task/demo/
  split/tau와 shared `graph[t]` SHA-256으로 검증한 뒤 join한다. left sample의 past action만
  `past_action_window`으로 복사하고 right sample의 future action은 output schema에 넣지 않는다.
- `io.py`에 gzip JSONL streaming reader와 atomic JSON/JSONL writer를 추가했다. 오류가 나면
  destination을 덮어쓰지 않고 QA report에 실패 원인과 counter를 남긴다.
- `validate_action_timing.py`와 pure helper test를 추가했다. 실제 HDF5/LIBERO 실행은 아직
  하지 않았으며, SSH에서 train/validation tolerance를 먼저 고정한 뒤 test frame에는 frozen
  tolerance를 적용해야 한다.
- Synthetic test 8개가 통과했고, 새 package `compileall` 및 `git diff --check`도 통과했다.
  현재 구현은 Milestone 1까지이며 DecisionNCE feature store, CLaD wrapper, structured model은
  아직 구현하지 않았다.
## 2026-08-17 — Phase 3C Milestone 2 semantic feature store implementation

- Added `scripts/phase3c/build_semantic_feature_store.py`. It separates a
  dependency-free camera/config layer from the SSH-only HDF5/LIBERO/DecisionNCE
  extraction path. The builder consumes the causal joined manifest and renders
  only unique `t-6`, `t`, and `t+6` state frames per demo.
- Camera selection is an exact two-key contract. The selected key, channel
  order, vertical-flip flag, frame shape/dtype, and observation inventory are
  recorded; missing keys, inconsistent shapes, non-finite pixels, or implicit
  orientation/preprocessing fallbacks fail the build.
- Added a thin frozen DecisionNCE wrapper with explicit image/text encoding,
  feature-dimension checks, finite-value checks, and checkpoint provenance.
  Per-demo `.npz` shards store `steps`, `view0`, `view1`, `language`, and
  simulator state-restore error; `manifest.json` records source/HDF5/checkpoint
  hashes and the shard index.
- Added the path template
  `configs/phase3c_semantic_store_example_v1.json`, package documentation, and
  CPU tests for camera normalization, exact camera keys, frame deduplication,
  and wrapper shape contracts.
- Local verification with the bundled Python runtime: 12 Phase 3C tests passed
  (one torch-dependent wrapper test skipped because local Python has no torch),
  `compileall` passed, and `git diff --check` reported no whitespace errors.
- This milestone does not claim that SSH extraction has completed. The real
  camera keys, DecisionNCE package/checkpoint, HDF5 mapping, and one-episode
  render smoke remain required before the store is accepted as a completed
  artifact.
## 2026-08-17 — Phase 3C Milestone 3 controlled CLaD wrapper implementation

- Added `scripts/phase3c/models/semantic_clad.py` with `ControlledCLaD` and a
  typed `CLaDBatch` contract. The original `baseline_code.LatentDynamics` is
  left untouched; the wrapper enforces `v_history=[B,2,2,D]`, `p_history=[B,2,16]`,
  `past_action=[B,6,7]`, `language=[B,D]`, and target-only training inputs.
- `encode_foresight` temporarily enters eval mode, passes `action_mask_ratio=0`,
  never reads target/future tensors, and returns `[B,2D]`. The training path
  requires the four original CLaD losses and the explicit post-optimizer EMA
  update is exposed as `update_ema_after_optimizer_step`.
- Added wrapper tests for dimension rejection, synthetic loss/foresight shape,
  EMA call order, and target-free inference. Local bundled-Python verification:
  15 Phase 3C tests passed with four torch-dependent tests skipped because the
  local runtime has no torch; compileall and diff checks passed.
- The H=1024 real CLaD smoke remains an SSH-only gate and has not been claimed.
## 2026-08-17 — Phase 3C Milestone 4 structured model implementation

- Added a shared `GraphBatch`/`StructuredBatch` tensor contract and five
  structured encoders: `SceneSetPastAct`, `PairPastAct`, `GeomMPNNPastAct`,
  `RelPoolPastAct`, and `RelMPNNPastAct`. All consume exactly the two graph
  snapshots plus `[B,6,7]` past action; no future graph/action field is part of
  their forward interface.
- The GeomMPNN path uses temporal geometry/contact only. RelPool and RelMPNN
  share the exact temporal relation-token encoder; RelPool performs masked
  edge-token pooling without message passing, while RelMPNN adds two residual
  message-passing layers. This is the planned relation-token fairness control.
- Added `models/adapters.py` with a shared semantic/structured projector,
  relation-change head, and scene-motion head so candidate comparisons do not
  change output head or fusion dimensions.
- Added CPU-skippable tests for common output shape, RelPool permutation
  invariance, and GeomMPNN insensitivity to relation-channel changes. The local
  environment has no torch, so these runtime tests are deferred to SSH; import
  compilation and the remaining 18 Phase 3C tests pass.
- This milestone implements model interfaces and invariance guards only; no
  GPU training, parameter matching, or performance claim has been made.
## 2026-08-17 — Phase 3C Milestone 5 dataset/trainer implementation

- Added `scripts/phase3c/dataset.py`: immutable semantic-store lookup,
  deterministic node ordering, Phase 2D task-slot guard, 16-D proprio
  extraction (`joint_pos[7] + joint_vel[7] + gripper_qpos[2]`), temporal geometry,
  contact/relation value-valid tensors, and `Phase3CBatch` collation.
- Added `losses.py` and `metrics.py`. Relation BCE is masked by
  `relation_valid`; unknown relations never become negative examples. Motion
  uses a separate smooth-L1 term, and metric helpers return `null`/`None` for
  single-class relations rather than inventing PR-AUC/F1 values.
- Added `train_base_clad.py` and `train_core.py` with fixed action/history
  schema, atomic checkpoints, source/checkpoint hashes, seed/device/runtime
  manifests, frozen-base core training, and common adapter/head wiring.
- Local verification: 20 Phase 3C tests collected, 11 dependency-free tests
  passed and 9 torch-dependent tests skipped because local Python has no torch;
  compileall and diff checks passed. No GPU training result is claimed.
- Added `run_core.py`, `analyze_core.py`, and `parameter_match.py` interfaces
  for sequential model/fold/seed execution, paired task bootstrap, and explicit
  trainable-parameter accounting. `run_core` treats `test_taskN` as a held-out
  task filter rather than silently reusing that task in training.
- Added smoke/full config templates for the semantic store, base CLaD, one core
  model, and the six-model screen. The templates use `$GRAPH_CLAD_ARTIFACT_ROOT`
  and intentionally require real SSH paths and completed base checkpoints.
- Core trainer now emits atomic checkpoint, prediction, metrics, and runtime
  artifacts; evaluation is restricted to held-out test task when the fold name
  encodes `test_taskN`. These are implementation interfaces only; no run has
  been executed or performance claim has been made.
- Added `run_base_clad.py` so the three held-out task folds can be trained with
  one explicit command; `test_taskN` is converted into an exclusion filter for
  base training. Added the corresponding 25K-update screen template.
- Added train-only `NormalizationStats` for node continuous channels, proprio,
  and relative geometry/distance; the stats are saved in the base checkpoint and
  reused by core training. Binary validity/task-reserved channels are not
  normalized.
- Tightened semantic-store extraction to enforce one global camera frame shape
  and batch both configured views through DecisionNCE. No fallback camera or
  orientation is introduced.
## 2026-08-17 — SSH Gate 0 Phase 3C test result

- After pulling the implementation to SSH and activating the project
  environment, all 23 `test_phase3c_*.py` tests passed in 1.608 seconds.
- Unlike the local desktop runtime, SSH executed the PyTorch-dependent CLaD,
  dataset, and structured-model tests successfully. This confirms the basic
  tensor shapes, EMA/wrapper behavior, graph permutation guards, masked metric
  behavior, and causal future-action poison test at the unit-test level.
- Gate 0 is therefore **passed**. This is not yet evidence that the real HDF5,
  camera, DecisionNCE, or full training pipeline works; the next gate is the
  joined-manifest and action-timing data smoke.

## 2026-08-17 Phase 3C pre-run full code audit and correction

- The first real joined-manifest attempt failed with `expected tau=6, got 1`.
  The canonical Phase 2D artifact intentionally interleaves horizons 1, 3, and
  6. The joiner now filters the requested horizon, counts the ignored 1/3-step
  samples, and has a mixed-horizon regression test. No successful real join is
  claimed until the corrected command is rerun on SSH.
- Corrected the graph tensor contract. Each `GraphBatch` now stores one
  snapshot (`contact=2`, seven non-contact relations `=14`) and the model forms
  prev/current/delta once. The old code duplicated contact and left half of the
  relation tensor zero. Node input now matches the written protocol:
  type-one-hot 4 + position 3 + validity 1 = 8 dimensions. The Phase 2D
  24-dimensional vector is used only to audit the forbidden task slot. Robot
  joints/gripper remain in the common proprio branch and are not duplicated in
  the graph branch.
- Fixed executable-path defects: NumPy semantic arrays are converted before
  `torch.stack`, target proprio no longer tensorizes the future graph, isolated
  nodes are retained, MPNN messages aggregate incoming rather than outgoing
  edges, invalid proprio/motion/edge endpoints fail, and semantic shard file
  handles use a bounded LRU cache.
- Replaced whole-manifest RAM loading and deterministic repeated ordering with
  a bounded-memory seeded shuffle. Validation/test evaluation also streams.
  Runtime manifests now bind config, joined-manifest, semantic-store, and base
  checkpoint hashes; completed matching runs can resume safely.
- Fixed evaluation leakage and planned loss behavior: relation eligibility and
  capped positive weights use train/validation support only; F1 thresholds are
  selected on non-held-out validation and frozen for held-out test; motion is
  trained with a train-only RMS scale and reported back in meters. Hierarchical
  bootstrap now resamples paired samples within sampled tasks.
- Core training now evaluates non-held-out validation at a frozen interval,
  restores the best validation PR-AUC state, and applies the configured
  patience/minimum-update early-stop rule before evaluating held-out test once.
- Restored the planned action-only semantic adapter, residual gated fusion, and
  automatic ±5% trainable-parameter matching against RelMPNN width 128. The
  previous zero branch and concatenation head did not implement the written
  six-model comparison.
- Froze the full-screen optimizer at AdamW `lr=3e-4`, `weight_decay=1e-4` and
  made shuffle, loss weights, support thresholds, validation split, and
  parameter-matching reference explicit in the example configs rather than
  relying on hidden defaults.
- Semantic extraction now requires the DecisionNCE repository commit,
  checkpoint SHA, and frozen simulator restore tolerance; it caches HDF5 hashes,
  validates both views/language, and hashes each feature shard. Semantic-store,
  base/core checkpoint, runtime, and screen schemas were bumped to v2 so no
  pre-audit artifact can be silently reused.
- A richer node schema (orientation, size/AABB, fixture unary state) is reserved
  for a later `RichNode` ablation after availability QA. It is not mixed into
  the six-model core because that would confound graph-structure gain with
  additional state features.
- Local verification after correction: 30 Phase 3C tests collected; 17 passed
  and 13 PyTorch-dependent tests were skipped in the desktop runtime. Full
  package compilation and `git diff --check` passed. The earlier SSH 23/23 Gate
  0 result predates these corrections, so the updated 30-test suite must be
  rerun on SSH before the real join is retried.

## 2026-08-17 SSH Gate 0 rerun after Phase 3C audit

- After pulling the audited Phase 3C implementation, the updated SSH test suite
  passed **30/30** tests, including all PyTorch-dependent dataset, collation,
  structured-model, semantic-adapter, and parameter-matching regressions.
- The post-audit Gate 0 is therefore **passed**. This supersedes the earlier
  23/23 result, which covered the pre-audit tensor and trainer implementation.
- No real joined-manifest success is claimed yet. The next action is to rerun
  the interrupted real-data join and confirm that mixed `tau=1/3/6` samples are
  filtered to `tau=6`, with a positive joined count and a nonzero
  `ignored_other_tau_samples` count in QA.

## 2026-08-17 Phase 3C join command-path correction

- The first retry of the real join stopped before reading data because the
  previously suggested `configs/phase3c_kcloudvpn_data_smoke_v1.json` is not a
  committed file in the repository. This was a command/config-path error, not
  a data or implementation failure.
- The repository provides `configs/phase3c_contract_v1.json` for the tau,
  relation, and causal-input contract only. On SSH, the three Phase 2D task
  shards and the joined-manifest output/QA paths must be supplied explicitly
  with repeated `--input`, `--output`, and `--qa-output` arguments.

## 2026-08-17 Phase 3C horizon-control decision

- Phase 2D contains samples with `tau=1/3/6`, and a future architecture could
  support multiple horizons. The current Phase 3C primary screen intentionally
  fixes `tau=6` so every C3 model and the original CLaD control solve the same
  six-step prediction problem with the same `[6, 7]` action window.
- Mixing horizons in the primary screen would change action-window length,
  target displacement, relation-change prevalence, class balance, and task
  difficulty at the same time as graph architecture. That would make a graph
  improvement impossible to attribute cleanly to graph structure.
- A multi-horizon extension is therefore reserved as a follow-up robustness
  experiment. It requires an explicit horizon field or embedding, padding/mask
  rules (or separate manifests), horizon-aware normalization and metrics, and
  re-running the leakage/parameter-matching tests. The current `tau=6` join is
  the controlled architecture-comparison benchmark, not a claim that other
  horizons are unusable.

## 2026-08-17 Phase 3C real joined-manifest QA passed

- The corrected SSH join completed with `status=pass` and wrote
  `/home/ubuntu/graphclad-artifacts/phase3c_oracle_graph_clad_v1/data_contract/joined_manifest.jsonl.gz`.
- The three Phase 2D task shards contained 51,471 samples. The builder selected
  16,757 `tau=6` left candidates and emitted 15,857 joined samples. The 900
  boundary drops are expected at episode/temporal boundaries; there were zero
  missing right samples, duplicate left keys, invalid samples, or graph-hash
  mismatches.
- 34,714 `tau=1/3` samples were explicitly ignored, confirming the primary
  `tau=6` horizon control. No future action field was emitted (`0`).
- Relation support is eligible without test leakage for `left`, `right`,
  `front`, `behind`, `above`, `below`, and `contact`. `on` has zero positives
  in train/validation/test and is therefore excluded from the eligible loss and
  model-selection relation set for this artifact.

## 2026-08-17 Phase 3C semantic-store dependency discovery

- The first SSH discovery attempt did not locate the raw task HDF5 files,
  LIBERO BDDL path, DecisionNCE import, or a DecisionNCE checkpoint from the
  current shell. The shell prompt did not show the project virtual environment
  as active, so Python-package absence must first be rechecked after activating
  `/home/ubuntu/Graph-CLaD/.venv`.
- Raw HDF5 and model-checkpoint discovery is independent of virtual-environment
  activation. If the filesystem searches remain empty, the server currently
  contains only the derived Phase 2D graph artifacts, not the original state
  replay inputs required to render semantic frames. Semantic-store extraction
  is blocked until those immutable inputs and their provenance are restored;
  graph JSON artifacts cannot substitute for the missing simulator states.
- Follow-up SSH checks confirmed `ModuleNotFoundError: No module named
  'libero'`, no raw HDF5 files under `/home/ubuntu`, and no locally discoverable
  DecisionNCE installation/checkpoint. `requirements-phase3c.txt` intentionally
  excludes LIBERO and DecisionNCE, so the earlier 30/30 Gate 0 established code
  contracts only; it did not establish real semantic-extraction readiness.
- The semantic-store gate is therefore blocked on external runtime assets:
  official task HDF5 demonstrations, a pinned LIBERO installation/BDDL root,
  and a pinned DecisionNCE repository/model artifact. The successful joined
  manifest remains valid and does not need to be rebuilt.
- SSH network access to both official repositories was verified. The observed
  remote HEADs were LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`
  and DecisionNCE `ebdc585c5e6833ec3a2ba77f801b15c74d7a28f8`.
  These are discovery values only; they must be checked out explicitly and
  recorded with downloaded dataset/model hashes before semantic extraction.
- Direct inspection of the pinned official DecisionNCE source exposed a real
  integration mismatch in the pre-run example: the import module is
  `DecisionNCE`, `load(name, device=...)` downloads to
  `~/.cache/DecisionNCE/<model-id>` and accepts no checkpoint keyword, and the
  returned encoder performs its own tensor transform instead of exposing a
  separate `preprocess` callable.
- The Phase 3C wrapper and example config were corrected to use the official
  module, keep the auto-downloaded checkpoint path for SHA provenance without
  passing it to the loader (`checkpoint_argument=null`), pass `device=cuda`,
  and provide normalized RGB `[0,1]` tensors (`preprocess=rgb_01`). A loader
  regression test was added. The dependency-light suite now collects 31 tests:
  17 passed and 14 torch-dependent tests skipped locally; compilation and
  `git diff --check` passed. The new loader test must run in SSH PyTorch after
  the correction is pulled.
- On SSH, both editable install commands completed and the pre-install pip
  snapshot was written, but the immediate import check still raised
  `ModuleNotFoundError: No module named 'libero'`. This indicates an editable
  package path/installation visibility issue rather than a joined-manifest
  failure; installed torch remained in the environment while DecisionNCE
  dependencies changed `timm` to 0.9.12 and installed its runtime packages.
- The follow-up import check showed `import libero` resolving as a namespace
  package (`__file__ is None`), which is not itself a failure; the nested
  `libero.libero` package must be checked. DecisionNCE import reached the
  official `clip` module but failed because the current setuptools removed the
  legacy `pkg_resources` module. The immediate compatibility fix is a
  setuptools version below 81, followed by nested LIBERO and DecisionNCE import
  checks.
- The SSH compatibility checks then passed: `DecisionNCE` imports after pinning
  setuptools below 81, and `from libero.libero import get_libero_path` resolves
  BDDL to `/home/ubuntu/external-src/LIBERO/libero/libero/bddl_files`. The
  `pkg_resources` deprecation warning is emitted by the pinned legacy CLIP
  dependency and is non-fatal. LIBERO initialized its config at
  `/home/ubuntu/.libero/config.yaml`; its default dataset directory is still
  empty, so the official `libero_spatial` HDF5 download is the next gate.
- The official `libero_spatial` dataset download then completed and produced
  ten demo HDF5 files under
  `/home/ubuntu/graphclad-artifacts/phase3c_oracle_graph_clad_v1/libero_datasets`.
  The task-id-to-file mapping is not inferred from alphabetical order; it will
  be obtained from LIBERO's benchmark task metadata before the semantic config
  is written.
- LIBERO benchmark metadata was queried successfully with the pinned source.
  The first three task IDs are now fixed: task 0 is
  `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`,
  task 1 is `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`,
  and task 2 is `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
  These names match the corresponding downloaded HDF5 demo filenames exactly;
  this removes the task-order ambiguity for the Phase 3C semantic-store config.
- The first Phase 3C action/state timing smoke was blocked before data access by
  `ModuleNotFoundError: No module named 'h5py'` in the SSH virtual environment.
  This is an environment dependency gap for reading the already-downloaded HDF5
  files, not a protocol or dataset failure; install `h5py` in `.venv` and rerun
  the unchanged smoke command.
- After adding `h5py`, the timing smoke reached LIBERO environment construction
  and exposed the next omitted runtime dependency: `ModuleNotFoundError: No
  module named 'robosuite'`. The LIBERO dependency manifest pins
  `robosuite==1.4.0`; install that package in the existing SSH `.venv` without
  reinstalling the full legacy requirements (which would risk replacing the
  working Torch environment), then rerun the same smoke command.
- `robosuite==1.4.0` installed successfully, but its import now stops in the
  OpenCV renderer because the Ubuntu host lacks the system library
  `libGL.so.1`. The preceding `macros_private` message is handled by
  robosuite's fallback import; the actionable failure is the final OpenCV
  `ImportError`. Install the host `libgl1` package (or use headless OpenCV if
  sudo is unavailable) and rerun the import check.
- After `libgl1` was added, the import progressed to MuJoCo's headless EGL
  backend and failed with `OpenGL.EGL` reporting a null platform. This is an
  EGL initialization/configuration issue, not a missing Python module. Set
  `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` before importing LIBERO and
  ensure the host has the runtime EGL/GLES libraries (`libegl1`, `libgles2`);
  retain OSMesa only as a slower fallback if the NVIDIA EGL path is unavailable.
- With the EGL environment active, the timing validator reached HDF5 opening
  and failed because the configured task-0 demo file was absent at
  `/home/ubuntu/graphclad-artifacts/phase3c_oracle_graph_clad_v1/libero_datasets`.
  This confirms the Python/simulator imports are now past the previous gates;
  the next action is to locate or redownload the official `libero_spatial`
  HDF5 files into that exact artifact directory. Installing the full legacy
  LIBERO requirements is not a substitute and risks changing the working
  Torch environment.
- The subsequent Hugging Face download completed all 10 `libero_spatial` files
  successfully. LIBERO's downloader stores them one level deeper than the
  parent download directory, at
  `/home/ubuntu/graphclad-artifacts/phase3c_oracle_graph_clad_v1/libero_datasets/libero_spatial`;
  the validator's previous paths omitted this `libero_spatial` component. The
  consolidated dependency command also hit an optional `robomimic` build failure
  because `egl_probe` requires system `cmake`; this is separate from the HDF5
  download and is not needed for the immediate state-replay validator unless a
  later import explicitly requires `robomimic`.
- The shell subsequently showed an empty `libero_spatial` directory despite the
  downloader having just reported 10 files and a complete dataset. Treat this as
  a path/persistence discrepancy: verify the canonical path with `pwd -P`,
  `readlink -f`, recursive `find`, and disk usage before downloading again.
- The canonical path check confirmed the discrepancy is real: the directory is
  `/home/ubuntu/graphclad-artifacts/phase3c_oracle_graph_clad_v1/libero_datasets/libero_spatial`
  and occupies only 4.0K, with zero HDF5 files anywhere under the Phase 3C
  artifact root. The next recovery is a targeted Hugging Face download of only
  task 0/1/2 HDF5 files, followed by an immediate count and size check, rather
  than another unverified full-suite download.
- The targeted download produced the three required HDF5 files (approximately
  509MB, 590MB, and 672MB). The timing smoke now reaches LIBERO simulator reset,
  but EGL cannot open `/dev/dri/renderD129` because the SSH user lacks render
  device permission. This is an OS group/ACL issue; add the user to the
  `render` group and start a new group session before retrying GPU EGL.
- After the render-group change, the device permission error disappeared, but
  Mesa EGL still failed to create the required `EGL_PLATFORM_DEVICE_EXT` display
  (`egl: failed to create dri2 screen`). This is a driver/backend limitation,
  not a dataset problem. For the no-image timing smoke, use MuJoCo's OSMesa
  software backend; reserve GPU EGL troubleshooting for the later semantic
  image feature extraction stage.
- OSMesa initialization then succeeded, exposing a separate simulator API
  mismatch: `robosuite==1.4.0` calls the legacy `mujoco.MjData.qM` field, while
  the unconstrained install had selected `mujoco==3.11.0`, where that field is
  absent. Pin MuJoCo to the 2.3.x line (use `mujoco==2.3.7`) without changing
  the Torch stack, then rerun the same OSMesa timing smoke.
- The SSH environment was corrected to `mujoco==2.3.7` and the timing smoke was
  relaunched with `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`. The run has
  reached active validation without a traceback; remaining robosuite private-
  macro and Gym notices are non-fatal warnings. Await the final JSON report.
- The OSMesa action-timing smoke completed with no runtime errors over 150
  transitions (`errors=[]`, execution `status=pass`). Because no tolerance was
  supplied, `within_tolerance` is correctly `null`; the measured mean
  max-absolute state error is `0.0349203188` and the worst transition is
  `0.3311892104`. These values are diagnostic only until task/demo-level error
  quantiles and the train/validation-only freeze rule are inspected; do not set
  the semantic-store tolerance directly to the global maximum yet.
- The detailed report shows the first HDF5/task has substantially larger
  outliers: mean `0.0450849717`, maximum `1.8350234838` at `demo_45` step 0,
  followed by `0.9970277089` at `demo_18` step 0. The second task peaks at
  `0.2457675948` (mean `0.0151608390`) and the third at `0.3311892104` (mean
  `0.0349203188`). These outliers require split-aware train/validation
  quantiles and should not be silently absorbed by a global max tolerance.
- Split-aware aggregation gives train `n=339`, mean `0.0286199658`, p95
  `0.2236581006`, p99 `0.3019911628`, max `0.9970277089`; validation `n=57`,
  mean `0.0297684994`, p95 `0.1828297597`, p99 `0.2224259152`, max
  `0.2267141653`; test `n=54`, mean `0.0532582699`, p95 `0.1197738957`, p99
  `0.1233433740`, max `1.8350234838`. Thus a frozen tolerance near `1.05`
  would cover all observed train/validation transitions while flagging the
  isolated test outlier; this is a candidate pending the explicit tolerance
  rerun, not yet a final config value.
- Interpretation note: timing QA restores recorded `state[t]`, executes the
  recorded `action[t]` once, and compares the resulting simulator flattened
  state against recorded `state[t+1]`. The reported error is the maximum
  absolute difference over mixed state coordinates (time, joint positions,
  velocities, and related simulator state), not a model metric. `tolerance`
  is the allowed maximum for this QA gate; `1.05` is a conservative candidate
  driven by a train outlier near `0.997`, not evidence of high simulator
  fidelity. Large errors indicate possible action/state indexing, controller
  internal-state, action scaling, or simulator-version mismatch. They do not
  directly invalidate labels created by direct state restoration, but they
  weaken confidence in action-conditioned temporal alignment.
- Frozen-tolerance rerun (`1.05`) passed all 150 transitions for task 1 and all
  150 for task 2. Task 0 had 149/150 within tolerance and one test transition
  outside (`demo_45`, error `1.8350234838`); its report and the batch report are
  therefore `fail` by design. Train/validation remain covered by the frozen
  threshold, while the isolated test outlier is retained as an explicit QA
  finding rather than used to inflate tolerance.
- Terminology clarification: a DecisionNCE semantic feature-store **smoke** is
  a small, non-training preflight. It restores a few official LIBERO states,
  renders the two configured camera views, applies DecisionNCE-P image and task
  language encoding, and checks camera keys/orientation, embedding shape,
  finite values, checkpoint provenance, deterministic repeatability, and frozen
  state-restore tolerance. It creates a tiny artifact only to validate the
  preprocessing contract; the full feature store later processes every unique
  `(task, demo, step)` required by the joined manifest. This is needed because
  Phase 2D graph/action artifacts do not themselves contain the semantic image
  embeddings expected by the controlled CLaD input path.
- Restore rationale clarification: earlier geometric/action graph experiments
  consumed already-materialized Phase 2D graph tensors, so no simulator render
  or state restore was needed at model-input time. The semantic CLaD control now
  needs image/text embeddings aligned to the exact graph timestamps; because
  Phase 2D JSONL contains no RGB frames or DecisionNCE features, the official
  HDF5 state must be restored to render the matching `t-6`, `t`, and `t+6`
  observations. This is an input-alignment/data-preparation step, not a change
  to the original CLaD loss or backbone.
- Camera-input distinction: raw RGB is not an intrinsic requirement of the
  CLaD temporal-dynamics core. In the current C3-Sem-PastAct control, RGB is an
  upstream source for frozen DecisionNCE visual-language embeddings, so the
  semantic branch must render the two configured camera views from official
  HDF5 states. C3-SceneSet-PastAct, C3-Pair-PastAct,
  C3-GeomMPNN-PastAct, C3-RelPool-PastAct, and C3-RelMPNN-PastAct operate on
  graph/proprioceptive/action features and remain camera-free. Thus
  the camera requirement belongs to the chosen semantic input path and its
  provenance/alignment check, not to every CLaD or graph model.
- Semantic feature-store smoke attempt stopped before rendering with
  `joined manifest contains no frame keys`. The preceding filter explicitly
  reported `smoke records: 0`, so this is an empty derived-manifest failure,
  not a DecisionNCE, camera, or CLaD model failure. The likely cause is a demo
  key spelling mismatch such as `demo0` versus the HDF5/manifest convention
  `demo_0`. Rebuild the smoke manifest by selecting the first actual task-0
  `demo_key` from the full joined manifest and require a nonzero count before
  rerunning extraction.
- Follow-up: the rebuilt filter printed a blank `selected demo` and 4,468
  records, indicating that the joined records may carry an empty `demo_key`;
  this is not a valid one-demo smoke selection. Do not run extraction on this
  file. Inspect `episode_id`/`demo_key` first, then derive a single HDF5 group
  key (for example `demo_0`) from a valid episode if the upstream field is
  missing, and preserve the corrected key in the smoke manifest.
- Code correction: `scripts/phase3c/build_joined_manifest.py` now resolves a
  missing/blank `demo_key` from `demo_id` or an episode suffix such as
  `task0_demo0 -> demo_0`, and emits the repaired key in the joined record.
  A regression test was added. The already-existing joined artifact must be
  rebuilt after this patch; the old artifact remains invalid for semantic
  feature extraction because its demo keys are blank.
- Semantic feature-store smoke completed successfully for the corrected
  `task0_demo0 -> demo_0` one-episode manifest. Output status was `completed`,
  with one shard and DecisionNCE feature dimension `1024`. This confirms the
  HDF5 state restore, two configured camera views, DecisionNCE-P image/text
  encoding, shard writing, and provenance path are operational for the smoke
  case; it is not yet a full-dataset extraction or model-training result.
- Full-store check after the smoke reported that
  `phase3c_oracle_graph_clad_v1/semantic_store/manifest.json` or
  `semantic_store/0/demo_0.npz` is absent. Therefore only the one-shard smoke
  artifact is confirmed so far; existing full shards, if any, must be located
  and their manifest/source checked before C3 training.
- Operational clarification: generating `semantic_store_full_config.json` does
  not validate or read the fixed joined manifest; it only writes configuration
  paths. Therefore step 2 can report success even when step 1 has not created
  `joined_manifest_full_demo_fixed.jsonl.gz`. Full extraction remains pending
  until the manifest-repair step and the semantic-store command both complete.
- Full semantic feature-store extraction was started on SSH using the repaired
  full joined manifest and `semantic_store_full_config.json`. DecisionNCE RN50
  loaded successfully; repeated LIBERO task-order and Gym/robosuite warnings
  observed during environment creation are non-fatal so far. Final completion
  status and shard count are pending.
- The SSH terminal output for the full extraction was not retained in the
  visible scrollback; final status is therefore to be recovered from the
  immutable output `semantic_store/manifest.json` and shard inventory rather
  than rerunning the extraction blindly.
- Full semantic feature-store extraction completed successfully. Artifact
  verification reported `status=completed`, `manifest shards=150`,
  `actual npz files=150`, `feature_dim=1024`, and the expected
  `0/demo_0.npz` shard exists. The full two-camera DecisionNCE store is now
  available for the controlled CLaD baseline and core model screen.
- Pre-training audit conclusion: the 100-update technical smoke may be used to
  test executability, but full performance training is not yet protocol-ready.
  Blocking discrepancies are (1) semantic extraction wrote camera inventory
  only, while the design requires an orientation contact sheet and determinism
  report; (2) action-timing QA remains overall `fail` because task-0 test
  `demo_45@0` exceeded the frozen tolerance; (3) the corrected demo-key
  manifest/config is an ad-hoc SSH artifact while the builder fix and regression
  test remain local/unverified; and (4) base CLaD training currently saves only
  `last.pt`, whereas the frozen plan specifies selection by minimum validation
  Stage-1 loss.
- Additional audit findings: core budget is inconsistent across plan/config
  (10,000 versus 3,000 updates), planned AMP/deterministic/runtime reporting and
  artifact/end-to-end tests are absent, example configs still point to the old
  blank-demo manifest, and environment commits/package pins are not enforced
  by code. Scientifically, oracle graph versus RGB semantic CLaD is not an
  information-matched architecture-only contrast; inverse relation pairs
  should not be treated as independent evidence, `on` is unsupported, and a
  final action ablation plus more than seed 0 is required for strong claims.
- Camera-orientation audit: vertical image flipping is a real upstream
  convention issue, not merely a hypothetical wrapper bug. Official robosuite
  1.4.0 defaults to `IMAGE_CONVENTION="opengl"`, while the OpenCV convention
  is the explicitly selected unflipped alternative. The current Phase3C
  config sets `vertical_flip=false`, but the extractor does not inspect or
  override robosuite's upstream convention and therefore this flag alone does
  not prove that stored frames are upright. The completed semantic-store
  extraction is consequently an extraction-success result, not yet a visual
  orientation-QA pass. Before performance training, inspect HDF5 convention
  metadata and runtime macro values and generate a representative contact
  sheet/determinism check; regenerate the store only if that check finds a
  mismatch.
- Camera-convention metadata check on SSH: robosuite runtime reported
  `IMAGE_CONVENTION=opengl`, and all three Phase3C LIBERO-Spatial HDF5 files
  reported `macros_image_convention=opengl`. Thus there is no recording-versus-
  replay convention mismatch for these files. This does not establish that the
  arrays are upright for a conventional vision encoder: both sides use the
  native OpenGL convention, while the current extractor applies no corrective
  flip (`vertical_flip=false`). A rendered contact sheet remains the decisive
  gate. The accompanying `libEGL` DRI2/KMS software-renderer fallback warnings
  did not prevent the metadata check and are treated as non-fatal.
- The pre-fix Base CLaD 100-update three-fold smoke completed on SSH under
  schema `phase3c-base-clad-run.v2`. All losses were finite and runtimes were
  approximately 74.1s, 36.9s, and 38.2s for held-out tasks 0, 1, and 2. These
  runs reported mean-last-100 losses 0.212357, 0.202391, and 0.194557,
  respectively. The fixed joined-manifest SHA-256 was
  `3a39178376114d0a03dfd9dff8d35b691f0a69abdc3cf11e3bca2810e25ae3bd`.
  These
  artifacts are retained only as GPU executability evidence because they save
  `last.pt` and perform no validation checkpoint selection; they are not valid
  base checkpoints for the Phase3C core screen.
- Phase3C pipeline hardening implemented locally pending SSH verification:
  the canonical builder now repairs blank demo keys and emits causal-join v2
  provenance; all Phase3C example configs point to
  `joined_manifest_full_demo_fixed.jsonl.gz`; camera QA renders configured and
  vertically flipped contact-sheet alternatives plus repeat-render hashes;
  Base CLaD selects `best.pt` by minimum validation Stage-1 loss while keeping
  `last.pt` for resume; Core uses the frozen 10,000-update maximum with its own
  best/resume split; completed runners verify artifact hashes; and metrics now
  report inverse-pair-aware relation-family macros alongside per-relation
  macros. Dependency-light local tests pass, while torch-dependent trainer and
  resume tests remain to be executed on the SSH environment.
- New Base/Core smoke and full-screen configs use `*_v3` output roots, so the
  old v2 `last.pt` smoke artifacts cannot be mistaken for resumable v3 runs or
  overwritten by the new trainers. Core rejects any base checkpoint that is
  not explicitly schema v3 and `kind=validation_best`.
- To avoid silently relabeling the one-off fixed gzip as builder output, a
  streaming migration-attestation command was added. It compares the existing
  fixed manifest with a fresh builder-v2 candidate using ordered canonical JSON
  row SHA-256 (not gzip container bytes), records both raw hashes and row counts,
  and permits reuse of the existing semantic store only when they are exactly
  equivalent. A mismatch fails and requires investigation/rebuild.
