# Graph-CLaD 현재 상태와 실행 인계

기준 시각: 2026-08-17 (Asia/Seoul)
공식 현재 단계: **Phase 3B action-alignment gate 판정 완료 — 실패, Phase 3C Milestone 1 구현 중**
이 문서는 다음 세션에서 가장 먼저 읽는 단일 현재상태 문서다. 공식 연구 질문과 단계별
gate는 `01-plan/features/graph-clad-integrated-research-v4.plan.md`, 시간순 근거는
`research_log.md`를 따른다. `revised_research_roadmap_v3.md`는 v4 이전의 단계 개정
근거로 보존한다.

## 1. 한 줄 상태

Pair-local H0–H3 three-fold seed-0 screen은 12/12 runs가 끝났고 H3의 natural
PR-AUC가 가장 높았다. 이어서 episode-disjoint matched train-shuffled H3 control
3/3 runs를 완료했지만 aligned H3가 shuffled H3보다 우세한 task는 1/3뿐이었다.
사전 정의한 2/3-task 기준에 미달하므로 action-alignment gate는 실패했고 H3를 최종
representation으로 확정하지 않는다.

## 2. 현재까지 확정된 결과

Task-macro natural 결과는 다음과 같다.

| 모델 | PR-AUC | F1 | Release F1 | Hard-negative FPR |
|---|---:|---:|---:|---:|
| H0 state | 0.3626 | 0.3134 | 0.0931 | 0.3168 |
| H1 history | 0.4348 | 0.4920 | 0.0215 | 0.3358 |
| H2 action | 0.3941 | 0.3186 | 0.1729 | 0.2241 |
| H3 history+action | **0.4824** | 0.3617 | 0.1193 | 0.2339 |

- H3−H1 PR-AUC는 +0.0476이며 3/3 tasks에서 양수다.
- H3−H1 release F1은 +0.0978, hard-negative FPR은 −0.1019다.
- Task 0에서 H3 hard-negative FPR이 0.4762로 높으므로 안전성 문제는 남아 있다.
- 상세 task별 값과 SHA는 `phase3_pair_local_temporal_threefold_seed0_result.md`에 있다.
- 위 결과는 one-seed architecture screen이며 일반화 결론이 아니다.

## 3. KCloudVPN 실행 환경

| 항목 | 현재 값 |
|---|---|
| SSH | `ubuntu@172.10.5.118` |
| 서버 repository | `/home/ubuntu/Graph-CLaD` |
| 가상환경 | `/home/ubuntu/Graph-CLaD/.venv` |
| artifact root | `/home/ubuntu/graphclad-artifacts` |
| 실제 GPU | NVIDIA GeForce RTX 3090, 24 GB |
| driver / 표시 CUDA | `595.71.05` / `13.2` |
| PyTorch | `2.13.0+cu130` |
| Python | `3.10.12` |

실행마다 실제 값은 `nvidia-smi`, PyTorch preflight, 결과의
`runtime_manifest.json`으로 다시 기록한다.

## 4. 서버 입력 artifact

다음 입력이 서버에 복사되어 존재함을 확인했다.

| 경로 (`$GRAPH_CLAD_ARTIFACT_ROOT` 기준) | 크기 | 역할 |
|---|---:|---|
| `phase2d/data/phase2d_full_demo_v2_inputclean_stream1` | 836 MB | natural sample과 causal history 재구성 |
| `phase2d/data/phase2d_holding_target_v2_inputclean_stream1` | 357 MB | target-aligned train/stress rows |
| `phase2d/data/phase2d_demo_split_manifest.json` | 76 KB | episode split 고정 |

Corrected evaluation manifest는 다음 위치에 있다.

`/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/phase3B_R1_eval_manifest_v2.json`

검증 결과는 `status=pass`, `folds=3`이다. 서버에서 새 manifest를 만들려던 process는
`Killed`로 종료됐다. 현재 builder가 압축 해제된 natural/target graph payload 전체를
동시에 메모리에 보관하는 eager 구현이어서 OOM이 발생한 것으로 판단한다. 같은 명령을
반복하지 않는다. 기존 Colab manifest를 복사한 뒤 source root만
`/home/ubuntu/graphclad-artifacts`로 바꿨으며, 원본은 같은 디렉터리의
`phase3B_R1_eval_manifest_v2_colab_original.json`으로 보존했다. Sample key, fold,
payload hash, QA 결과는 바꾸지 않았다.

## 5. 완료된 action-alignment control

Config:

`configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json`

Output:

`/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1`

범위는 H3 train-shuffled 1개 모델 × 3 folds × seed 0으로 총 3 runs이며, 2026-08-16
KCloudVPN에서 3/3 runs를 완료했다. Runner 최종 stdout은 `status=completed`, `runs=3`과
위 output 아래의 `runtime_manifest.json` 경로를 기록했다. 완료 artifact가 있으므로 같은
config를 재실행하지 않는다.

## 6. 완료 artifact와 비교 입력

완료 artifact의 유지 조건은 다음과 같다.

1. Result JSON의 `status`가 `completed`다.
2. Result의 `results` 길이가 3이다.
3. `runtime_manifest.json`이 있고 RTX 3090/CUDA/config SHA/manifest SHA가 기록됐다.
4. `checkpoints/`에 fold별 checkpoint 3개가 있다.
5. `predictions/`에 fold별 per-sample prediction 3개가 있다.
6. `code_snapshot/`이 있다.
7. stderr 또는 tmux 출력에 traceback이 없다.

Aligned H3의 기준 결과는 Colab Drive의 다음 root에 있다.

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/pair_local_temporal_threefold_seed0_v1`

Alignment runner는 shuffled H3만 학습했다. Config의 `comparison_source` 문자열이
aligned result를 자동으로 불러와 비교해 주는 것은 아니다. 후속 same-fold/seed
hierarchical bootstrap과 상세 보고에는 위 aligned H3 result와 H3 prediction artifact를
읽거나 서버로 별도 전송해야 한다. 기존 checkpoint 전체는 비교에 필요하지 않다.

## 7. Phase 3B gate

Natural conditional/oracle-current event 결과는 다음과 같다.

| Task | Aligned H3 PR-AUC | Shuffled H3 PR-AUC | Aligned−Shuffled | Aligned F1 | Shuffled F1 | Aligned release F1 | Shuffled release F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4009 | 0.4245 | −0.0236 | 0.4074 | 0.4524 | 0.2353 | 0.4000 |
| 1 | 0.5731 | 0.5059 | +0.0672 | 0.3357 | 0.7086 | 0.1075 | 0.5882 |
| 2 | 0.4734 | 0.4913 | −0.0179 | 0.3419 | 0.6034 | 0.0150 | 0.3733 |
| Task macro | 0.4824 | 0.4739 | 약 +0.0085 | 0.3617 | 0.5881 | 0.1193 | 0.4539 |

Paired hierarchical bootstrap 2,000회 결과는 aligned minus shuffled 기준으로 다음과 같다.

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Event PR-AUC | +0.0085 | [−0.0506, +0.0772] |
| Event F1 | −0.2264 | [−0.3955, −0.0459] |
| Release F1 | −0.3346 | [−0.5339, −0.1016] |
| Hard-negative FPR | +0.0057 | [−0.0796, +0.1007] |

Aligned H3의 task-macro PR-AUC는 약간 높지만 task별 우세는 1/3이라 사전 정의한 최소
2/3-task 기준에 미달한다. Release F1도 aligned가 3/3 tasks에서 낮다. 따라서 gate는
**실패**다. H3/H1/H3-train-shuffled seeds 1/2 확대를 중단하고 pair-local 결과는
architecture finding으로만 보고한다. 올바르게 정렬된 action의 causal/semantic 효과를
입증했다고 주장하지 않는다.

- 90-item weak-label human review는 0/90 상태에서 일시 보류한다. Phase 3C technical
  smoke는 진행할 수 있지만 label validity나 작은 성능 차이에 대한 최종 주장은 audit
  완료 전까지 금지한다.
- Natural test가 primary이며 challenge는 독립 test가 아닌 stress analysis다.
- Conditional event metric은 oracle current holding을 사용하므로 end-to-end metric도 함께 보고한다.

## 8. 이후 전체 CLaD 계획

사용자가 선택한 제출 전 기본 방향은 다음과 같다.

1. 최종 목표는 controlled semantic CLaD 대비 Graph-CLaD improvement이며, 최종 주장은
   같은 Stage 2 policy/budget의 paired rollout success로 판단한다.
2. Phase 3C core는 `C3-Sem-PastAct`, `C3-SceneSet-PastAct`, `C3-Pair-PastAct`,
   `C3-GeomMPNN-PastAct`, `C3-RelPool-PastAct`, `C3-RelMPNN-PastAct`다. 모든 모델은 같은
   `<=t` causal history와 `a[t-tau:t]` past action을 받는다. `RelPool`은 RelMPNN과 동일한
   relation edge token에서 message passing만 제거한 필수 fairness control이다. Transformer
   두 cell은 core 결과 후 secondary로 둔다.
   기존 Phase 2D의 연속 `(t-tau -> t)`와 `(t -> t+tau)` sample을 join해 새 replay 없이
   구성하며, 공유 `graph[t]` hash와 episode/split/tau 일치를 먼저 검사한다.
3. Phase 3C primary는 `tau=6` sample-level valid spatial-relation any-change task-macro
   PR-AUC이며, primary contrast는 `RelMPNN−controlled semantic CLaD`다.
   `RelMPNN−RelPool`은 exact-token message-passing guard, `RelMPNN−SceneSet`은 broader
   scene-state guard다. Primary relation은 현재 구현된 `left/right`, `front/behind`,
   `above/below`, `contact`, `on`만 사용한다.
4. Object displacement와 source→destination은 secondary다. Holding은 audit 전 loss,
   checkpoint, threshold, model selection과 gate에서 제외하고 diagnostic으로만 저장한다.
5. Relation model이 유망할 때만 선택된 relation encoder의
   no-action/shuffled-past-action control을 실행한다. `RelPool`은 이미 core에 포함한다.
6. Future action `a[t:t+tau]`과 future graph는 모든 Phase 3C 입력에서 금지한다.
7. Phase 4에서 선택 representation을 CLaD Stage 1 latent foresight에 통합한다.
8. Stage 1을 freeze하고 canonical DDPM Diffusion Policy를 Stage 2로 연결한다.
9. Policy-only, controlled semantic CLaD, semantic+SceneSet, semantic+RelPool,
   semantic+RelMPNN Graph-CLaD를 같은 policy 용량과 학습 budget으로 비교한다. RelPool은
   policy 수준의 exact-token/no-message control이다.
10. Action chunk horizon은 원 논문 설정과 맞춰 `tau=6`을 우선 사용한다.
11. Current observation과 predicted foresight를 modality별 FiLM으로 결합하고 표준
   epsilon-prediction objective를 사용한다.

Weak-label audit 완료는 별도 encoder를 추가하는 조건이 아니다. Audit가 holding label을
지지하면 같은 frozen representations에 holding onset/release probe를 정식 secondary
target으로 추가한다. Audit 전 Phase 3C에서는 holding을 diagnostic으로만 기록하며
checkpoint나 확대 결정에 사용하지 않는다.

저장소에는 공식 Stage 2 코드와 rollout pipeline이 없다. 따라서 이 구현은 논문 설명을
따른 **CLaD-compatible controlled reimplementation**으로 표현하고 공식 재현이라고
주장하지 않는다. 현재 데이터도 LIBERO-LONG 10-task 공식 protocol과 다르므로 논문의
94.7%와 직접 비교하지 않는다. RTX 4090에서 Stage 2 200K steps가 약 20시간이라는 논문
기준을 고려하면 RTX 3090에서는 one-task smoke 후 reduced budget 비교를 먼저 확보한다.

## 9. 아직 해결되지 않은 항목

- Action-alignment end-to-end/challenge/calibration 상세 보고와 analysis artifact SHA 기록.
- 90-item weak-label human review: 0/90, Phase 3C technical smoke 동안 일시 보류.
- Phase 3C Milestone 1의 SSH 실제 HDF5 action-timing/data smoke.
- DecisionNCE semantic feature store, CLaD Stage 1 wrapper, Phase 3C six-model implementation.
- Phase 3C/4 구현과 notebook.
- Stage 2 Diffusion Policy, LIBERO rollout environment, success-rate evaluator.
- Semantic/VLM observation pipeline과 실제 CLaD Stage 1 trainer의 미공개 세부사항.
- `inside` label support.

## 10. 다음 세션에서 읽는 순서

1. 이 파일 `docs/CURRENT_STATUS.md`.
2. `docs/NEXT_SESSION_PROMPT.md`.
3. `docs/01-plan/features/graph-clad-integrated-research-v4.plan.md`의 Phase 3C 이후.
4. `docs/01-plan/features/phase3c-oracle-graph-clad-core.plan.md`.
5. `docs/02-design/features/phase3c-oracle-graph-clad-core.design.md`.
6. `docs/phase3_pair_local_temporal_action_alignment_seed0_result.md`.
7. `docs/phase3_pair_local_temporal_threefold_seed0_result.md`.
## 2026-08-17 implementation update

Phase 3C Milestone 2 is now implemented locally. The semantic feature-store
builder, exact two-camera contract, frozen DecisionNCE wrapper, per-demo shard
schema, and CPU contract tests are present. SSH extraction has not been run;
the real HDF5 mapping, BDDL roots, camera keys, DecisionNCE installation, and
checkpoint must be filled and smoke-tested there. CLaD wrapper and structured
six-model adapters remain the next implementation milestones.

Latest implementation update: Milestones 3–5 are now also present locally:
controlled CLaD wrapper, dataset/tensorizer, five structured encoders plus the
semantic control, masked losses/metrics, base/core trainers, fold runners, and
paired analyzer. This does not supersede the SSH gates: torch/LIBERO/DecisionNCE
imports, real camera inventory, H=1024 smoke, parameter/runtime checks, and
actual training artifacts are still unverified until you run them on SSH.

## 2026-08-17 Phase 3C audit correction status

The first real join exposed mixed Phase 2D horizons (`tau=1/3/6`). The corrected
join selects only `tau=6`. A subsequent full pre-run audit also corrected the
8-D node schema, per-snapshot edge dimensions, NumPy-to-torch collation,
incoming message aggregation, bounded streaming/shard caches, train-only
support and motion scaling, validation-fixed thresholds, parameter matching,
provenance hashes, and safe resume. Local dependency-light verification passes,
but the revised 30-test suite has not yet been rerun in the SSH PyTorch
environment; therefore the old 23/23 Gate 0 must not be treated as sufficient
for the revised code.
