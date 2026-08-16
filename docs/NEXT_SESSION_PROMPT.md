# 다음 Codex 세션 재개 프롬프트

기준일: 2026-08-16  
연구계획서: Graph-CLaD 통합 연구계획서 v4.2  

아래 코드 블록 전체를 새 세션의 첫 메시지로 붙여 넣는다. 가능하면 마지막에 KCloud의
최신 `tmux` 출력, result JSON 요약 또는 오류도 함께 붙인다. 기록 시점 이후 실행 상태가
달라질 수 있으므로 새 세션은 문서의 “미완료” 표현을 그대로 믿지 말고 artifact를 먼저
확인해야 한다.

```text
현재 workspace는 C:\Users\User\Graph-CLaD이다. 이전 Graph-CLaD 연구를 이어서 진행해라.
답변, 계획서, 설명서, 연구기록은 한국어로 작성하되 code identifier, metric, config ID,
path, SHA256은 재현성을 위해 원문 표기를 유지해라.

## 작업 원칙

- 기존 사용자 파일, git diff, Colab/KCloud artifact를 삭제하거나 덮어쓰지 마라.
- 시작 즉시 `git status --short`와 관련 diff를 읽어라. 기록 시점 local branch는 `main`,
  HEAD는 `deecd6a`였고 문서 정리 변경이 modified/untracked 상태였다. 이 값은 다시
  확인하되, 현재 diff를 사용자 작업으로 간주하고 reset/checkout/clean하지 마라.
- 기존 legacy 결과와 corrected 결과를 같은 protocol 결과처럼 합치지 마라.
- 실행 중인 training process를 중단·재시작하지 마라. 완료된 run도 이유 없이 반복하지
  마라. 오래 걸리는 학습을 tmux에서 시작했다면 계속 polling하며 세션을 점유하지 말고
  output 경로와 확인 명령을 알려준 뒤 사용자에게 돌려줘라.
- 직접 SSH, Colab 또는 Drive에 접근할 수 없다면 실행하거나 확인한 척하지 마라. 필요한
  read-only 확인 명령을 사용자에게 정확히 제공하고 그 출력으로 판단해라.
- 새 실험은 항상 새 version/output directory를 사용하고 config, manifest, checkpoint,
  per-sample prediction, aggregate result, runtime manifest, code snapshot을 함께 보존해라.
- 결과가 가설과 반대여도 그대로 기록하고 gate 기준을 사후 변경하지 마라.

## 가장 먼저 읽을 파일

1. C:\Users\User\Graph-CLaD\docs\CURRENT_STATUS.md
2. C:\Users\User\Graph-CLaD\docs\01-plan\features\graph-clad-integrated-research-v4.plan.md
3. C:\Users\User\Graph-CLaD\docs\research_log.md
4. C:\Users\User\Graph-CLaD\docs\README.md
5. C:\Users\User\Graph-CLaD\docs\RESEARCH_WORKFLOW_FOR_BEGINNERS.md
6. C:\Users\User\Graph-CLaD\docs\CODEBASE_GUIDE_FOR_BEGINNERS.md
7. C:\Users\User\Graph-CLaD\docs\phase3_corrected_protocol_v2.md
8. C:\Users\User\Graph-CLaD\docs\phase3_corrected_threefold_seed0_result.md
9. C:\Users\User\Graph-CLaD\docs\phase3_pair_local_temporal_threefold_seed0_result.md
10. C:\Users\User\Graph-CLaD\docs\phase3_weak_label_audit_v2.md
11. C:\Users\User\Graph-CLaD\docs\unknowns.md
12. C:\Users\User\Graph-CLaD\docs\kcloudvpn_linux_ssh_runbook_ko.md
13. C:\Users\User\Graph-CLaD\configs\phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json
14. C:\Users\User\Graph-CLaD\scripts\phase3\run_corrected_architecture_gate.py
15. C:\Users\User\Graph-CLaD\scripts\phase3\analyze_corrected_predictions.py

문서 우선순위는 다음과 같다. 현재 실행 사실은 `CURRENT_STATUS.md`와 실제 artifact,
향후 연구 질문과 gate는 통합 계획서 v4.2, 날짜순 근거는 `research_log.md`를 따른다.
`revised_research_roadmap_v3.md`는 v4 이전 개정 근거이며 현재 canonical 계획서가 아니다.

## 연구 목표와 현재 주장 범위

최종 질문은 같은 data, split, action availability, capacity, probe/policy budget에서 다음
representation 중 무엇이 robot–object interaction과 spatial transition을 더 잘 보존하고,
그 이점이 동일 Stage 2 policy의 rollout으로 전달되는가이다.

1. 기존 CLaD semantic-transition representation.
2. Robot–object pair-local temporal representation.
3. Object–relation graph-transition representation.

Holding onset/release는 최종 목적이 아니라 architecture/representation probe다. Graph의
일반적 우월성, causal action 효과, 최종 Graph-CLaD 우월성은 아직 입증되지 않았다.
Current G1은 sparse GNN + late/global action이며 action-conditioned temporal edge model이
아니다.

## 완료된 핵심 근거

- Phase 0–2D 완료. LIBERO spatial task 0/1/2의 official demonstration 150개를 exact
  replay해 natural graph dataset, holding target view, fixed episode split을 만들었다.
- Legacy Phase 3 controlled experiment 45 runs 완료. Protocol confound가 있어 최종
  architecture 주장에는 사용하지 않는다.
- Near-parameter-matched reduced cross-fold는 3 folds × 3 seeds × 4 models = 36/36 runs다.
  Natural PR-AUC는 B1 0.4047, G1 0.3872, S-0 0.3691, G1-shuffled 0.2793이었다. B1이 가장
  강하고 방어 가능한 baseline이며 G1의 일관된 추가 이점은 입증되지 않았다.
- Corrected seed-0 gate에서 G1−B1 conditional PR-AUC는 +0.0626,
  95% CI [−0.0892,+0.1905]였고 G1−shuffled는 −0.0567
  [−0.1943,+0.0572]였다. G1은 release와 hard-negative를 개선했지만 action-alignment
  gate를 통과하지 못해 deeper GNN 확대를 중단했다.
- Pair-local H0–H3 three-fold seed-0는 12/12 runs 완료했다. Task-macro natural 결과:
  H0 PR-AUC 0.3626 / F1 0.3134 / release 0.0931 / hard-neg 0.3168
  H1 PR-AUC 0.4348 / F1 0.4920 / release 0.0215 / hard-neg 0.3358
  H2 PR-AUC 0.3941 / F1 0.3186 / release 0.1729 / hard-neg 0.2241
  H3 PR-AUC 0.4824 / F1 0.3617 / release 0.1193 / hard-neg 0.2339
- H3−H1 PR-AUC는 +0.0476이고 3/3 tasks에서 양수이며 release +0.0978,
  hard-negative FPR −0.1019다. 그러나 task 0 H3 hard-negative FPR은 0.4762다.
- H0–H3 result SHA256은
  `492d45521e6ccecbc4f0d89923f50d49642962d60c1c53daf093b5aec9b4d188`다.
- Weak-label audit package는 task 0/1/2 × onset/release/hard-negative × 10 = 90 rows,
  unique trajectory graph 592/592까지 준비됐다. Human decision은 기록 시점 0/90이므로
  label accuracy가 확인됐다고 말하지 마라.

Natural held-out test가 primary다. Challenge는 natural held-out episode에서 future event로
선택한 stress view이지 독립 generalization test가 아니다. Conditional event metric은
oracle current holding을 사용하므로 end-to-end predicted-current/future metric을 별도로
보고해야 한다. 같은 task의 seeds는 test episode를 공유하므로 9개 독립 표본이 아니다.

## 현재 가장 먼저 확인할 실행

공식 현재 단계는 Phase 3B H3 action-alignment control이다. 사용자가 KCloud에서 H3
action-alignment를 실행했다고 확인했다. 다만 기록 시점에는 process가 계속 실행 중인지
이미 완료됐는지와 artifact 무결성은 확정되지 않았다. 시작 명령을 다시 실행하지 말고
반드시 서버 process와 artifact를 먼저 확인해라.

KCloud 환경:

- SSH: `ubuntu@172.10.5.118`
- Repository: `/home/ubuntu/Graph-CLaD`
- Venv: `/home/ubuntu/Graph-CLaD/.venv`
- Artifact root: `/home/ubuntu/graphclad-artifacts`
- 실제 GPU: NVIDIA GeForce RTX 3090 24 GB
- Python 3.10.12, PyTorch 2.13.0+cu130

Action-alignment output:

`/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1`

Expected result filename:

`phase3_pair_local_temporal_action_alignment_seed0_v1.json`

Expected scope는 `H3-train-shuffled` × 3 task folds × seed 0 = 3 runs다. Config는 common
action-free current head를 사용하고 `current_loss_weight=0.25`이며, training action donor는
task-local episode-disjoint matched 방식이다.

다음 순서로 판정해라.

1. `tmux ls`, `tmux capture-pane`, process 목록으로 실행 중인지 확인한다.
2. 실행 중이면 중단하지 말고 현재 log와 partial artifact만 읽는다.
3. 완료됐다면 result `status=completed`, `results` 길이 3, checkpoint 3개, prediction 3개,
   `runtime_manifest.json`, `code_snapshot/`, traceback 없음까지 확인한다.
4. 완료 artifact가 있으면 재실행하지 않는다.
5. 시작되지 않았을 때만 `CURRENT_STATUS.md`의 tmux 명령을 사용자에게 제공한다.

서버 입력은 다음 세 개가 존재함을 이미 확인했다.

- `phase2d/data/phase2d_full_demo_v2_inputclean_stream1` 약 836 MB
- `phase2d/data/phase2d_holding_target_v2_inputclean_stream1` 약 357 MB
- `phase2d/data/phase2d_demo_split_manifest.json` 약 76 KB

Corrected manifest는
`/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/phase3B_R1_eval_manifest_v2.json`이며
`status=pass`, `folds=3`이다. 서버에서 manifest builder를 다시 실행했을 때 eager
full-payload loading 때문에 OOM `Killed`가 발생했다. 같은 builder 명령을 반복하지 마라.
기존 Colab pass manifest에서 source root만 KCloud 경로로 바꾼 portable copy를 사용하며,
Colab 원본은 `phase3B_R1_eval_manifest_v2_colab_original.json`으로 보존돼 있다.

## Action-alignment 완료 후 분석

현재 runner는 shuffled H3만 학습한다. Config의 `comparison_source` 문자열이 aligned H3
result를 자동으로 읽거나 paired comparison을 생성하지 않는다.

Aligned H3 기준 artifact는 Colab Drive의 다음 root다.

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/pair_local_temporal_threefold_seed0_v1`

Same fold/seed comparison과 hierarchical bootstrap에는 이 root의 H3 result와 per-sample
prediction이 필요하다. 전체 checkpoint를 옮길 필요는 없다. 필요한 result JSON과 H3
prediction만 provenance/SHA를 기록해 분석 환경으로 복사한다.

Gate 기준:

- Aligned H3가 natural conditional PR-AUC에서 shuffled H3보다 최소 2/3 tasks 우세.
- Release F1이 일관되게 붕괴하지 않음.
- Hard-negative FPR이 심하게 악화되지 않음.
- Donor QA와 paired prediction artifact가 완전함.

통과할 때만 H3/H1/H3-shuffled seeds 1/2 확대를 검토한다. 실패하면 추가 seed 확대를
중단하고 H1 또는 action-free pair-local을 Phase 3C 후보로 검토한다. 어느 경우든 결과
문서, `CURRENT_STATUS.md`, `research_log.md`, 계획서 gate 상태를 갱신한다.

## 이후 제출용 연구 순서

1. 90-item human weak-label review를 완료하고 pass/error/ambiguous와 오류 유형을 기록한다.
2. Phase 3C에서 future action/graph 없이 semantic, 선택 pair-local, 필요 시 graph
   representation을 같은 data와 same-capacity frozen probe로 비교한다.
3. Holding 외 object displacement, source→destination, valid spatial relation transition과
   100/25/10% label fraction sample efficiency를 평가한다.
4. Gate를 통과한 representation만 Phase 4 CLaD Stage 1 residual adapter에 연결한다.
   `alpha=0` 또는 adapter-off가 semantic baseline과 수치적으로 같아야 한다.
5. Stage 1을 freeze하고 canonical DDPM Diffusion Policy Stage 2 one-task smoke를 수행한다.
   Current observation+predicted foresight를 modality별 FiLM으로 conditioning하고 우선
   action horizon `tau=6`, epsilon-prediction objective를 사용한다.
6. Policy-only, semantic foresight, 최종 structured foresight만 동일 policy
   capacity/data/training/rollout budget으로 비교한다.

Official Stage 2 source와 official rollout pipeline은 없다. Network width/depth, noise
schedule, inference step, rollout wrapper, checkpoint criterion은 아직 확인되지 않았으므로
baseline smoke 전에 versioned config로 고정하고 모든 variant에 동일하게 적용해라. 결과는
`CLaD-compatible controlled reimplementation`으로 부르고 official reproduction 또는
논문 LIBERO-LONG 94.7%와 직접 비교라고 주장하지 마라.

제출 일정이 촉박하므로 gate 전에 대규모 학습하지 마라. 우선순위는 action-alignment
판정 -> weak-label audit -> Phase 3C technical smoke -> Stage 1 adapter-off check -> Stage 2
one-task smoke -> 시간이 남을 때 reduced paired rollout이다.

## 새 세션 첫 응답에서 할 일

1. 위 파일과 현재 git diff를 read-only로 확인한다.
2. 문서 기록과 실제 KCloud/Colab artifact 상태가 일치하는지 짧게 보고한다.
3. 이미 완료된 것, 실행 중인 것, 미완료인 것을 명확히 구분한다.
4. 필요한 파일 변경과 실행 범위를 먼저 제시한다.
5. 안전한 범위에서는 확인 후 바로 진행하되, 장시간 training을 시작했다면 사용자에게
   process/output/monitor 방법을 전달하고 생각을 중단한다.
```
