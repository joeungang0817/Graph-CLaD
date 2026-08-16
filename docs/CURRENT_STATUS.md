# Graph-CLaD 현재 상태와 실행 인계

기준 시각: 2026-08-16 (Asia/Seoul)  
공식 현재 단계: **Phase 3B — H3 action-alignment control**  
이 문서는 다음 세션에서 가장 먼저 읽는 단일 현재상태 문서다. 공식 연구 질문과 단계별
gate는 `01-plan/features/graph-clad-integrated-research-v4.plan.md`, 시간순 근거는
`research_log.md`를 따른다. `revised_research_roadmap_v3.md`는 v4 이전의 단계 개정
근거로 보존한다.

## 1. 한 줄 상태

Pair-local H0–H3 three-fold seed-0 screen은 12/12 runs가 끝났고 H3의 natural
PR-AUC가 가장 높았다. 그러나 action의 의미 정렬 효과를 분리하기 위한
episode-disjoint matched train-shuffled H3 control이 아직 완료되지 않았으므로 H3를
최종 representation으로 확정하지 않았다.

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

## 5. 지금 실행할 control

Config:

`configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json`

Output:

`/home/ubuntu/graphclad-artifacts/phase3_holder_action_v1/corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1`

예상 범위는 H3 train-shuffled 1개 모델 × 3 folds × seed 0으로 총 3 runs다.

```bash
tmux new -s graphclad-align
cd ~/Graph-CLaD
source .venv/bin/activate
export GRAPH_CLAD_PROJECT_ROOT="$HOME/Graph-CLaD"
export GRAPH_CLAD_ARTIFACT_ROOT="$HOME/graphclad-artifacts"
python -u -m scripts.phase3.run_corrected_architecture_gate \
  --config configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json
```

2026-08-16 사용자 확인으로 H3 action-alignment 실행은 시작했다. 다만 현재 process가
계속 실행 중인지 이미 완료됐는지와 artifact 무결성은 아직 확인하지 않았다. 다음
세션에서 `tmux ls`, output directory, result JSON으로 확인해야 한다. 완료 artifact가
있으면 재실행하지 않는다.

## 6. 완료 직후 검증

다음을 모두 만족해야 control이 완료된 것이다.

1. Result JSON의 `status`가 `completed`다.
2. Result의 `results` 길이가 3이다.
3. `runtime_manifest.json`이 있고 RTX 3090/CUDA/config SHA/manifest SHA가 기록됐다.
4. `checkpoints/`에 fold별 checkpoint 3개가 있다.
5. `predictions/`에 fold별 per-sample prediction 3개가 있다.
6. `code_snapshot/`이 있다.
7. stderr 또는 tmux 출력에 traceback이 없다.

Aligned H3의 기준 결과는 Colab Drive의 다음 root에 있다.

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2/pair_local_temporal_threefold_seed0_v1`

현재 alignment runner는 shuffled H3만 학습한다. Config의 `comparison_source` 문자열이
aligned result를 자동으로 불러와 비교해 주는 것은 아니다. 완료 후 same fold/seed 비교와
hierarchical bootstrap을 하려면 위 aligned H3 result와 H3 prediction artifact를 읽거나
서버로 별도 전송해야 한다. 기존 checkpoint 전체는 비교에 필요하지 않다.

## 7. Phase 3B gate

Aligned H3가 shuffled H3보다 natural PR-AUC에서 최소 2/3 tasks 우세한지 먼저 본다.
동시에 release F1과 hard-negative FPR이 심하게 악화되지 않는지 확인한다.

- 통과: H3, H1, H3-train-shuffled의 seeds 1/2 확대를 검토한다.
- 실패: 추가 seed 확대를 중단하고 pair-local 결과를 architecture finding으로 보고한다.
- 어느 경우든 90-item weak-label human review는 별도 완료해야 한다.
- Natural test가 primary이며 challenge는 독립 test가 아닌 stress analysis다.
- Conditional event metric은 oracle current holding을 사용하므로 end-to-end metric도 함께 보고한다.

## 8. 이후 전체 CLaD 계획

사용자가 선택한 제출 전 기본 방향은 다음과 같다.

1. Phase 3C에서 future action/graph 없이 semantic CLaD, pair-local temporal, 선택 graph
   representation의 foresight bridge를 같은 데이터와 same-capacity probe로 비교한다.
2. Phase 4에서 선택 representation을 CLaD Stage 1 latent foresight에 통합한다.
3. Stage 1을 freeze하고 canonical DDPM Diffusion Policy를 Stage 2로 연결한다.
4. Policy-only, semantic foresight, 선택 pair-local/graph foresight를 같은 policy 용량과
   학습 budget으로 비교한다.
5. Action chunk horizon은 원 논문 설정과 맞춰 `tau=6`을 우선 사용한다.
6. Current observation과 predicted foresight를 modality별 FiLM으로 결합하고 표준
   epsilon-prediction objective를 사용한다.

저장소에는 공식 Stage 2 코드와 rollout pipeline이 없다. 따라서 이 구현은 논문 설명을
따른 **CLaD-compatible controlled reimplementation**으로 표현하고 공식 재현이라고
주장하지 않는다. 현재 데이터도 LIBERO-LONG 10-task 공식 protocol과 다르므로 논문의
94.7%와 직접 비교하지 않는다. RTX 4090에서 Stage 2 200K steps가 약 20시간이라는 논문
기준을 고려하면 RTX 3090에서는 one-task smoke 후 reduced budget 비교를 먼저 확보한다.

## 9. 아직 해결되지 않은 항목

- Action-alignment control 완료 및 aligned-vs-shuffled paired 분석.
- 90-item weak-label human review.
- Manifest builder의 streaming/low-memory 구현. 현재 실행에는 기존 pass manifest를 사용한다.
- Phase 3C/4 구현과 notebook.
- Stage 2 Diffusion Policy, LIBERO rollout environment, success-rate evaluator.
- Semantic/VLM observation pipeline과 실제 CLaD Stage 1 trainer의 미공개 세부사항.
- `inside` label support.

## 10. 다음 세션에서 읽는 순서

1. 이 파일 `docs/CURRENT_STATUS.md`.
2. `docs/NEXT_SESSION_PROMPT.md`.
3. `docs/01-plan/features/graph-clad-integrated-research-v4.plan.md`의 Phase 3C 이후.
4. `docs/phase3_pair_local_temporal_threefold_seed0_result.md`.
5. 새 action-alignment result JSON과 runtime manifest.
