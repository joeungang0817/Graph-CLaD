# Graph-CLaD 통합 연구계획서 v4

> Summary: CLaD의 semantic foresight를 pair-local 또는 object–relation graph transition으로 구조화했을 때 robot–object interaction과 spatial transition, 나아가 동일 조건의 diffusion policy 성능이 개선되는지 단계적으로 검증한다.
>
> Project: Graph-CLaD  
> Version: 4.2  
> Author: Graph-CLaD 연구팀  
> Date: 2026-08-16  
> Status: Review

---

## 1. 문서의 역할

이 문서는 Graph-CLaD 연구의 **새 canonical 연구계획서**다. 초기 계획서의 연구 동기와
최종 목표는 유지하되, Phase 0–3에서 실제로 얻은 근거와 실패, corrected evaluation
protocol, KCloudVPN 실행 환경, 제출 직전의 최소 실행 범위를 반영한다.

문서 간 역할은 다음처럼 구분한다.

- 이 계획서: 연구 질문, 비교 조건, 단계, gate, 성공·중단 기준을 결정한다.
- `docs/CURRENT_STATUS.md`: 지금 실행 중인 작업과 경로를 기록한다.
- `docs/research_log.md`: 날짜순 실행 사실과 판단을 기록한다.
- `docs/revised_research_roadmap_v3.md`: v4 이전의 단계 개정 근거를 보존한다.
- `RESEARCH_GUIDE.md`: 폴더와 실행 방법을 설명한다.

계획과 현재 상태가 충돌하면 이미 완료된 실험 사실은 `research_log.md`와 artifact를
우선하고, 향후 의사결정 기준은 이 계획서를 따른다.

---

## 2. 연구 배경과 문제 정의

기존 CLaD는 현재 관측과 미래 관측의 semantic feature 차이, 그리고 action history를
이용해 latent transition 또는 foresight representation을 학습한다. 그러나 이 표현에는
다음 구조가 명시되어 있지 않다.

- 어떤 물체가 이동하거나 잡혔는가.
- robot과 각 object 사이의 관계가 어떻게 변했는가.
- source와 destination, support, near 같은 공간 관계가 어떻게 바뀌었는가.
- 여러 물체 중 task-relevant object가 무엇이며 action이 어느 pair에 작용했는가.

Graph-CLaD의 출발 가설은 두 시점의 상태를 object–relation graph로 만들고,
action-conditioned transition encoder를 통해 이 구조를 보존하면 semantic-only
representation보다 robot–object interaction과 spatial transition을 명확하게 나타낼 수
있다는 것이다.

다만 현재까지의 결과는 **complete-scene message passing 자체가 항상 유리하지 않음**을
보였다. Near-parameter-matched corrected gate에서 sparse G1은 강한 B1 pair MLP를 task
전반에서 일관되게 이기지 못했다. 따라서 v4는 graph라는 형식에 결론을 맞추지 않고,
다음 세 representation을 같은 조건에서 비교하는 연구로 재정의한다.

1. 기존 CLaD semantic-transition representation.
2. Robot–object pair-local temporal representation.
3. Object–relation graph-transition representation.

Holding onset/release는 최종 목적이 아니라 representation이 상호작용 변화를 포함하는지
검사하는 architecture/probe다. 최종 연구 질문은 선택된 representation을 CLaD Stage 1과
동일한 Stage 2 policy에 연결했을 때 실제 control 성능까지 이어지는가이다.

---

## 3. 연구 질문

### RQ1. 구조화된 representation의 정보 보존

같은 data, split, action availability, parameter budget, 학습 예산에서 pair-local 또는
graph-transition representation이 semantic representation보다 holding onset/release,
object displacement, source→destination, valid spatial relation transition을 잘 보존하는가?

### RQ2. Temporal history와 action의 기여

현재 snapshot만으로 부족한 holding transition 정보가 `<= t` causal history로
보완되는가? Action은 history와 독립적으로 추가 정보를 제공하며, 의미가 어긋난 action을
넣으면 성능이 일관되게 하락하는가?

### RQ3. Pair-local과 scene-level graph의 차이

Holding과 같은 target-specific interaction에서는 complete scene message passing보다
robot–object pair별 temporal encoding과 제한된 set context가 더 적합한가? Spatial
transition에서는 graph context가 추가 이점을 제공하는가?

### RQ4. Foresight에서 policy로의 전달

Offline probe 또는 Stage 1 latent 수준의 개선이 동일 capacity와 동일 budget의 canonical
DDPM Diffusion Policy에서 task success와 sample efficiency 개선으로 이어지는가?

---

## 4. 검증 가설과 반증 기준

| ID | 가설 | 지지 근거 | 반증 또는 제한 해석 |
|---|---|---|---|
| H1 | 구조화된 representation은 semantic baseline보다 relation transition을 잘 보존한다. | 같은-capacity frozen probe의 natural held-out PR-AUC·spatial metric이 여러 task에서 paired 개선 | 개선이 한 task/seed에만 있거나 hard-negative가 크게 악화되면 일반 우월성 주장 금지 |
| H2 | Causal past history는 snapshot에 누락된 interaction state를 보완한다. | H1−H0, H3−H2가 task 전반에서 양수이고 release/robustness도 유지 | gain이 weak-label feature 복제나 특정 threshold에만 의존하면 제한적 결과로 해석 |
| H3 | 올바르게 정렬된 action은 미래 transition에 유효한 조건이다. | aligned가 episode-disjoint matched shuffled보다 same fold/seed에서 일관되게 우세 | action-free current head에서도 차이가 없거나 shuffle이 우세하면 causal action 주장 금지 |
| H4 | Graph context는 pair-local 정보 이상의 spatial 이점을 제공한다. | exact/near-capacity pair baseline 대비 valid spatial relation과 perturbation test에서 개선 | G1처럼 일관된 개선이 없으면 pair-local inductive bias가 더 적합하다는 결과로 보고 |
| H5 | 더 유용한 Stage 1 foresight는 Stage 2 policy 성능을 개선한다. | 같은 policy 조건에서 paired rollout success·efficiency 개선 | offline 개선이 rollout으로 전달되지 않으면 representation과 control utility를 분리 보고 |

가설은 loss 감소만으로 채택하지 않는다. 각 가설에는 held-out metric, matched baseline,
perturbation/control, task 단위 반복 근거가 필요하다.

---

## 5. 연구 범위

### 5.1 포함 범위

- LIBERO official demonstration의 exact state replay로 만든 task 0·1·2 dataset.
- Simulator state에서 얻는 canonical object identity와 oracle graph.
- Holding, contact 및 valid support가 확인된 spatial relation.
- Semantic, pair-local temporal, graph-transition representation 비교.
- CLaD-compatible Stage 1 foresight와 canonical DDPM Diffusion Policy 연결.
- Natural held-out test, stress analysis, perturbation, sample-efficiency 평가.
- Config, manifest, code snapshot, checkpoint, prediction, runtime metadata 보존.

### 5.2 제외 또는 보류 범위

- RGB detector부터 graph를 만드는 end-to-end perception 성능 주장.
- Valid label support가 없는 `inside` relation.
- Human review 전 weak holding label을 ground truth로 표현하는 것.
- Stress view를 독립 challenge/generalization dataset으로 표현하는 것.
- 세 task 결과를 전체 LIBERO task family의 일반화로 확대하는 것.
- 제공되지 않은 official Stage 2 code의 완전 재현 주장.
- 현재 protocol 결과를 원 논문의 LIBERO-LONG 94.7%와 직접 비교하는 것.

Oracle graph 결과는 structured representation의 feasibility와 upper-bound 성격을 갖는다.
이는 RGB 기반 배치 가능성을 자동으로 보장하지 않는다.

---

## 6. 현재까지 확보한 근거

### 6.1 완료 상태

| 단계 | 상태 | 핵심 산출물 또는 판단 |
|---|---|---|
| Phase 0 | 완료 | 제공 CLaD baseline의 실행 경계와 unknown 확인 |
| Phase 1A | 완료 | LIBERO state/API/runtime 계약 확인 |
| Phase 2A | 완료 | deterministic static GraphSpec/extractor |
| Phase 2R | diagnostic 완료 | scripted contact/handler/frame regression; main claim에서 제외 |
| Phase 2D | 완료 | official-demo replay, temporal graph, holding target dataset, episode split |
| Phase 3A | code/data QA 완료, human QA 대기 | quota/leakage/hash 검사와 90-item audit package |
| Phase 3B legacy/corrected GNN | 완료 | B1이 가장 강한 baseline; G1의 일반 우월성 미입증 |
| Phase 3B pair-local H0–H3 | three-fold seed-0 완료 | H3가 후보이나 action-alignment control 대기 |
| Phase 3C 이후 | 미시작 | 앞 gate 통과 전 대규모 실행 금지 |

### 6.2 Phase 0–2D 기반 결과

Phase 0에서는 제공된 `baseline_code/`를 수정하지 않고 training forward의 네 loss,
finite gradient, evaluation embedding `[B,H]`, EMA copy/update 계약을 검사하는 smoke
경로를 만들었다. 제공 자료에는 실제 VLM preprocessing, action packing, Stage 2 policy가
없다는 경계도 확인했다.

Phase 1A와 2A의 live/state 결과는 다음과 같다.

| 항목 | 결과 | 의미 |
|---|---:|---|
| LIBERO live capture | snapshot 2개 | simulator state와 observation 접근 확인 |
| Snapshot당 entity | 23개 | object/fixture/site identity 조사 |
| Flattened simulator state | 92차원 | privileged state 계약 확인 |
| Extracted graph | 2개 | deterministic extractor live 검증 |
| Graph당 node | 24개 | robot 1, object 5, fixture 2, site 16 |
| Graph당 directed edge | 552개 | self-loop 없는 complete spatial topology |
| Node numeric feature | 24차원 | pose/state와 validity mask 포함 |

Unsupported predicate는 false로 바꾸지 않고 `valid=0`으로 유지했다. 이 결과는 simulator
state를 사용하는 controlled oracle graph의 구현 가능성을 확인한 것이며 RGB perception
성능 결과가 아니다.

Phase 2R scripted probe에서는 robot-base coordinate, containment semantic,
contact/handler/frame 오류를 발견해 수정했다. 그러나 scripted distribution의 shortcut
가능성 때문에 main training source에서는 제외했다.

Phase 2D에서는 `libero_spatial` task 0·1·2의 official demonstration 150개를 exact
replay했다. Gripper contact geom/body mapping 오류를 고친 뒤 전체 replay와 QA를 다시
수행했고, sample/episode leakage가 없는 natural graph dataset과 fixed split을 완성했다.
Holding-positive, onset, release, hard-negative index를 만들었고 `inside`는 valid support가
없어 deferred했다. 현재 KCloud에 보존된 input-clean natural dataset은 약 836 MB,
holding target-aligned dataset은 약 357 MB다.

### 6.3 Legacy Phase 3와 초기 architecture 탐색

Corrected protocol 이전에 총 45 runs를 수행했다. 이 단계는 trainer, model, metric,
artifact 경로와 failure mode를 확립했지만 validation/threshold와 current auxiliary-head
confound가 남아 있어 최종 architecture 주장의 근거로 사용하지 않는다.

Task 0, seed 0의 width-unmatched holder–object smoke는 다음 신호를 보였다.

| 모델 | Parameters | Natural event F1 | Stress event F1 | Natural AP | Stress AP |
|---|---:|---:|---:|---:|---:|
| P0 flat MLP | 20,514 | 0.3662 | 0.7400 | 0.3783 | 0.4984 |
| B1-v2 pair MLP | 35,490 | 0.4022 | 0.7872 | 0.4327 | **0.7153** |
| G1 sparse GNN | 44,946 | **0.4257** | **0.8000** | **0.4426** | 0.5767 |

G1의 F1 우위는 B1 대비 natural +0.0236, stress +0.0128로 작았다. B1의
hard-negative FPR은 natural 0.0381, stress 0.0000으로 G1의 0.1048/0.0984보다 좋았다.
따라서 이 smoke는 graph 우월성보다 **pair baseline이 매우 강하다**는 신호도 함께 줬다.

같은 task/seed의 parameter-identical topology 비교에서는 sparse가 complete보다 높았다.

| 비교 | Natural F1 차이 | Stress F1 차이 |
|---|---:|---:|
| Sparse late-action − complete late-action | +0.0426 | +0.2127 |
| Sparse edge-FiLM − complete edge-FiLM | +0.0684 | +0.3014 |

Complete graph 안에서 FiLM을 추가한 이득은 natural +0.0047, stress +0.0162에 그쳤고,
edge shuffle 후 오히려 좋아지는 경우도 있었다. 이 결과로 complete topology와 추가 FiLM
complexity의 확대를 중단했다. 다만 one fold/seed 결과이므로 sparse topology의 일반적
우월성으로 해석하지 않는다.

Task 0의 3-seed action-semantics gate에서는 aligned G1의 natural event F1이 0.3965,
train-shuffled가 0.3207이었다. Episode-disjoint global shuffle 대비 natural PR-AUC는 평균
+0.0718로 3/3 seeds, stress PR-AUC는 +0.0831로 3/3 seeds에서 aligned가 높았다. 반면
no-action S-0의 natural PR-AUC 0.3761과 hard-negative FPR 0.1714는 G1의 0.3720과
0.2476보다 좋았다. 즉 task 0에서 action 정렬 신호는 있었지만 모든 metric과 task로
일반화되지 않았다.

### 6.4 Near-parameter-matched reduced cross-fold 결과

Phase 3B-R1에서 3 folds × 3 seeds × 4 models, 총 36/36 runs와 checkpoint 36개를
완료했다. Parameter count는 B1 45,668, G1 44,946, S-0 44,784로 near-matched다.

| 모델 | Natural F1 | Stress F1 | Natural PR-AUC | Stress PR-AUC | Natural hard-neg FPR | Stress hard-neg FPR |
|---|---:|---:|---:|---:|---:|---:|
| B1-v2 | 0.3285 | **0.7670** | **0.4047** | 0.4898 | **0.1075** | **0.1017** |
| G1-correct | 0.3400 | 0.7465 | 0.3872 | 0.4972 | 0.2146 | 0.1884 |
| S-0 | **0.3604** | 0.5927 | 0.3691 | **0.5159** | 0.2836 | 0.2613 |
| G1 train-shuffled | 0.3470 | 0.4746 | 0.2793 | 0.4093 | 0.3277 | 0.2931 |

G1−B1 natural event F1은 +0.0115였지만 G1이 높은 run은 2/9뿐이었다. Stress 차이는
−0.0206이며 역시 2/9만 G1이 높았다. Task별 3-seed 평균에서도 G1은 stress 기준 task
2에서만 명확히 우세했다. 동일 task의 세 seed가 test episode를 공유하므로 위 9-run
평균은 9개 독립 표본이 아니다.

이 결과로 B1을 가장 강하고 방어 가능한 non-graph baseline으로 확정했고, graph의
일관된 추가 이점은 입증되지 않았다고 판단했다. Action shuffle 하락은 action 정보가
사용될 가능성을 보였지만 당시 current auxiliary head에도 action이 들어갔으므로 인과적
action semantics 증거로 사용할 수 없었다.

### 6.5 Corrected protocol three-fold seed-0 결과

Auxiliary-head confound, checkpoint, threshold, calibration, oracle-current metric 문제를
교정한 뒤 task 0·1·2, seed 0의 12 runs를 비교했다.

| 모델 | Conditional PR-AUC | Event F1 | End-to-end PR-AUC | Release F1 | Hard-negative FPR |
|---|---:|---:|---:|---:|---:|
| B1-v2 | 0.3792 | 0.4083 | 0.2456 | 0.0000 | 0.4980 |
| G1 | 0.4418 | 0.3272 | **0.2557** | 0.1626 | **0.1653** |
| S-0 | 0.4704 | 0.3302 | 0.2377 | 0.0570 | 0.3923 |
| G1 train-shuffled | **0.4985** | **0.4333** | 0.1671 | **0.1769** | 0.2387 |

Paired hierarchical bootstrap의 주요 결과는 다음과 같다.

| 비교 | 추정값 | 95% CI |
|---|---:|---:|
| G1−B1 conditional PR-AUC | +0.0626 | [−0.0892, +0.1905] |
| G1−B1 event F1 | −0.0811 | [−0.2017, +0.0294] |
| G1−B1 release F1 | +0.1626 | [+0.0701, +0.2967] |
| G1−B1 hard-negative FPR | −0.3326 | [−0.4201, −0.2429] |
| G1−shuffled conditional PR-AUC | −0.0567 | [−0.1943, +0.0572] |
| G1−shuffled event F1 | −0.1062 | [−0.2001, −0.0379] |

G1은 B1보다 release와 hard-negative rejection은 개선했지만 primary PR-AUC의 CI가 0을
포함했고 event F1은 낮았다. 더 중요하게 train-shuffled G1이 task-macro PR-AUC와 F1에서
G1보다 높았다. 따라서 `stop_gnn_three_seed_expansion_and_pivot_pair_local_temporal_encoder`
결정을 내렸다. 이는 graph가 쓸모없다는 결론이 아니라 현재 late-action G1이
architecture/action gate를 통과하지 못했다는 뜻이다.

### 6.6 Pair-local H0–H3 task-1 smoke

Task 1, seed 0에서 causal history와 action의 2×2 factorial을 먼저 검사했다.

| 모델 | History | Action | PR-AUC | Event F1 | End-to-end PR-AUC | Release F1 | Hard-neg FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 없음 | 없음 | 0.3810 | 0.3671 | 0.1939 | 0.0000 | 0.5000 |
| H1 | 있음 | 없음 | 0.4960 | **0.5352** | 0.2207 | 0.0000 | 0.3000 |
| H2 | 없음 | 있음 | **0.5869** | 0.4184 | 0.2439 | **0.2143** | **0.0900** |
| H3 | 있음 | 있음 | 0.5731 | 0.3357 | **0.4227** | 0.1075 | 0.1100 |

이 smoke에서는 action-only H2가 conditional PR-AUC, release, hard-negative에서 가장
좋았고, H3에 history를 추가하면 conditional PR-AUC가 0.0138 낮아졌다. 반면 H3의
end-to-end PR-AUC는 가장 높았다. 한 task/seed이므로 H2 또는 H3를 확정하지 않고 세
fold로 확대했다.

### 6.7 Pair-local H0–H3 three-fold seed-0 결과

총 12/12 runs의 task-macro natural 결과는 다음과 같다.

| 모델 | History | Action | Natural PR-AUC | Conditional event F1 | Release F1 | Hard-negative FPR |
|---|---:|---:|---:|---:|---:|---:|
| H0-state | 없음 | 없음 | 0.3626 | 0.3134 | 0.0931 | 0.3168 |
| H1-history | 있음 | 없음 | 0.4348 | **0.4920** | 0.0215 | 0.3358 |
| H2-action | 없음 | 있음 | 0.3941 | 0.3186 | **0.1729** | **0.2241** |
| H3-history-action | 있음 | 있음 | **0.4824** | 0.3617 | 0.1193 | 0.2339 |

Factorial contrast는 다음처럼 해석한다.

| Contrast | PR-AUC 차이 | 양수 task | Release F1 차이 | Hard-negative FPR 차이 |
|---|---:|---:|---:|---:|
| H1−H0 | +0.0722 | 2/3 | −0.0716 | +0.0190 |
| H2−H0 | +0.0315 | 1/3 | +0.0798 | −0.0927 |
| H3−H1 | +0.0476 | **3/3** | +0.0978 | −0.1019 |
| H3−H2 | +0.0883 | 2/3 | −0.0536 | +0.0097 |
| H3−H0 | +0.1198 | **3/3** | +0.0262 | −0.0830 |

H3를 현재 후보로 유지하지만 최종 모델로 확정하지 않는다. Task 0에서 H3
hard-negative FPR이 0.4762로 높고, 아직 episode-disjoint matched action-alignment
control이 끝나지 않았기 때문이다. KCloudVPN runner는 shuffled H3 세 run만 생성하므로,
완료 후 기존 aligned H3 result와 per-sample prediction을 결합해 same-fold/seed로
분석해야 한다.

### 6.8 Weak-label audit 준비 결과

Task 0·1·2 × onset/release/hard-negative × 10개로 총 90개 audit row를 만들었다. 각
item의 `t`부터 `t+6`까지 trajectory evidence를 구성했고, overlap을 제거한 unique graph
frame 592/592를 회수했으며 missing/conflict는 없었다. Interactive viewer와 atomic review
CSV도 준비됐다.

그러나 human decision은 현재 0/90이다. 따라서 이 결과는 audit **도구와 표본 준비가
완료된 것**이지 label accuracy가 확인됐다는 뜻이 아니다.

### 6.9 재현성 및 해석 교정 결과

연구 과정에서 성능값 자체뿐 아니라 결과 해석을 바꾼 교정도 있었다.

1. **Metric display path 교정:** 초기 출력 일부가 nested holding metric의 잘못된 경로를
   읽었다. 이후 authoritative result JSON의 holding-specific nested field를 사용하도록
   고쳤고 문서의 수치도 교정된 field를 따른다.
2. **Legacy shuffle donor 교정:** Evaluation batch를 한 row roll하던 초기 shuffle은
   natural donor의 98.35%, stress donor의 95.81%가 같은 episode였다. Action L2 distance도
   각각 0.6835와 0.9284로 작아 강한 semantic perturbation이 아니었다. 이후 global
   episode-disjoint shuffle에서는 same-episode 비율이 0%가 되었고 action L2 distance는
   natural 4.0506, stress 2.6823으로 증가했다.
3. **Threshold 의존성 확인:** Corrected three-fold에서 natural validation이 선택한
   threshold는 model/task별로 크게 달랐고 G1은 흔히 0.91–0.95를 선택했다. 따라서
   thresholded F1만 비교하지 않고 natural PR-AUC를 primary, F1을 secondary로 고정했다.
4. **통계 단위 교정:** 3 tasks × 3 seeds를 9 independent samples로 세지 않고 task를
   outer unit으로 보고한다. Corrected seed-0 bootstrap은 seed uncertainty가 아니라
   held-out episode/event uncertainty만 나타낸다.
5. **Action 경로 명명 교정:** G1 action은 message update가 아닌 late/global prediction
   head 입력이다. 기존 결과를 action-conditioned temporal edge evidence라고 부르지 않는다.

Authoritative Phase 3 artifact는 다음 위치에서 찾는다.

| 결과 | Artifact 또는 persistent root | 상태 |
|---|---|---|
| Legacy reduced cross-fold | `phase3_holder_action_v1/phase3_reduced_crossfold_gate_v1.json` | 36/36 완료, legacy protocol |
| Corrected GNN seed-0 | `corrected_protocol_v2/threefold_seed0_combined/phase3_corrected_threefold_seed0_combined_v2.json` | 12 runs 결합 완료 |
| Pair-local task-1 smoke | `corrected_protocol_v2/pair_local_temporal_smoke_task1_seed0_v1` | SHA256 `ae986247...c502` |
| Pair-local three-fold | `corrected_protocol_v2/pair_local_temporal_threefold_seed0_v1` | SHA256 `492d4552...d188`, 12/12 완료 |
| H3 action-alignment | KCloud `corrected_protocol_v2/kcloudvpn_pair_local_temporal_action_alignment_seed0_v1` | 실행 시작 확인; 완료·무결성 재확인 필요 |

SHA 축약값은 탐색용 표기다. 무결성 검증에는 각 result 문서와 artifact에 저장된 전체
SHA256을 사용한다. 현재 action-alignment 상태는 연구계획서에 고정하지 않고 매 세션
`CURRENT_STATUS.md`, server process, result JSON으로 다시 확인한다.

### 6.10 누적 의사결정

지금까지의 결과에서 확정할 수 있는 것은 다음과 같다.

- B1 pair MLP는 반드시 유지해야 하는 강한 baseline이다.
- Complete scene message passing과 추가 edge FiLM complexity는 현재 holding probe에서
  확대 근거가 없다.
- Sparse G1은 release와 hard-negative에서 유용한 신호를 보였지만 primary metric과
  action control을 일관되게 통과하지 못했다.
- Pair-local temporal H3는 가장 유망한 현재 후보지만 action alignment와 weak-label
  validity가 확인되기 전에는 우월성 또는 causal action 효과를 주장할 수 없다.
- 다음 architecture 선택은 action-alignment 결과에 따라 H3, H1 또는 다른 action-free
  pair-local 후보 중에서 이루어진다.
- 최종 Graph-CLaD 주장은 Phase 3C frozen representation 비교와 Stage 2 paired rollout이
  끝난 뒤에만 가능하다.

---

## 7. 데이터와 split 계약

### 7.1 데이터 흐름

```text
Official HDF5 demonstration + BDDL
  -> episode split 고정
  -> exact simulator state replay
  -> canonical graph timeline
  -> causal temporal relation/holding weak label
  -> natural multi-horizon samples
  -> event-enriched stress view
  -> corrected evaluation manifest
  -> model training / frozen probe / policy conditioning
```

### 7.2 입력 경계

- Primary encoder input은 현재 시점 `t` 이하의 state/history만 사용한다.
- Future graph, future action, future event identity, future success/reward는 입력에서
  제외한다.
- `is_object_of_interest`, BDDL-derived target relevance처럼 target identity 또는 future
  selection shortcut이 될 수 있는 field도 primary model input에서 제외한다.
- Label 또는 target 생성에 future frame을 사용할 수 있으나 input payload와 엄격히
  분리한다.
- Normalization 통계는 train split만 사용해 적합한다.
- Episode split은 window/sample 생성 전에 고정한다.

### 7.3 Natural과 stress view

- Natural held-out test를 primary evaluation으로 사용한다.
- Stress view는 natural held-out episode에서 future event 정보를 이용해 선택한
  event-enriched subset이다.
- 두 view는 독립 test 두 개로 세지 않는다.
- 겹치는 sample ID는 graph, action, label payload SHA256이 동일해야 한다.

### 7.4 Weak-label 계약

Holding label은 contact, gripper closure, 3-frame stability, object-following을 결합한
heuristic weak label이다. Task 0·1·2에서 onset 10, release 10, hard negative 10씩 총
90개를 trajectory evidence로 사람이 검토한다.

Human review 전에는 다음 표현만 허용한다.

- “heuristic holding event에 대한 성능”
- “internal-consistency sensitivity group”
- “conditional/oracle-current event metric”

Label 통과율과 오류 유형이 문서화된 뒤에만 작은 model gain을 실제 interaction 보존
근거로 해석한다.

---

## 8. 공통 평가 protocol

### 8.1 학습과 checkpoint

1. 비교 모델은 같은 sample ID, split, loss, epoch, patience, batch 조건을 사용한다.
2. Current auxiliary head는 action-free로 통일하거나 `current_loss_weight=0`으로 둔다.
3. Checkpoint는 **natural validation holding event PR-AUC**로 선택한다.
4. Threshold는 natural validation에서 한 번 선택한다.
5. 선택 threshold를 natural test와 stress view에 그대로 적용한다.

### 8.2 Metric 우선순위

Primary:

- Natural held-out conditional/oracle-current event PR-AUC.
- Phase 3C 이후에는 valid spatial relation별 PR-AUC와 displacement/source→destination
  metric을 함께 primary outcome으로 둔다.

Secondary:

- Event F1, onset F1, release F1.
- Hard-negative false-positive rate.
- End-to-end predicted-current/predicted-future event PR-AUC·F1.
- Brier score, ECE, 선택 threshold.
- Stress-view sensitivity.

Conditional event metric은 ground-truth current holding과 predicted future holding의
XOR이므로 “현재 symbolic holding 상태를 알고 있다는 조건”을 항상 명시한다.

### 8.3 저장 단위

각 sample prediction에는 최소한 다음을 저장한다.

- Probability, binary prediction, target.
- Current and future relation target/prediction.
- Sample ID, task ID, episode ID, timestep, event type.
- Fold, seed, model/config ID, selected threshold.
- Hard-negative와 stress-view membership.

이 artifact는 paired comparison과 episode/event hierarchical bootstrap의 입력이다.

### 8.4 통계

- 같은 task의 seeds는 test episode를 공유하므로 독립 표본으로 세지 않는다.
- 먼저 task별 seed 평균과 같은 fold/seed의 paired difference를 보고한다.
- Task를 outer unit, episode/event를 inner resampling unit으로 한 hierarchical bootstrap
  confidence interval을 저장한다.
- 9-run 단순 평균만으로 task generalization 결론을 내리지 않는다.

---

## 9. 모델 비교 설계

### 9.1 Phase 3B architecture screen

| 계열 | 모델 | 역할 |
|---|---|---|
| Pair baseline | B1-v2 | object/robot pair MLP + late action; 강한 non-graph 기준 |
| Graph | G1 late-action | sparse robot–object GNN + late/global action |
| No-action graph | S-0 | G1 topology에서 action 제거 |
| Action control | G1 train-shuffled | 의미가 어긋난 action에 대한 민감도 |
| Pair-local factorial | H0–H3 | history/action의 독립·결합 효과 |

Parameter count는 exact 또는 near-matched로 만들고 차이를 표에 공개한다. G1의 후속 개발은
pair-local/semantic baseline보다 valid spatial relation에서 명확한 추가 이점이 있을 때만
재개한다.

### 9.2 Phase 3C representation bridge

같은 `<= t` data로 다음 encoder를 학습하고 freeze한다.

1. Semantic transition encoder.
2. Gate를 통과한 pair-local temporal encoder.
3. 선택된 graph-transition encoder 또는 graph context variant.

모든 encoder에 동일 capacity의 linear 또는 small probe를 붙여 다음을 예측한다.

- Holding onset/release.
- Object displacement.
- Source→destination 변화.
- Valid spatial relation transition: `on`, `support`, `near`, `left/right` 등.

Label fraction을 줄인 100%, 25%, 10% probe로 sample efficiency를 검사한다. Edge/action
permutation은 inference-time sensitivity와 donor provenance를 함께 기록한다.

### 9.3 Phase 4 Stage 1 통합

Baseline CLaD의 semantic latent interface를 유지하면서 구조화된 encoder output을
foresight adapter로 연결한다. 첫 구현은 residual adapter를 권장한다.

```text
z_semantic + alpha * adapter(z_pair_or_graph) -> z_foresight
```

필수 조건:

- `alpha=0` 또는 adapter-off가 semantic baseline과 수치적으로 일치한다.
- Stage 2가 받는 tensor shape와 normalization을 variant 간 동일하게 유지한다.
- Future action 또는 future graph가 Stage 1 input에 들어가지 않는다.
- Semantic, pair-local, graph variant는 같은 data와 update budget을 사용한다.
- Encoder freeze/unfreeze schedule을 config에 명시한다.

### 9.4 Phase 5–7 Stage 2

제공된 official Stage 2 source가 없으므로 논문 설명에 가까운 controlled
reimplementation을 사용한다.

- Stage 1은 우선 freeze한다.
- Current observation과 predicted foresight를 modality-specific FiLM으로 결합한다.
- Canonical DDPM epsilon-prediction objective를 사용한다.
- 기본 action chunk horizon은 `tau=6`이다.
- Policy capacity, optimizer, action horizon, training steps, demonstrations, rollout seed와
  budget을 variant 간 동일하게 유지한다.

Official Stage 2 source에서 확인되지 않은 network width/depth, noise schedule, inference
step, rollout wrapper, checkpoint criterion은 baseline one-task smoke 전에 versioned config로
명시한다. 한 번 고정한 뒤 semantic/structured variant에 동일하게 적용하며, 공식 설정으로
확인된 값과 연구팀의 controlled reimplementation 가정을 구분해 기록한다.

최소 비교:

1. Policy-only.
2. Semantic CLaD foresight.
3. Phase 3C/4에서 선택된 pair-local 또는 graph foresight.

가능하면 shuffled/zeroed foresight control을 추가해 policy가 conditioning을 실제로
사용하는지 확인한다.

---

## 10. 단계별 실행 계획과 gate

### Phase 3B-R2 — H3 action-alignment control

목적은 H3의 action gain이 단순 action magnitude/state shortcut이 아니라 task 내 의미 정렬과
관련되는지 검사하는 것이다.

실행:

- Held-out task 0·1·2, seed 0.
- Episode-disjoint donor.
- Action norm 또는 state distance가 과도하게 다르지 않도록 matching.
- Aligned와 shuffled는 같은 fold, sample, split, budget, threshold protocol 사용.

통과 기준:

- Aligned H3 natural PR-AUC가 shuffled보다 최소 2/3 tasks에서 높다.
- Release F1이 일관되게 붕괴하지 않는다.
- Hard-negative FPR이 실질적으로 악화되지 않는다.
- Prediction artifact와 donor QA가 모두 존재한다.

실패 시 H3의 causal action 주장을 중단하고 H1 또는 action-free pair-local encoder를
Phase 3C 후보로 검토한다. 이는 연구 실패가 아니라 action conditioning의 한계를 밝힌
결과다.

### Phase 3A-H — Human weak-label audit

90-item audit를 완료하고 task/event별 통과율, false onset, delayed release, contact-only,
occlusion/trajectory ambiguity 같은 오류 유형을 기록한다.

- 통과율이 낮거나 특정 task에 오류가 집중되면 해당 label을 수정하고 새 dataset/config
  version으로 관련 실험만 재평가한다.
- 기존 result는 legacy로 보존하고 덮어쓰지 않는다.

### Phase 3B-R3 — 제한적 seed 확대

Action-alignment gate를 통과한 후보만 seeds 1/2로 확대한다. H0–H3 전체를 이유 없이 다시
실행하지 않는다. 우선 비교 후보는 H3, H1, H3-shuffled이며 gate 결과에 따라 축소한다.

### Phase 3C — No-future-action foresight bridge

목표는 offline future-action predictor가 아니라 실제 CLaD 입력 경계에서 유효한
representation을 선택하는 것이다.

통과 기준:

- Natural held-out primary metric이 matched semantic baseline보다 최소 2개 task에서
  개선된다.
- Release 또는 valid spatial transition 중 적어도 하나에서 일관된 개선이 있다.
- Hard-negative와 calibration이 심하게 악화되지 않는다.
- 낮은 label fraction에서 sample-efficiency 이점이 관찰된다.
- Action/edge/history perturbation 결과가 모델 구조와 일치한다.

통과하지 못하면 “offline relation prediction에는 유효하나 CLaD foresight 근거는 부족”으로
결론 내리고 구조화된 Stage 1 확대를 중단한다.

### Phase 4 — CLaD Stage 1 통합

통과 기준:

- Adapter-off baseline equivalence test 통과.
- 모든 variant의 latent interface와 update budget 일치.
- Training/validation이 finite하고 reproducible하다.
- Frozen-probe 결과가 Phase 3C 방향과 모순되지 않는다.
- Foresight perturbation에 representation-sensitive response가 있다.

### Phase 5 — Stage 2 policy 구현과 one-task smoke

부담이 작은 한 task에서 다음만 확인한다.

- Batch와 action chunk shape가 기대값과 일치한다.
- DDPM training loss가 finite하다.
- Checkpoint resume과 deterministic seed가 동작한다.
- Zeroed/shuffled foresight에 policy output이 무반응하지 않는다.
- 짧은 rollout에서 환경 reset, action scaling, termination이 정상이다.

### Phase 6 — 동일 조건의 reduced policy 학습

Policy-only, semantic foresight, 선택 structured foresight만 같은 budget으로 학습한다.
Stage 1 후보를 여러 개 무분별하게 연결하지 않는다.

### Phase 7 — Rollout 평가

- Task success rate와 confidence interval.
- Episode return, completion stage, failure category.
- Demonstration 또는 training-step fraction별 sample efficiency.
- Same episode/task seed의 paired comparison.
- Action/foresight perturbation robustness.

### Phase 8 — 종합 분석과 보고

Offline representation 결과와 policy 결과를 분리한 뒤 연결 관계를 분석한다. 유리한
결과만 선택하지 않고 task별 차이, failure case, unsupported claim을 함께 보고한다.

---

## 11. 제출 직전 최소 실행 범위

시간이 매우 제한된 경우 전체 3-seed 또는 전체 rollout을 강행하지 않는다. 최소 산출물은
다음 순서를 따른다.

1. H3 action-alignment 3-run 완료와 paired table.
2. 90-item human audit 중 가능한 범위의 명시적 완료율과 미완료 항목 공개.
3. Phase 3C의 한 task/seed technical smoke.
4. Semantic baseline과 최종 structured 후보의 matched frozen probe.
5. Stage 1 residual adapter와 adapter-off equivalence.
6. Stage 2 one-task DDPM training/rollout smoke.
7. 시간이 남을 때만 동일 budget의 reduced policy comparison.

Stage 2 smoke까지만 끝난 경우 “전체 CLaD pipeline을 연결했다”고 말할 수는 있지만,
“Graph-CLaD가 policy 성능을 개선했다”고 말하려면 paired rollout 결과가 반드시 필요하다.

---

## 12. 재현성 산출물 계약

새 실험은 기존 output을 덮어쓰지 않고 새 protocol/version directory에 다음을 저장한다.

| 산출물 | 필수 내용 |
|---|---|
| Config | 모든 input/output 경로, model, seed, split, budget, metric protocol |
| Manifest | sample/episode/fold, provenance, QA, payload hash |
| Code snapshot | 실행 당시 source와 config |
| Runtime manifest | host, OS, Python, PyTorch, GPU, driver/CUDA, command, Git commit/diff 상태 |
| Checkpoint | fold/seed/model ID와 selection metric |
| Prediction | per-sample probability, target, metadata, frozen threshold |
| Aggregate result | task별 값, seed 평균, paired difference, claim limit |
| Analysis | hierarchical bootstrap와 perturbation/control 결과 |
| Logs | stdout/stderr, 시작·종료 상태, traceback 여부 |

대용량 dataset, checkpoint, prediction은 Git에 넣지 않는다. KCloudVPN의 authoritative
artifact root는 `/home/ubuntu/graphclad-artifacts`이며, 기존 Colab artifact는
`/content/drive/MyDrive/Graph-CLaD/artifacts` 아래에서 legacy 근거로 보존한다.

---

## 13. 위험과 완화책

| 위험 | 영향 | 완화 |
|---|---|---|
| Weak label 오류 | 작은 model gain의 의미 왜곡 | 90-item human audit, 오류 유형별 재평가, label version 분리 |
| Future leakage | foresight 성능 과대평가 | `<= t` feature 계약, payload audit, future field blacklist |
| Auxiliary-head action confound | shuffled control 해석 불가 | action-free current head 또는 current loss 0 |
| Validation/threshold contamination | test F1 과대평가 | natural validation checkpoint와 frozen threshold |
| Capacity mismatch | graph 우위/열위 왜곡 | exact/near parameter matching과 parameter 공개 |
| Task dependence | 9 runs를 독립 표본으로 오해 | task-first report와 hierarchical bootstrap |
| Stress-view 오해 | 두 번의 독립 검증처럼 보임 | natural primary, stress analysis only |
| Manifest builder CPU OOM | 서버 실행 중단 | 검증된 portable manifest 사용; builder streaming 전환 전 반복 금지 |
| Colab/KCloud path 차이 | 재현 실패 | environment-variable root와 runtime manifest |
| Stage 2 source 부재 | official reproduction 주장 불가 | controlled reimplementation으로 한정하고 protocol 차이 공개 |
| 제출 일정 압박 | 불완전 대규모 실험 | gate-first, one-task smoke, 최종 후보만 확대 |

---

## 14. 성공 기준과 Definition of Done

### 14.1 Architecture 수준 완료

- Corrected evaluation protocol이 모든 후보에 동일하게 적용됨.
- Human weak-label audit 결과가 문서화됨.
- Pair-local/graph/action/history control이 task별 paired 결과로 보고됨.
- 실패 후보를 확대하지 않은 이유가 gate와 함께 기록됨.

### 14.2 Representation 수준 완료

- Semantic, pair-local, graph representation이 같은 data와 probe로 비교됨.
- Holding 외 displacement와 최소 한 종류의 valid spatial transition이 포함됨.
- Sample efficiency, hard-negative, perturbation, task별 paired result가 저장됨.
- 가장 좋은 representation이 metric과 한계에 근거해 선택됨.

### 14.3 전체 CLaD 수준 완료

- Stage 1 foresight variant가 동일 latent interface로 연결됨.
- Stage 2 DDPM policy가 동일 capacity/budget으로 학습됨.
- Policy-only, semantic, structured foresight의 paired rollout이 완료됨.
- 성공률뿐 아니라 failure case와 uncertainty가 보고됨.
- Config, snapshot, checkpoint, prediction/rollout, analysis를 다른 연구자가 따라갈 수 있음.

### 14.4 최종 주장 기준

가능한 최종 결론은 결과에 따라 세 갈래다.

1. **Graph 우세:** graph가 pair/semantic보다 spatial과 policy에서 반복적으로 개선.
2. **Pair-local 우세:** holding과 policy에는 complete graph보다 target-pair temporal
   representation이 더 적합.
3. **구조화 이점 미확인:** offline gain이 불안정하거나 policy로 전달되지 않음.

세 결론 모두 유효한 연구 결과다. 사전에 정한 gate와 claim limit을 지키는 것이 모델의
승패보다 우선한다.

---

## 15. 즉시 다음 작업

1. KCloudVPN `graphclad-align` session과 output에서 H3 shuffled 3-run 완료 여부를 확인한다.
2. Aligned H3 result/prediction을 서버 또는 분석 환경에 준비한다.
3. Same fold/seed paired comparison과 hierarchical bootstrap artifact를 만든다.
4. Phase 3B gate 판정을 `CURRENT_STATUS.md`, result 문서, `research_log.md`에 기록한다.
5. 90-item human weak-label audit를 완료한다.
6. 통과한 representation만 Phase 3C technical smoke로 이동한다.
7. Stage 1 adapter-off baseline부터 구현한 뒤 Stage 2 one-task smoke를 수행한다.

---

## 16. 관련 문서

- 현재 상태: `docs/CURRENT_STATUS.md`
- 연구 전체 입문: `docs/RESEARCH_WORKFLOW_FOR_BEGINNERS.md`
- 코드 입문: `docs/CODEBASE_GUIDE_FOR_BEGINNERS.md`
- 수정 로드맵 v3: `docs/revised_research_roadmap_v3.md`
- Pair-local 설계: `docs/01-plan/features/phase3_pair_local_temporal_encoder.plan.md`
- Corrected protocol: `docs/phase3_corrected_protocol_v2.md`
- 초기 holder–object smoke: `docs/phase3_holder_action_v2_smoke_result.md`
- Topology/action 후속: `docs/phase3_topology_action_followup_result.md`
- Task-0 action semantics: `docs/phase3_action_semantics_gate_result.md`
- 36-run reduced cross-fold: `docs/phase3_reduced_crossfold_gate_result.md`
- Corrected three-fold seed-0: `docs/phase3_corrected_threefold_seed0_result.md`
- Pair-local smoke: `docs/phase3_pair_local_temporal_smoke_result.md`
- H0–H3 three-fold seed-0: `docs/phase3_pair_local_temporal_threefold_seed0_result.md`
- Weak-label audit: `docs/phase3_weak_label_audit_v2.md`
- KCloudVPN 실행: `docs/kcloudvpn_linux_ssh_runbook_ko.md`
- 시간순 기록: `docs/research_log.md`
- 초기 계획 원본: `Graph_CLaD_Stage2_최종목표_연구실행계획서.pdf`

---

## 17. 버전 기록

| Version | Date | 변경 내용 | Author |
|---|---|---|---|
| 4.2 | 2026-08-16 | 연구기록 대조 후 metric path, weak legacy shuffle, threshold·통계 단위, G1 action 명명 교정과 authoritative artifact 위치, Stage 2 미확정 설정 계약을 추가 | Graph-CLaD 연구팀 |
| 4.1 | 2026-08-16 | Phase 0–2D 기반 결과, legacy 45-run 탐색, topology/action gate, 36-run reduced cross-fold, corrected bootstrap, H0–H3와 weak-label audit 준비 결과를 시간순으로 추가 | Graph-CLaD 연구팀 |
| 4.0 | 2026-08-16 | 초기 PDF 계획, v3 로드맵, corrected GNN 결과, H0–H3 결과, Stage 2 controlled reimplementation 결정을 통합한 새 canonical 계획서 작성 | Graph-CLaD 연구팀 |
