# 문헌 기반 Phase 1·2 재평가

> **2026-08-06 후속 결정:** 이 문서의 action-conditioned Phase 2R 제안은 scripted
> prototype까지 실행되었다. 현재는 official LIBERO demonstration 전체의 state replay와
> temporal relation event를 사용하는 `docs/revised_research_roadmap_v3.md`가 실행
> 기준이다. 이 문서는 G-DOOM과 relational-classifier 문헌 판단의 역사적 근거로 남긴다.

작성일: 2026-08-05  
검토 문헌:

- G-DOOM, *Learning Latent Graph Dynamics for Visual Manipulation of Deformable Objects* (arXiv:2104.12149v2)
- *Planning for Multi-Object Manipulation with GNN Relational Classifiers* (arXiv:2209.11943v2)

## 결론

기존 작업은 실패한 것이 아니라 두 개의 하위 단계까지만 통과한 상태다.

- **Phase 1a — simulator state 접근 확인:** 통과
- **Phase 1b — 행동이 포함된 trajectory와 관계 label 데이터 계약:** 미완료, 재개 필요
- **Phase 2a — 단일 snapshot의 정적 GraphSpec과 extractor:** 통과
- **Phase 2b — oracle graph dataset 및 미래 관계 예측 준비:** 미완료, 재개 필요

따라서 기존 산출물은 정적 그래프 기준선으로 보존하되, 바로 CLaD에 GNN을 붙이지
않고 `Phase 2R`에서 데이터와 target을 먼저 보강한다.

## 논문에서 직접 확인된 내용

### 1. G-DOOM

G-DOOM은 deformable object를 sparse keypoint graph로 표현하고, 현재 graph와 action,
recurrent history를 이용해 미래 latent graph를 여러 step rollout한다.

핵심 구조는 다음과 같다.

- 모든 keypoint 쌍을 연결한 graph에서 learned soft edge weight를 사용한다.
- 각 node update에 action을 condition한다.
- observation history를 recurrent hidden state로 요약해 self-occlusion과 partial
  observability를 처리한다.
- graph 전체 표현은 단순 mean/max가 아니라 learned pooling으로 만든다.
- 미래 graph와 실제 미래 graph의 pooled latent 사이에 contrastive dynamics loss를
  적용하고 reward prediction을 보조 학습한다.
- no-graph, no-RNN, no-contrastive, 다른 attention/pooling을 제거한 ablation을 통해
  graph, history, dynamics objective가 각각 필요한지 검사한다.

LIBERO는 rigid object와 안정적인 logical ID를 제공하므로 G-DOOM의 “시간 간
keypoint identity를 직접 맞추지 않는다”는 선택을 그대로 복사할 필요는 없다. 반면
**action conditioning, history, multi-step rollout, learned edge, dynamics loss**는
Graph-CLaD의 latent foresight를 설계할 때 직접 참고할 수 있다.

### 2. GNN relational classifiers

이 논문은 variable-size object point cloud를 fully connected directed graph로 만들고,
pairwise relation을 예측한다. 입력 edge는 비어 있고, GNN이 만든 latent node/edge에서
relation classifier가 작동한다.

- 관계는 left/right, behind/in-front, above/below, contact의 7개 independent binary
  label이다. 서로 배타적인 softmax가 아니라 relation별 sigmoid를 사용한다.
- skill/action embedding을 condition한 latent dynamics가 node와 edge latent의
  residual change를 예측한다.
- 현재 relation loss, 미래 latent graph consistency loss, action 이후 relation loss를
  함께 사용하며 여러 planning step으로 recursive unroll한다.
- 현재 relation **검출**에서는 MLP도 GNN과 비슷했지만, action 이후 relation
  **예측**에서는 GNN이 훨씬 강했다. 따라서 현재 snapshot relation F1만으로 graph의
  유효성을 주장할 수 없다.
- pose-only supervision과 latent regularization 제거 모델을 포함한 ablation을
  수행한다. relation supervision과 latent consistency가 planning 성능에 중요했다.

중요한 정정이 있다. 이 논문은 **correct-edge 대 shuffled-edge 실험을 직접 보고하지
않는다.** 주요 비교는 relational-dynamics GNN, MLP, pose 계열 모델, latent
regularization 제거 모델이다. correct/shuffled edge는 여전히 좋은 인과적 control이지만,
해당 논문의 결과를 재현하는 항목이 아니라 본 연구가 추가하는 검증으로 표기해야 한다.

## Phase 1에 반영할 수정

### 유지할 것

- simulator raw state와 observation schema를 먼저 기록한 방식
- cross-episode identity로 `logical_id`를 사용하고 `body_id`를 audit metadata로만
  유지한 방식
- 누락값을 0으로 위장하지 않고 value와 validity mask를 함께 저장한 방식
- privileged oracle state와 실제 RGB-policy input을 구분한 통제 조건

### 추가하거나 변경할 것

1. **zero-action 두 snapshot을 실제 action trajectory로 교체**
   - reset과 zero action은 extractor 점검에는 충분하지만 미래 dynamics supervision이
     되지 않는다.
   - demonstration 또는 replay action을 사용해 `G_t`, action window,
     `G_{t+tau}`를 같은 episode에서 수집한다.
   - CLaD의 기본 horizon과 맞춰 우선 `tau=6`을 저장하고, 1/3/6 step 평가도 가능하게
     중간 snapshot을 보존한다.

2. **episode 단위 split manifest 추가**
   - 같은 episode의 인접 frame이 train과 validation/test로 나뉘면 미래 상태가 거의
     복제되어 leakage가 발생한다.
   - normalization, relation threshold calibration, label statistics도 train episode에서만
     계산한다.

3. **전용 relation labeler 정의**
   - 객체 wrapper마다 무차별적으로 `is_open` 등을 호출하는 generic probe를 학습
     label로 사용하지 않는다.
   - 기하 관계, MuJoCo contact, task predicate를 capability별 handler로 분리하고
     relation별 `value`, `valid`, `definition/version`을 기록한다.
   - 기본 후보는 left/right, front/behind, above/below, contact, on, inside, holding,
     open, close다. 적용 불가능한 관계는 false가 아니라 invalid다.

4. **좌표계와 orientation 계약 확정**
   - primary 좌표계는 robot-base 또는 task-local frame으로 고정하고 각 축 의미를
     문서화한다. raw world pose는 audit용으로 보존한다.
   - quaternion ordering을 runtime test로 확정한 뒤 필요한 경우 6D rotation으로
     변환한다. orientation은 relation-only probe의 필수 gate는 아니며 별도 ablation이
     가능하다.

5. **action schema 저장**
   - raw action, controller 종류, action dimension, action interval, gripper component를
     episode metadata로 함께 저장한다.
   - 미래 graph target과 action이 정확히 같은 시간 구간을 가리키는지 검사한다.

## Phase 2에 반영할 수정

### 유지할 것

- fully connected directed graph를 첫 topology baseline으로 둔 선택
- self-loop 제외, `target - source` 방향을 명시한 edge convention
- 안정적인 logical identity와 missing-value mask
- 현재 v1 extractor를 재현 가능한 정적 baseline으로 보존하는 것

### 추가하거나 변경할 것

1. **node 범위 축소 및 통제**
   - 현재 graph는 24개 node 중 site가 16개라 message passing을 지배하거나
     task-layout shortcut을 만들 수 있다.
   - primary graph는 robot, movable object, receptacle/fixture, task에 필요한 site만
     사용한다. all-site graph는 ablation으로 둔다.
   - `is_object_of_interest`는 instruction에서 얻은 privileged shortcut일 수 있으므로
     core model feature에서 제외하고 별도 oracle ablation으로만 사용한다.

2. **relation target과 edge input을 분리**
   - paper-faithful baseline은 fully connected topology에 empty edge input을 사용하고
     relation을 prediction target으로 둔다.
   - geometry edge feature를 쓰는 모델은 별도 variant로 둔다. relation label과 거의
     같은 상대 위치/거리를 GNN에만 주고 MLP에는 주지 않는 비교는 금지한다.
   - GNN과 MLP는 가능한 한 같은 raw information과 비슷한 parameter budget을
     받아야 한다.

3. **action-conditioned transition target 추가**
   - 정적 `GraphExtractor`와 별도로 `GraphTransitionDataset` 계약을 만든다.
   - 입력은 과거/current graph와 action window, target은 future graph의 relation 및
     latent state다.
   - 현재 relation BCE, post-action relation BCE, latent consistency를 기본 loss 후보로
     둔다. object displacement/orientation은 선택적 보조 target으로 분리한다.

4. **history와 learned edge는 단계적 ablation으로 추가**
   - 첫 baseline은 stable logical ID를 이용한 두 시점 graph다.
   - 그 다음 recurrent/history encoder와 learned soft-edge attention을 추가해
     G-DOOM의 기여를 각각 검증한다.

5. **relation-change 중심 평가 추가**
   - macro/per-relation F1뿐 아니라 `relation changed` subset의 F1을 따로 측정한다.
   - unchanged relation이 많은 데이터에서 accuracy나 전체 F1만 보면 trivial predictor가
     좋아 보일 수 있다.
   - 1/3/6-step horizon별 degradation과 multi-step rollout stability를 기록한다.

6. **필수 테스트 확대**
   - node permutation equivariance
   - logical ID의 시간축 alignment
   - inverse relation consistency: left(i,j)=right(j,i) 등
   - variable node count와 padding mask
   - action/target 시간 정렬과 no-future-leakage
   - episode split isolation
   - multi-step unroll shape와 mask propagation
   - correct/shuffled edge, no-action, no-message control이 실제로 다른 입력을 만드는지
   - GNN/MLP feature parity와 parameter-count 기록

## 현재 산출물에 대한 최종 판정

| 기존 산출물 | 판정 | 이후 사용법 |
|---|---|---|
| Phase 1 live state capture | 유효 | API와 schema 증거로 보존 |
| reset + zero-action 두 snapshot | 제한적 | extractor smoke test에만 사용 |
| `phase2.v1` GraphSpec | 유효한 정적 baseline | 최종 dynamics spec으로 간주하지 않음 |
| 24-node/552-edge live graph | 유효한 실행 증거 | all-site oracle graph ablation으로 보존 |
| generic predicate audit | 진단에는 유효 | 학습 label로 사용 금지 |
| Phase 2 unit test 4개 | 통과 | dynamics/data leakage test를 추가해야 최종 gate 통과 |

후속 실행 순서는 `docs/revised_research_roadmap_v3.md`를 기준으로 한다.
`docs/revised_research_roadmap_v2.md`와
`configs/phase2_graph_spec_v2_draft.json`은 scripted Phase 2R의 historical contract로
보존한다.
