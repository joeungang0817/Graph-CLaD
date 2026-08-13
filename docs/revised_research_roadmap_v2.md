# Graph-CLaD 수정 연구 로드맵 v2

> **2026-08-06 상태 변경:** 이 문서는 scripted Phase 2R과 최초 Phase 3 설계를
> 기록하는 historical roadmap이다. 현재 실행은 official demonstration의 exact state
> replay, temporal relation event, full timeline, multi-horizon manifest를 채택한
> `docs/revised_research_roadmap_v3.md`를 따른다.

작성일: 2026-08-05  
적용 범위: Phase 2R부터 Graph-CLaD Stage 1 통합까지  
근거: G-DOOM 및 GNN relational classifier 문헌 재검토

## 연구 질문

동일한 CLaD baseline과 동일한 observation/action data를 사용할 때, object-relation
graph가 flat latent transition보다 미래의 의미적 상태 변화를 더 정확하게 예측하고
그 개선이 policy 성능으로 이어지는가?

비교의 핵심은 graph를 사용했다는 사실 자체가 아니라 다음 세 가지다.

1. action 이후 **관계 변화**를 더 잘 예측하는가;
2. 올바른 edge/message/action이 제거되거나 섞일 때 성능이 떨어지는가;
3. 그 표현을 CLaD latent foresight에 넣었을 때 matched MLP adapter보다 일관되게
   개선되는가.

## 고정 통제 원칙

- 원 연구자에게 받은 세 baseline 코드는 변경하지 않고 reference로 보존한다.
- baseline 재현 조건, seed, task split, observation/action preprocessing을 graph 모델과
  동일하게 유지한다.
- oracle simulator graph와 RGB에서 추정한 graph의 결과를 혼합하지 않는다. 첫 실험은
  oracle graph의 상한선 검증이다.
- train/validation/test는 episode 단위로 분리한다.
- GNN과 MLP에는 동일한 원시 정보와 가능한 한 유사한 parameter budget을 제공한다.
- 각 단계 gate를 통과하기 전에는 다음 통합 단계의 positive claim을 하지 않는다.

## Phase 2R — trajectory와 oracle graph dataset 보수

### 목적

기존 정적 extractor를 action-conditioned 미래 graph 예측에 사용할 수 있는 dataset으로
확장한다.

### 작업

1. CLaD horizon에 맞춘 trajectory 수집
   - demonstration/replay의 nonzero action을 사용한다.
   - 각 sample에 episode/task/step ID, `G_t`, action window, `G_{t+tau}`를 저장한다.
   - 기본 `tau=6`과 함께 1/3/6-step target을 만들 수 있도록 중간 graph도 보존한다.

2. node selection v2
   - primary: robot, movable objects, receptacle/fixture, task-relevant sites
   - ablation: all sites 포함
   - logical name은 alignment key로만 쓰고 숫자 model feature에는 넣지 않는다.
   - `is_object_of_interest`는 core input에서 제외하고 oracle flag ablation으로만 둔다.

3. frame과 feature contract
   - primary pose는 robot-base/task-local frame을 사용한다.
   - raw world pose와 runtime body/site ID는 audit payload에만 둔다.
   - orientation은 6D representation과 validity mask를 준비하되 relation-only baseline과
     분리해 ablation 가능하게 한다.

4. relation labeler
   - geometry: left/right, front/behind, above/below
   - contact: MuJoCo contact pair
   - semantic/task: on, inside, holding, open, close
   - relation별 definition version, threshold/margin, validity mask를 기록한다.
   - inverse/symmetry 규칙과 capability applicability를 unit test로 고정한다.

5. split과 통계
   - episode split manifest를 먼저 생성한다.
   - normalization과 threshold calibration은 train split에서만 수행한다.
   - relation별 positive/negative/change count를 보고 희소 label을 확인한다.

6. 시각 검증
   - 최소 두 episode에서 action 전/후 graph와 changed relations를 시각화한다.
   - logical ID alignment, frame 방향, edge 방향을 사람 눈으로 확인한다.

### 완료 gate

- nonzero action trajectory에서 미래 relation change가 실제로 존재한다.
- episode leakage 검사와 시간 정렬 검사가 통과한다.
- 모든 relation은 value와 validity를 구분하며 unknown을 false로 바꾸지 않는다.
- 두 시점 graph 시각화와 확대된 unit test suite가 통과한다.
- 수집 coverage가 부족한 relation은 학습 metric에서 제외하거나 별도 sparse category로
  명시한다.

## Phase 3 — offline relational dynamics probe

### 비교 모델

| ID | 모델 | 목적 |
|---|---|---|
| P0 | parameter-matched flat MLP | graph 없는 핵심 baseline |
| P1 | node encoder + no message passing | node set만으로 충분한지 확인 |
| P2 | fully connected GNN, empty edge input | relational-classifier 논문에 가까운 baseline |
| P3 | fully connected GNN, geometry edge input | 수작업 geometry edge의 추가 이득 확인 |
| P4 | P2/P3 + learned soft edge attention | G-DOOM식 edge selection 효과 확인 |
| P5 | P4 + recurrent/history state | partial observability/history 효과 확인 |

### 공통 입력과 target

- 입력: 동일한 current/past node state와 동일한 action window
- primary target: action 이후 pairwise relation
- auxiliary target: current relation, future latent graph consistency
- optional target: displacement/orientation; relation target과 분리해 ablation

### 손실

기본 후보는 다음과 같다.

```text
L = lambda_now * BCE(current_relations)
  + lambda_future * BCE(future_relations)
  + lambda_dyn * latent_consistency
  + lambda_aux * optional_pose_or_contrastive_loss
```

초기 실험에서는 term을 한꺼번에 모두 켜지 않고, future relation BCE를 중심으로
`latent_consistency`, history, contrastive objective를 차례로 추가한다.

### 필수 control

- no action / shuffled action
- no message passing
- correct topology 대 shuffled sender/receiver assignment
- relation label permutation sanity check
- all-site 대 task-relevant-node graph
- empty edge 대 geometry edge
- latent consistency loss 제거

`shuffled-edge`는 두 참고 논문의 직접 재현 항목이 아니라 Graph-CLaD가 graph 구조를
실제로 사용하는지 확인하기 위한 본 연구의 추가 control로 표기한다.

### 지표

- macro F1 및 relation별 F1
- action 이후 relation F1
- changed-relation subset의 precision/recall/F1
- 1/3/6-step horizon별 성능
- latent consistency 또는 retrieval metric
- seed별 평균과 변동
- parameter count, training time, sample count

현재 relation detection은 보조 지표다. Phase 3의 주 지표는 반드시 **action 이후
relation 예측**이어야 한다.

### 완료 gate

Graph encoder가 matched MLP보다 미래/changed-relation F1에서 반복적으로 우수하고,
no-action 또는 shuffled-edge control에서 성능이 유의미하게 하락해야 한다. 이 조건을
만족하지 못하면 CLaD 통합보다 graph definition, action alignment, relation coverage를
먼저 재검토한다.

## Phase 4 — Graph-CLaD Stage 1 통합

### 통합 위치

CLaD의 latent foresight 의미를 보존하면서 graph transition representation을 semantic
transition residual에 주입한다. 구체적인 tensor 위치는 baseline code tracing으로 확정하고
기존 path를 삭제하지 않는다.

개념적 구조는 다음과 같다.

```text
graph_context = GraphTransitionEncoder(G_{t-tau:t}, a_{t-tau:t})
z_graph = projection(graph_context)
z_transition_graph = z_transition_clad + alpha * z_graph
```

- `alpha=0` 또는 adapter disabled일 때 baseline과 수치적으로 동일해야 한다.
- graph dimension과 baseline latent dimension은 projection layer에서만 맞춘다.
- matched MLP adapter도 동일한 위치와 비슷한 parameter 수로 연결한다.
- relation/dynamics auxiliary loss는 policy loss와 분리해 on/off ablation이 가능해야 한다.

### 비교군

1. 원본 CLaD
2. CLaD + matched MLP transition adapter
3. CLaD + graph adapter, empty edges
4. CLaD + graph adapter, learned/geometry edges
5. 4번의 no-action 또는 shuffled-edge control
6. 필요하면 history encoder 추가 모델

### 완료 gate

- adapter-off equality test 통과
- 동일 seed/task/evaluation protocol에서 baseline 재현 범위 유지
- graph adapter가 matched MLP와 control보다 우수
- offline relation 개선과 policy success 개선의 상관관계를 보고
- 계산량과 parameter 증가를 함께 공개

## Phase 5 이후 — Stage 2 확장 조건

Phase 3과 Phase 4가 모두 통과한 경우에만 Stage 2의 privileged signal 또는 더 복잡한
graph supervision을 검토한다. oracle graph가 offline probe에서조차 matched MLP를
이기지 못하면 perception graph로 확장하지 않는다.

## 실행 시 참조 파일

- 문헌 판단: `docs/literature_alignment_and_phase_reassessment.md`
- GraphSpec 초안: `configs/phase2_graph_spec_v2_draft.json`
- 기존 정적 baseline: `configs/phase2_graph_spec.json`
- 누적 연구 기록: `docs/research_log.md`

이 문서는 실행 중 발견된 사실에 따라 version을 올리되, 이미 수행한 실험 결과는
`docs/research_log.md`에서 덮어쓰지 않고 날짜순으로 추가한다.
