# Graph-CLaD 연구 전과정 점검 및 로드맵 변경 분석

작성일: 2026-08-06  
점검 범위: 제공된 CLaD 코드 3개, Phase 0-3 구현과 테스트, 결과 JSON, 연구노트,
연구실행계획서, CLaD/G-DOOM/GNN relational classifier 문헌 정리  
결론 문서: `docs/revised_research_roadmap_v3.md`

## 1. 종합 판정

현재 연구는 실패한 것이 아니다. 다만 지금까지의 작업은 다음 두 종류로 분리해야
한다.

1. **계속 사용할 수 있는 기반**
   - 제공된 CLaD 핵심 모듈의 코드 경계와 미확인 사항 정리
   - LIBERO simulator state 접근 확인
   - logical ID, robot-base 좌표계, validity mask를 사용하는 정적 oracle graph
   - contact/on/inside/holding handler의 실행 경로와 구조 검증
   - episode/task-family split, train-only normalization, parameter-count 기록을 포함한
     offline probe 실행 골격

2. **본 실험의 증거로 사용하면 안 되는 결과**
   - bounded 또는 scripted probe action으로 만든 trajectory
   - `contact AND closed_gripper`만으로 만든 holding label
   - 단일 `tau=6` pair만 저장한 Phase 2R dataset
   - 현재 구현의 `changed_relation` 수치를 실제 relation-change event 예측 F1로
     해석하는 것
   - complete graph에서 현재 방식의 shuffled sender/receiver 결과를 곧바로
     “올바른 graph edge의 인과적 효과”로 해석하는 것

따라서 다음 단계는 Phase 4가 아니라 **Phase 2D: official demonstration 기반
relation-event graph dynamics dataset 재구축**이다. Phase 3 기존 결과는 가설과
도구를 점검한 diagnostic baseline으로만 보존하고, Phase 2D 데이터 QA 후 다시
실행해야 한다.

## 2. 단계별 상태 재판정

| 단계 | 현재 판정 | 유지할 산출물 | 제한 또는 수정점 |
|---|---|---|---|
| Phase 0 | 통제된 smoke gate 통과 | 원본 3개 파일, shape/loss/EMA 검사, `unknowns.md` | 공식 CLaD 재현은 아님. trainer, VLM, Stage 2가 없음 |
| Phase 1A | 통과 | simulator/object/robot state 접근 계약 | demo HDF5와 state replay 경로는 아직 구현되지 않음 |
| Phase 2A | 통과 | 정적 GraphSpec, logical ID, robot-base transform, unknown/false 분리 | `phase2.v1`의 `is_object_of_interest`는 신규 model input에서 제거해야 함 |
| Phase 2R prototype | 계측용 통과 | action/graph 시간 정렬, semantic handler, scripted probe 회귀검사 | 본 학습 데이터로 부적합. state replay/history/multi-horizon/event index가 없음 |
| Phase 3 historical | 실행 완료, 연구 gate 미통과 | 공통 trainer, task-held-out 실행 경험, 실패한 control 결과 | 데이터와 metric/control 정의를 고친 뒤 재실행 필요 |
| Phase 4 | 차단 | 기존 residual fusion 아이디어 | Phase 2D와 새 Phase 3 gate 통과 전 시작하지 않음 |

## 3. 잘한 선택과 계속 유지할 원칙

### 3.1 원본 코드와 통제된 재구현을 분리한 점

제공된 세 파일만으로는 공식 CLaD trainer, VLM pipeline, Stage 2 diffusion policy를
복원할 수 없다. 이를 숨기지 않고 공식 수치 재현이 아닌 **같은 자체 재구현 환경의
아키텍처 비교**로 범위를 제한한 것은 타당하다. Phase 4 이후에도 이 주장 범위를
유지해야 한다.

### 3.2 simulator oracle과 RGB perception을 분리한 점

oracle graph를 먼저 검증하는 것은 graph representation의 가치와 perception 오류를
분리한다. 신규 demo dataset도 simulator state에서 만든 oracle label을 사용하되,
model input과 label 생성용 privileged metadata를 엄격히 분리해야 한다.

### 3.3 logical identity, robot-base frame, validity mask

runtime body ID 대신 logical ID로 시간축을 맞추고, world pose를 robot-base pose로
변환하며, unavailable relation을 false가 아닌 invalid로 둔 결정은 모두 유지할 가치가
있다. 이는 새 state-replay extractor의 기반이 된다.

### 3.4 실패한 control을 gate 실패로 인정한 점

Phase 3에서 no-action/shuffled-action이 correct action보다 나쁘지 않았고,
shuffled-edge도 안정적으로 하락하지 않았다는 사실을 숨기지 않은 것은 중요하다.
이 결과는 “GNN이 효과가 없다”는 최종 결론보다 **현재 데이터가 action-conditioned
dynamics를 식별할 만큼 구성되지 않았다**는 진단으로 해석하는 것이 맞다.

## 4. 핵심 간극과 더 나은 방향

### 4.1 데이터 원천: scripted probe보다 official demonstration

현재 collector는 bounded probe 또는 물체 중심으로 접근하는 scripted holding probe를
실행한다. 이는 contact mapping, 좌표 변환, handler 단위검사에는 유용하지만 인간의
성공적인 manipulation transition 분포를 나타내지 않는다. action이 반복적이고 관계
변화가 action과 약하게 연결되어 있어 no-action/shuffled-action control이 약해질 수
있다.

개선 방향은 task 0·1·2의 official human-teleoperation demonstration 전체를 원천으로
사용하는 것이다. scripted capture는 `smoke/regression fixture`로 남기고 Phase 3의
train/validation/test에는 포함하지 않는다.

### 4.2 replay 방식: action replay보다 state replay 우선

저장된 action을 현재 simulator/controller에서 다시 실행하면 버전과 수치 오차가 누적되어
원래 demonstration과 다른 trajectory가 될 수 있다. 정식 label 생성은 각 시점의 저장된
MuJoCo state를 직접 복원하고 `sim.forward()` 후 graph evidence를 추출하는 방식이
우선이다. 저장 action은 `G_t -> G_{t+tau}`의 조건 입력으로 사용한다.

action replay는 일부 episode에서 다음을 확인하는 consistency audit로만 사용한다.

- 저장 state의 `t+1`과 action replay의 `t+1` 차이
- controller/action dimension과 timing 계약
- 환경·LIBERO·robosuite·MuJoCo 버전 차이에 따른 drift

### 4.3 holding: 단일 frame predicate가 아니라 temporal state

현재 `scripts/phase2r_relation_handlers.py`의 fallback holding은 robot-object contact와
gripper qpos threshold의 conjunction이다. 이 규칙은 순간 충돌, 물체 밀기, 빈손 close를
구분하지 못하고 finger별 contact provenance도 잃는다.

신규 holding label은 다음 evidence를 이용한 state machine으로 바꿔야 한다.

```text
free -> contact_candidate -> holding -> release
```

- `contact_candidate`: finger-object contact는 있으나 상대 pose 안정성과 follow evidence가
  아직 부족함
- `holding`: finger contact, sufficiently closed gripper, K-frame object-EFF relative pose
  안정성, object-follow 또는 lift evidence를 모두 충족
- `release`: contact가 끊기거나 relative pose가 hysteresis threshold를 넘어 변함
- 애매한 transition/boundary frame: `valid=false`; 억지로 negative로 만들지 않음

양쪽 finger contact는 high-confidence evidence로 사용하되, 물체 형상 때문에 항상
hard requirement로 두지는 않는다. palm/wrist collision과 left/right finger contact를
raw provenance에 분리해 저장해야 한다.

### 4.4 dataset 단위: positive frame 집합보다 full timeline + event index

holding=true frame만 따로 저장하면 이미 grasp가 끝난 상태의 분류는 학습할 수 있지만,
grasp가 **언제 발생하는지** 또는 action이 어떤 변화를 일으켰는지 배우기 어렵다.

더 나은 구조는 물리 데이터를 중복 저장하지 않고 다음 두 계층을 갖는 것이다.

1. episode 전체의 canonical graph timeline
2. contact/holding/lift/release/on/inside/spatial-change event를 가리키는 sample index

학습 sampler가 positive event, hard negative, background의 선택 확률을 조절한다. test는
전체 timeline의 자연 분포와 event-centered subset을 모두 보고한다.

### 4.5 Phase 2R 구현과 로드맵의 계약 차이

현재 `build_phase2r_dataset.py`는 `graph_t`, 길이 `tau`의 action window,
`graph_target=graph_{t+tau}`만 저장한다. 로드맵에 있던 past graph sequence, 1/3/6
horizon target, recurrent history는 실제로 구현되지 않았다.

신규 dataset은 horizon 1/3/6과 history를 지원하는 참조형 sample을 만들어야 한다.
graph payload 자체는 episode timeline에 한 번만 저장하고 sample manifest는 frame
index를 참조해야 한다. 그래야 저장 중복 없이 G-DOOM식 history/recurrent 모델과
multi-horizon 모델을 같은 데이터에서 비교할 수 있다.

### 4.6 forward dynamics와 CLaD foresight의 시간축 차이

`G_t + action_{t:t+tau-1} -> G_{t+tau}`는 저장된 미래 action을 조건으로 하는
forward-dynamics 문제다. GNN relational classifier 방식의 action-conditioned relation
prediction을 검증하기에는 적합하다. 그러나 CLaD Stage 1은 현재 시점에서 과거 상태와
과거 action transition을 보고 미래 latent를 예측한다. 배포 시점에는 아직 실행하지 않은
`action_{t:t+tau-1}`가 없으므로 이 action window를 Graph-CLaD adapter에 그대로 넣으면
future-action leakage가 된다.

따라서 하나의 canonical demo timeline에서 두 sample view를 별도로 만들어야 한다.

```text
View A - forward dynamics:
G_t + action_{t:t+tau-1} -> G_{t+tau}

View B - CLaD-aligned foresight:
G_{t-h:t} + action_{t-h:t-1} -> G_{t+tau}
```

View A는 graph와 action이 relation dynamics를 설명할 수 있는지 확인하고, View B는 그
graph representation이 CLaD와 같은 causal information boundary에서 유효한지 확인한다.
View A의 성공만으로 Phase 4에 들어가지 않고 View B의 bridge gate도 통과해야 한다.

### 4.7 기존 changed-relation metric의 이름과 의미

현재 Phase 3 코드는 실제로 변한 label 위치만 mask한 다음 **미래 relation value의 F1**을
계산한다. 특히 true->false offset은 positive class가 아니므로, 이 값은 relation-change
event를 직접 예측한 F1이 아니다. 따라서 과거 결과의 `changed_relation_macro_f1`은
`future_value_f1_on_true_changed_subset`으로 재해석해야 한다.

신규 평가는 다음을 분리한다.

- 전체 frame의 future relation macro/per-relation F1
- 실제 changed subset에서의 future-value F1
- `predicted_change = current_value XOR predicted_future_value`로 계산한 change-event F1
- relation별 onset F1과 offset F1
- 희귀 event의 PR-AUC와 event-time tolerance F1

### 4.8 complete graph와 shuffled-edge control

complete directed graph에는 이미 가능한 모든 sender-receiver pair가 들어 있다. 따라서
edge feature가 비어 있는 모델에서 topology만 “shuffle”하는 것은 올바른 edge와 틀린
edge의 대비가 명확하지 않다. 또한 기존 inference-time roll은 self-loop/중복 여부와
label alignment가 control 목적과 정확히 일치하는지 추가 검사가 필요하다.

수정된 control은 다음처럼 역할을 분리한다.

- empty-edge complete GNN: `no-message` 또는 message block 제거가 주 control
- geometry/semantic edge model: node-pair와 edge attribute의 대응을 깨는
  `edge-attribute permutation`
- learned sparse edge를 도입한 경우에만 degree와 self-loop 조건을 보존한
  topology corruption 사용
- 모든 control은 실제 입력이 달라졌는지 checksum/통계로 검사

기존 shuffled-edge 결과는 exploratory control로 보존하되 논문의 직접 재현이나 graph
causality의 최종 증거로 사용하지 않는다.

### 4.9 non-graph baseline의 공정성

현재 P0 flat MLP는 전체 padded node tensor를 flatten하지만 edge head에는 각 edge의
source/target node feature를 직접 주지 않는다. 반면 P1-P4는 source/target encoding을
받는다. 또한 P0는 node ordering에 민감하다. 따라서 P0와 GNN의 차이가 graph message
passing뿐이라고 보기 어렵다.

다음 재실행에서는 P0를 historical baseline으로 남기고, **pairwise MLP 또는 DeepSets
context + pair query**를 primary non-message baseline으로 둔다. 모든 모델은 같은 node,
action, history, edge geometry 정보에 접근하고, parameter count는 목표 대비 ±5% 이내로
맞추며 FLOPs와 latency도 함께 보고한다.

### 4.10 indirect task leakage

`is_object_of_interest`를 0으로 덮은 것은 올바른 수정이지만, BDDL goal을 사용해
“task-relevant node만 선택”하면 node set 자체가 goal shortcut이 될 수 있다. 신규 primary
graph의 node inventory는 goal과 무관한 scene entity/capability 규칙으로 결정한다.

- model input 금지: BDDL goal, task ID, logical name embedding, object-of-interest flag,
  reward, success, terminal, future state
- label 생성/감사에만 허용: BDDL relation applicability, success 확인, object-of-interest
- BDDL-derived task-relevant graph는 privileged oracle ablation으로만 보고

### 4.11 provenance와 재현성

현재 repository에는 demo HDF5 원본이나 checksum manifest가 없고, 일부 상세 결과는
Colab runtime 절대경로에만 남아 있다. 신규 dataset에는 다음 provenance가 필요하다.

- 원본 HDF5 경로/파일 checksum/demo key
- task/BDDL checksum과 environment metadata
- LIBERO, robosuite, MuJoCo, controller 버전 또는 commit
- state replay 성공/실패와 action replay drift
- graph spec, relation labeler, threshold, split manifest version
- raw contact geom/body name과 logical mapping

대용량 HDF5를 Git에 넣을 필요는 없지만, 다운로드 출처와 checksum manifest는 반드시
repository에 남겨야 한다.

## 5. 갱신된 데이터 설계의 핵심

정식 데이터셋의 성격은 다음과 같이 고정한다.

> **LIBERO Demo-derived, Event-centered, Multi-horizon, Oracle Relational
> Dynamics Dataset**

### 저장 계층

| 계층 | 내용 | 목적 |
|---|---|---|
| Raw manifest | HDF5/demo/task/version/checksum | 원천 재현성 |
| Replayed state timeline | state, action, robot/EFF, raw contacts, qpos, poses | oracle evidence |
| Graph timeline | robot-base `G_0...G_T`, relation value/valid/confidence/evidence | 중복 없는 canonical data |
| Event index | contact/holding/lift/release/on/inside/spatial onset·offset | event-centered sampling |
| Window manifest | history와 horizon 1/3/6 frame reference | 모델 입력/target |
| Split manifest | demo ID 기반 in-task/task-generalization split | leakage 방지 |
| QA report | coverage, flicker, manual audit, replay consistency | Phase 3 진입 gate |

### sample 종류

| 종류 | 예시 | 역할 |
|---|---|---|
| Positive event | grasp 후 물체가 EFF를 따라 lift | onset/dynamics 학습 |
| Hard negative | 접촉했지만 밀기만 함, 빈손 close, 순간 collision | false positive 억제 |
| Background | relation 변화가 없는 일반 구간 | 전체 dynamics 분포 유지 |
| Ambiguous audit | state-machine 경계 또는 evidence 불일치 | loss에서 mask, label 품질 분석 |

## 6. split과 평가 protocol 변경

label threshold와 event sampling 전에 demo ID split을 먼저 고정한다.

1. **In-task protocol**
   - task 0·1·2 각각 demo ID 단위 train/validation/test
   - 같은 demo의 모든 frame/window는 하나의 split에만 속함

2. **Task-generalization protocol**
   - task 0·1 train
   - task 0·1의 held-out demo로 validation
   - task 2는 마지막 test

3. **Train-only fitting**
   - normalization, gripper threshold, K, pose-stability margin, hysteresis는 train demo로만
     정함
   - task 2 결과를 보고 threshold를 수정하지 않음

4. **Relation applicability**
   - task 2에 holding event가 실제로 없다면 이를 실패한 수집으로 간주해 억지로
     positive를 만들지 않음
   - 해당 protocol에서 holding task-generalization metric은 `N/A: no positive support`로
     보고하고, in-task 또는 holding-support가 있는 별도 held-out task에서 평가

## 7. 권장 실행 순서

1. official task 0·1·2 demo HDF5와 environment metadata 확보
2. raw manifest와 demo-ID split을 먼저 생성
3. 소수 demo의 exact state replay와 state/action timing 검증
4. raw finger/contact provenance를 포함한 graph timeline 생성
5. temporal holding 및 on/inside/contact event labeler 구현
6. full trajectory coverage와 event audit 통과
7. history `h`와 horizon 1/3/6에 대해 forward-dynamics view와 CLaD-aligned view 생성
8. 수정된 baseline/control/metric으로 Phase 3 action-conditioned probe 재실행
9. CLaD-aligned causal-input bridge를 별도로 검증
10. 두 gate 통과 후에만 Graph-CLaD Stage 1 adapter 구현

세부 gate와 파일 계획은 `docs/revised_research_roadmap_v3.md`를 기준으로 한다.

## 8. 이번 점검의 범위 제한

- official demonstration HDF5는 아직 workspace에 없으므로 실제 state replay와 relation
  coverage는 이번 점검에서 실행하지 않았다.
- 로컬 재검증 시 기본 `python` 실행 환경을 사용할 수 없었고 bundled Python에는
  `pytest`가 없어 전체 test suite를 다시 실행하지 못했다. 이번 변경은 문서만 수정하며,
  기존 Colab 실행 결과를 새 데이터셋의 통과 증거로 재사용하지 않는다.
- Phase 4 이후의 전체 CLaD trainer와 Stage 2는 여전히 제공 코드에 없으며, 이후에도
  controlled reimplementation으로만 주장해야 한다.
