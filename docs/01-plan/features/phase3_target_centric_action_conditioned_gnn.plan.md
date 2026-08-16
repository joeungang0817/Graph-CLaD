# Phase 3B-R1: target-centric action-conditioned GNN 연구 계획

작성일: 2026-08-11  
상태: 실행 전 승인 기준 계획  
적용 범위: official LIBERO demonstration 기반 oracle relational forward dynamics  
선행 문서: `docs/literature_alignment_and_phase_reassessment.md`,
`docs/phase3_holder_object_action_graph_design.md`

## 1. 최종 결정

최종 후보는 **target-centric holder-object topology와 action-conditioned temporal edge를 결합한 모델**이다. 다만 두 변경을 처음부터 하나의 모델로만 실험하지 않는다. 먼저 평가와 shortcut control을 고정하고, 그다음 아래 2 x 2 구조로 두 가설의 단독 효과와 결합 효과를 분리한다.

| | action을 마지막 예측부에만 사용 | action을 message/update에 직접 사용 |
|---|---|---|
| complete graph | G00: 현재 geometry GNN 기준선 | G01: action-conditioning 단독 효과 |
| holder-object sparse graph | G10: target-centric topology 단독 효과 | G11: 두 아이디어를 결합한 최종 후보 |

예상 최종 후보는 G11이지만, 결론은 G00·G01·G10·G11의 고정 평가 결과로 결정한다. 이 구조는 다음 세 질문에 각각 답할 수 있다.

1. G10 대 G00: 불필요한 complete edge를 제거하는 것만으로 좋아지는가?
2. G01 대 G00: complete topology를 유지해도 action을 message에 넣으면 좋아지는가?
3. G11 대 G10 및 G01: sparse topology와 edge-level action conditioning의 결합 이점이 있는가?

즉, 기존에 제안한 baseline·shortcut·평가 고정 절차를 **실험 프레임워크**로 사용하고, target-centric edge와 action-conditioned temporal edge를 **그 안에서 검증할 모델 가설**로 사용한다.

## 2. 이 결정을 내린 근거

### 2.1 현재 결과가 말하는 것

- official demo task 0·1·2에서 target-aligned holding dataset 13,978개 index row와 holding-positive, holding-changed, contact-without-holding hard negative support를 확보했다.
- category-aware v3 challenge 조건에서 P0 flat MLP가 changed-relation macro-F1 0.4081, holding future F1 0.6210, holding changed-subset future-value F1 0.7701로 가장 높았다.
- P3/P4의 평균은 P0보다 낮았지만 fold 변동이 커서 “GNN이 부적합하다”고 결론 내릴 수 없다.
- 모든 모델에서 correct action이 no-action보다 나았고, correct action과 shuffled action 차이는 작았다. action에 이용 가능한 정보는 있지만 올바른 action–state 대응을 충분히 학습했다는 근거는 약하다.
- 현재 P3/P4 구현은 두 번의 message-passing 동안 action을 사용하지 않는다. action은 message가 모두 섞인 뒤 마지막 edge 예측부에만 붙는다. 따라서 현재 결과는 action-conditioned graph update 자체를 시험한 결과가 아니다.
- 현재 input-clean sample은 8개 node와 56개 complete directed edge를 사용한다. Holding은 주로 `robot0`과 각 movable object의 관계인데, 모든 object/fixture pair의 message를 두 번 섞으면 holding 신호가 희석될 수 있다.

### 2.2 데이터가 허용하는 범위

현재 sample에는 `graph_t`, target 구간의 `action_window`, `graph_target`이 있다. 완전한 past graph sequence는 없다. 따라서 첫 모델은 현재 시점에서 얻을 수 있는 위치·거리·접촉·gripper 상태와 action window만 사용한다. 상대 속도, object-following stability, recurrent history는 데이터 계약을 확장한 뒤 별도 ablation으로 미룬다.

이 실험은 target 구간의 실행 action을 입력받는 **forward dynamics(View A)** 실험이다. future action을 볼 수 없는 CLaD foresight(View B)와 동일한 주장으로 해석하지 않는다.

## 3. 세 선택지 비교

### 선택지 A: target-centric edge만 사용

장점은 complete graph의 불필요한 message를 제거하고 holding의 실제 물리 단위인 holder–object pair에 집중한다는 것이다. 구현과 해석도 비교적 단순하다.

한계는 action이 여전히 마지막 예측부에만 들어가면 “gripper를 닫고 들어 올리는 action이 어느 object edge를 어떻게 바꾸는가”를 message 단계에서 표현하지 못한다는 점이다. 따라서 필요한 첫 ablation이지만 최종 구조로 바로 확정하기에는 불완전하다.

### 선택지 B: action-conditioned temporal edge만 사용

장점은 현재 구현의 가장 직접적인 약점인 late action concat을 고친다는 것이다. action에 따라 edge message의 크기와 방향을 바꿀 수 있다.

한계는 complete graph의 모든 pair가 그대로 남는다는 것이다. 데이터가 많지 않고 action 신호도 약한 현재 조건에서는 모델이 action으로 필요한 edge를 스스로 찾아내는 동시에 relation dynamics까지 학습해야 한다. 이는 학습 부담이 크고, 개선되더라도 graph topology가 필요한지 설명하기 어렵다.

### 선택지 C: 두 아이디어를 결합

Holding의 구조와 가장 잘 맞는 최종 가설이다. target-centric topology가 “어떤 관계를 볼 것인가”를 제한하고, action-conditioned update가 “그 관계가 action에 따라 어떻게 변할 것인가”를 모델링한다.

다만 결합 모델 하나만 실행하면 어느 요소가 성능을 만들었는지 알 수 없다. 따라서 **G11을 최종 후보로 두되 G00·G01·G10을 함께 두는 2 x 2 검증**이 가장 좋은 선택이다.

## 4. leakage 없는 target-centric 정의

여기서 target-centric은 미래에 실제로 들린 object를 label에서 골라 입력한다는 뜻이 아니다. 그렇게 하면 target leakage가 된다.

첫 구현은 다음처럼 정의한다.

- `robot0`과 모든 movable object 사이를 candidate holder–object pair로 만든다.
- 모든 모델은 동일한 candidate pair와 동일한 current-state/action 정보를 받는다.
- `graph_target`, holding event category, future relation, reward, success, BDDL goal은 입력에 사용하지 않는다.
- primary output은 각 `robot0`–object pair의 holding/contact 및 holding change-event다.
- task instruction에서 target object를 얻는 기능은 현재 실험에 넣지 않는다. 나중에 넣는다면 모든 baseline에 동일하게 제공하는 별도 language-conditioned ablation으로 취급한다.

따라서 “특정 pair 중심”은 oracle target 하나를 미리 선택하는 것이 아니라, **각 candidate pair를 독립적인 예측 질의로 다룬다**는 의미다.

## 5. 모델 정의

### B0: flat MLP

현재 가장 강한 baseline이다. 전체 정렬 node set과 action을 사용한다. 기존 수치와 연결하는 기준점으로 유지한다.

### B1: pair-only MLP

각 `robot0`–object candidate pair에 대해 robot feature, object feature, current pair feature, action만 사용한다. 다른 object의 message를 받지 않는다. 이 모델이 G10/G11과 비슷하다면 graph message-passing이 아니라 pair feature가 성능의 핵심이라는 뜻이다.

### G00: complete + late/global action

현재 P3 geometry GNN에 해당한다. complete graph message-passing 후 action을 예측부에 붙인다. 새 protocol에서 다시 측정하는 graph 기준선이다.

### G10: sparse holder–object + late/global action

message topology만 `robot0`과 movable object의 양방향 star로 제한한다. action 경로는 G00과 동일하게 유지해 topology 변경 효과만 본다.

### G01: complete + action-conditioned update

complete topology를 유지하고 각 edge message/gate에 action encoding을 직접 넣는다. topology는 고정하고 action routing의 효과만 본다.

### G11: sparse holder–object + action-conditioned update

최종 후보다. 각 candidate holder–object edge의 current relation, robot/object 상태, action window를 함께 사용해 edge gate와 message를 만들고 residual node update를 수행한다.

첫 버전은 한 개의 sparse interaction block만 사용한다. 깊은 GNN, recurrent history, unrestricted fixture/site context는 넣지 않는다. 필요하면 G11이 통과한 뒤 직접 contact/support가 있는 context edge만 추가한 G12를 별도 검증한다.

### 첫 버전의 허용 feature

- current robot/object node feature와 validity mask
- gripper–object relative position, distance, geometry validity
- current contact value와 validity
- gripper qpos/closure 상태
- target interval action window와 action mask
- object/type validity 정보

첫 버전에서 제외하는 feature는 future graph, future event tag, oracle target object ID, reward/success, past sequence가 필요한 상대 속도·stability다.

## 6. 모델보다 먼저 고정할 평가 protocol

### 6.1 sampler와 sample ID

- 기존 balanced-v3는 `category_aware_v1` 결과로 보존한다.
- corrected `category_aware_episode_round_robin_v2`의 quota, uniqueness, episode coverage QA를 먼저 통과시킨다.
- 기존 v4 config의 old fold를 그대로 45회 재실행하는 것은 최종 실험이 아니다. sampler 구현 검증에만 사용하고, 최종 학습은 새 fold/evaluation manifest를 사용한다.
- 각 fold의 train, validation, natural test, holding challenge test sample ID를 JSON manifest와 hash로 고정한다.
- 모든 모델과 seed가 정확히 같은 ID를 사용한다.

### 6.2 outer task split

각 fold에서 task 하나를 완전히 held-out task로 둔다. 나머지 두 task의 기존 train demo를 학습에, 두 task의 기존 validation demo를 validation에 사용한다. held-out task의 demo는 학습과 checkpoint 선택에 사용하지 않는다.

세 held-out task를 각각 한 번씩 평가하고, task별 결과를 먼저 보고한다. 세 task/fold를 아홉 개의 독립 표본처럼 합치지 않는다.

### 6.3 natural test와 challenge test

- natural test: held-out task의 고정 demo에서 얻은 모든 적합 transition. category balancing을 적용하지 않는다.
- holding challenge test: 같은 held-out task에서 미리 고정한 future holding positive, holding change, hard-negative event subset.
- 학습용 category-aware selection과 test challenge selection을 구분해 기록한다.
- 두 test의 결과를 별도 표로 보고하며 하나의 평균으로 합치지 않는다.

### 6.4 horizon

첫 primary 비교는 `tau=6` 한 horizon으로 고정해 ragged action padding과 horizon 혼합을 제거한다. G11 후보가 통과한 뒤 `tau=1/3/6`을 각각 보고한다. 여러 horizon을 한 모델에 넣을 때는 action mask와 horizon 정보를 모든 모델에 동일하게 제공한다.

### 6.5 label audit

Holding은 simulator의 명시적 ground-truth predicate가 아니라 contact, closure, relative motion을 조합한 weak label이다. 학습 전에 task별 onset, release, hard negative를 최소 10개 event씩 수동 확인한다. 잘못된 label 또는 애매한 hard negative가 반복되면 모델을 바꾸기 전에 state-machine/event rule을 고친다.

## 7. 반드시 먼저 수행할 shortcut baseline

1. B0 flat MLP
2. B1 pair-only MLP
3. B0 node-order randomized training/evaluation
4. B0 non-target object masking

해석은 다음과 같다.

- B1이 B0와 비슷하면 전체 scene graph는 필요하지 않고 pair feature가 핵심일 가능성이 높다.
- node-order randomization에서 B0가 크게 하락하면 현재 P0 우세에 정렬/identity shortcut이 포함됐을 가능성이 있다.
- non-target mask에서 B0가 크게 하락하면 제한된 context가 필요하다는 근거가 된다.
- B1보다 G10/G11이 좋아야 message-passing의 추가 가치를 주장할 수 있다.

## 8. 실행 순서와 gate

### Gate E0: 평가 고정

- v2 sampler QA 통과
- split/sample manifest hash 저장
- natural/challenge test 분리
- task별 relation/event support 확인
- holding 수동 audit 완료
- leakage와 episode overlap 0건

E0가 통과하기 전에는 모델 결과를 최종 비교로 사용하지 않는다.

### Gate E1: shortcut 진단

B0, B1, node-randomized B0, non-target-masked B0를 `test_task2`, seed 0으로 기술 smoke run한다. 이 한 번의 결과는 성능 결론이 아니라 입력·metric·training 경로 검증용이다. 이후 세 held-out task의 seed 0 pilot으로 shortcut 패턴이 task 하나에만 생기는지 확인한다.

### Gate E2: 2 x 2 architecture pilot

G00, G01, G10, G11을 동일 fold·seed·parameter budget에서 실행한다. 먼저 `test_task2`, seed 0으로 shape/gradient/metric/control smoke를 통과시키고, 이후 세 held-out task의 seed 0 pilot을 수행한다.

모델 선택을 위해 결과가 좋은 셀만 사후적으로 남기지 않는다. 기술 오류가 없는 네 셀을 같은 protocol로 보고해 topology main effect, action-routing main effect, interaction을 분리한다.

### Gate E3: full comparison

최종 비교 대상은 B0, B1, G00, G01, G10, G11이다. 세 held-out task와 세 seed를 사용하므로 총 54개 main run이다. smoke/pilot checkpoint가 최종 config와 완전히 같으면 재사용하고, 다르면 별도 기록한다.

보고 단위는 다음과 같다.

- task별 3-seed 평균과 표준편차
- 같은 task·seed의 paired difference
- episode/event 단위 hierarchical bootstrap confidence interval
- parameter count, update 수, training time, inference latency

### Gate E4: action·edge 통제 강화

G11이 E3에서 유망할 때만 수행한다.

- correct, zero, within-task shuffled action test
- G11과 가장 강한 non-graph baseline의 no-action retrain
- G11과 가장 강한 non-graph baseline의 shuffled-action retrain
- holder–object pair identity/edge attribute shuffle
- action window와 sample ID가 유지되는 paired-control QA

Test-time action perturbation은 action reliance로, retrained control과 paired counterfactual은 더 강한 action-conditioning 근거로 구분해 보고한다.

### Gate E5: 제한적 context 또는 history

다음 조건에서만 확장한다.

- non-target masking에서 명확한 하락이 있으면 G11에 직접 contact/support context edge만 추가해 G12를 비교한다.
- action-conditioned G11이 topology baseline보다 좋아도 onset/release timing이 부족하면 past graph sequence를 새 데이터 contract로 추가한다.
- G11이 B1을 이기지 못하면 깊은 GNN을 쌓지 않고 pair model을 주 결과로 받아들인다.

## 9. metric을 정확히 구분한다

현재 보고서의 `holding_changed_f1`은 실제 change-event classifier F1이 아니라 true-changed subset에서의 future holding value F1이다. 다음 실험에서는 이름과 의미를 분리한다.

### Primary

- actual holding change-event F1: 모든 valid pair에서 current와 future의 변화 여부를 예측
- holding onset F1과 release F1
- hard-negative holding false-positive rate
- holding PR-AUC

### Secondary

- future holding state F1
- future holding value F1 on true-changed subset: 기존 `holding_changed_f1`의 명확한 새 이름
- robot–object pair의 changed-relation macro-F1
- natural test 전체 future-relation macro-F1
- relation별 support, precision, recall, F1, PR-AUC

Threshold는 validation에서만 정하고 test에 고정한다. support가 없는 relation은 macro 평균에 0점으로 넣지 않는다.

## 10. 성공 기준

G11을 다음 단계 후보로 선택하려면 최소한 다음을 모두 만족해야 한다.

1. G11 대 B1의 holding change-event paired difference가 세 held-out task 중 최소 두 task에서 양수다.
2. episode/event hierarchical bootstrap에서 G11 대 B1의 pooled primary-metric 차이가 양수 방향으로 안정적이다.
3. G11 대 G10에서 action-conditioned update의 추가 이점이 있고, correct action이 within-task shuffled action보다 일관되게 낫다.
4. G11 대 G01에서 sparse topology의 추가 이점이 있거나, 적어도 hard-negative false-positive rate를 낮춘다.
5. holder–object edge/pair shuffle에서 primary metric이 하락한다.
6. challenge test 개선이 natural test의 holding 성능을 희생해서 생긴 것이 아니다. natural primary metric의 허용 회귀 폭은 사전에 절대값 0.02로 고정한다.
7. 특정 task 하나의 큰 개선만으로 전체 성공을 선언하지 않는다.

이 기준을 통과하지 못했을 때의 결론도 명확하다.

- G10만 개선: topology 선택은 유효하지만 edge-level action conditioning은 입증되지 않음.
- G01만 개선: action routing은 유효하지만 sparse target topology는 필요하지 않음.
- G11이 가장 좋지만 interaction이 불안정: 결합 모델은 후보이나 추가 task/seed가 필요함.
- B1이 모든 GNN과 같거나 우수: 현재 holding 문제에는 graph message-passing의 추가 가치가 없고 pair model이 더 적절함.

## 11. 이번 단계에서 하지 않는 것

- `inside` relation 재도입
- 여러 GNN layer와 대규모 soft-attention 탐색
- focal loss, sampler, architecture를 동시에 변경
- 새 official demo 수집
- past graph history를 현재 sample에 있는 것처럼 가정
- Phase 4 CLaD adapter 통합
- policy success 개선 주장

Loss weighting이나 focal loss는 G00·G01·G10·G11의 구조 비교가 끝난 뒤 동일 winner에 대한 별도 ablation으로만 수행한다.

## 12. 저장 및 재현 계획

로컬 source of truth는 이 계획서와 이후 생성할 config/manifest다. Colab 결과는 다음 새 root 아래에 저장한다.

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1`

최소 저장물은 다음과 같다.

- protocol config와 SHA256
- train/validation/natural-test/challenge-test sample ID manifest
- sampler QA와 relation/event support report
- 모델별 checkpoint와 run config
- task·seed별 raw metrics와 paired analysis
- runtime manifest와 사용 코드 hash

기존 `phase3`, `phase3_holding_target_v2`, `phase3_holding_target_balanced_v3` 결과는 수정하거나 덮어쓰지 않는다.

## 13. 바로 다음 작업

다음 구현 순서는 모델 코드가 아니라 평가 protocol 고정이다.

1. corrected sampler QA 결과를 생성한다.
2. 새 outer-task fold와 natural/challenge sample ID manifest를 만든다.
3. `tau=6` relation/event support와 holding 수동 audit 대상을 확정한다.
4. 실제 change-event metric과 hard-negative FPR 계산을 먼저 추가한다.
5. B1 pair-only MLP와 shortcut controls를 구현한다.
6. 그다음 G00·G01·G10·G11 2 x 2 smoke를 수행한다.

이 순서는 sampler 변화, P0 shortcut, graph topology, action routing을 서로 다른 원인으로 해석할 수 있게 만드는 최소한의 순서다.
