# 교수 관점 연구 리스크 점검 — 2026-08-07

## 한 줄 판단

현재 결과는 **official demonstration에서 holding-positive support를 만들고 action-conditioned relation 예측을 검증할 수 있는 실험 기반을 확보했다**는 점에서는 의미가 있다. 다만 아직 **Graph-CLaD 또는 GNN의 유효성**을 주장할 단계는 아니다. 가장 먼저 고쳐야 할 것은 모델보다 sampling, 평가 분포, fold 정의, 통계 단위다.

## 현재까지 확실히 말할 수 있는 것

- task 0·1·2 official demonstration의 exact state replay를 기반으로 holding-positive, holding-changed, contact-without-holding hard-negative를 만들었다.
- target-aligned release는 13,978개 transition sample을 포함하며 모든 task와 train/validation/test split에서 holding-positive와 holding-changed support가 확인됐다.
- balanced-v3 조건에서 changed-relation macro F1은 P0 flat MLP 0.408, P3 geometry GNN 0.392, P4 soft-attention GNN 0.386이었다.
- 같은 조건에서 P0의 holding future F1은 0.621, holding changed F1은 0.770이었다.
- test-time action 제거 및 shuffle에서 성능 감소가 관찰돼 action input에 이용 가능한 신호가 있다는 정황은 있다.
- 현재 complete geometry graph와 message passing은 flat baseline보다 우월하지 않았다.

## 교수님이 바로 지적할 가능성이 큰 문제

### 1. balanced-v3는 엄밀한 balanced sampling이 아니다

600개 sample에서 task별 category count 합이 670, 689, 664이다. 한 sample이 여러 category에 동시에 속하는 multi-label count이므로, 이 수치를 서로 배타적인 class balance처럼 해석할 수 없다. 실제 목표는 category별 최소 support 120을 확보한 것이지 네 category를 동일 비율로 맞춘 것이 아니다.

추가로 Colab의 category quota 구현은 매 sample 선택 후 episode 순회를 처음부터 다시 시작한다. 따라서 quota 단계는 이름과 달리 true episode round-robin이 아니며 앞쪽 episode를 먼저 소진할 수 있다. 기존 결과 재현용 구현은 `category_aware_v1`으로 보존했고, 수정 구현은 `category_aware_episode_round_robin_v2`로 분리했다.

### 2. target label로 test sample을 고른 challenge-set 평가다

`future_holding_positive`와 `holding_changed`는 `graph_target` label을 보고 선택한다. 이 정보가 model input에는 들어가지 않으므로 feature leakage는 아니지만, 평가 분포를 target-conditioned하게 만든다. 따라서 balanced-v3 성능은 자연 발생 빈도에서의 일반 성능이 아니라 의도적으로 holding event를 증폭한 challenge-set 성능이다.

향후에는 다음 두 평가를 분리해야 한다.

- natural test: 고정된 official demo test episode의 모든 적합 transition
- holding challenge test: target label로 구성한 positive/hard-negative event subset

### 3. 현재 fold는 표준적인 leave-one-task-out 학습이 아니다

각 fold에서 한 task를 train, 다른 task를 validation, 나머지 task를 test로 사용한다. 즉 두 task로 학습하고 하나를 held-out test하는 구조가 아니라 one-task-to-one-task transfer에 가깝다. 특히 task 0이 두 fold의 train으로 재사용되어 fold 간 대칭성도 약하다.

task가 세 개뿐인 상태에서 task-level train/validation/test를 모두 분리하려면 train family가 하나밖에 남지 않는다. 더 타당한 선택은 held-out task 하나를 test로 고정하고 나머지 두 task를 train으로 사용한 뒤, train task 내부 episode 일부를 validation으로 분리하는 것이다. task-level validation까지 요구한다면 더 많은 task family가 필요하다.

### 4. 9개 run을 9개의 독립 표본처럼 해석하면 안 된다

모델별 9개 값은 3개 fold와 3개 seed의 조합이다. 서로 다른 fold와 seed를 한꺼번에 평균한 표준편차는 seed uncertainty와 task-transfer variance를 섞는다. 또한 인접 horizon/window는 동일 demo와 event를 공유하므로 sample 단위 독립성도 없다.

보고 시에는 다음이 필요하다.

- fold별 3-seed 평균
- 동일 fold·seed에서 P0 대비 paired difference
- episode 또는 event 단위 bootstrap confidence interval
- task별 결과를 숨기지 않은 표

### 5. P0 우세가 model architecture 때문인지 shortcut 때문인지 불명확하다

P0는 정렬된 전체 node set을 한 번에 보므로 object ordering, identity 위치, scene-level layout을 직접 사용할 수 있다. 반면 GNN은 complete graph에서 552개 edge message를 섞는다. P0의 holding changed F1이 0.770인데 P1/P2가 거의 0이라는 큰 격차는 P0가 유용한 global signal을 잡았다는 뜻일 수도 있지만, node ordering 또는 task-specific shortcut 가능성도 확인해야 한다.

필수 control은 node permutation, non-target object masking, target-object-only MLP다. 이것 없이 “flat representation이 본질적으로 더 적합하다”고 결론 내리기 어렵다.

### 6. test-time perturbation만으로 action causality를 주장할 수 없다

no-action과 shuffled-action은 학습 분포 밖의 입력을 test 때만 넣는다. 성능 하락은 model이 action feature를 사용한다는 증거지만, 올바른 action–state 대응을 인과적으로 학습했다는 증거는 아니다.

다음에는 train-time no-action model, train-time shuffled-action model, 동일한 current state에 서로 다른 action을 대응시킨 paired counterfactual 평가가 필요하다.

### 7. holding label은 heuristic weak label이다

holding은 contact, gripper closure, relative-pose stability, object-following evidence로 생성한 상태기계 label이다. exact simulator-state replay는 state 복원이 정확하다는 뜻이지 holding semantic label이 사람 기준으로 정확하다는 뜻은 아니다.

최소한 task별 onset/release/hard-negative sample을 수동 검토하고 event-level precision 또는 agreement를 보고해야 한다. hard-negative도 “두 endpoint 중 한 곳에서 contact이고 holding은 양쪽 모두 false”이므로 실제 grasp 직전의 어려운 negative인지 별도 검증이 필요하다.

### 8. 모델 간 평균 차이가 불확실성보다 작다

P0와 P3의 changed-relation macro F1 차이는 약 0.016, P0와 P4 차이는 약 0.022다. P3/P4의 표준편차는 각각 약 0.110과 0.103으로 훨씬 크다. 현재 집계만으로 P0의 통계적 우위를 선언하거나 GNN이 부적합하다고 단정할 수 없다.

## 현재 결과로 하지 말아야 할 주장

- GNN은 holding prediction에 부적합하다.
- action-conditioning이 인과적으로 입증됐다.
- balanced sampling으로 일반 성능이 개선됐다.
- official CLaD 성능을 재현하거나 개선했다.
- 45 runs가 45개의 독립 실험이다.

## 권장 결론 문장

> Official demonstration 기반 target-aligned holding challenge set에서 flat MLP가 현재 geometry-based GNN보다 높은 평균 성능을 보였다. 이 결과는 complete spatial message passing의 이점이 아직 입증되지 않았음을 의미하며, GNN 일반의 부적합성을 의미하지는 않는다. 다음 실험은 sampling과 fold를 수정한 고정 평가에서 holder–object 중심 edge와 action-conditioned temporal update의 기여를 검증한다.

## 다음 gate

1. `category_aware_episode_round_robin_v2` sampler QA를 수행하되 기존 v3 결과와 섞지 않는다.
2. natural test와 target-conditioned holding challenge test를 분리한다.
3. held-out task test + remaining two tasks train + episode-level validation protocol을 고정한다.
4. task별 label 수동 audit를 수행한다.
5. target-object-only MLP, node permutation, non-target masking control을 추가한다.
6. holder–object edge와 action-conditioned temporal edge를 구현한 뒤 geometry GNN과 paired 비교한다.

