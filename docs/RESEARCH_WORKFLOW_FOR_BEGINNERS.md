# 처음 보는 사람을 위한 Graph-CLaD 연구 설명서

이 문서는 Graph-CLaD 연구를 처음 접한 연구자가 코드보다 먼저 읽는 입문서다.
현재 실행 상황은 `CURRENT_STATUS.md`, 세부 코드 역할은
`CODEBASE_GUIDE_FOR_BEGINNERS.md`를 함께 본다.

## 1. 이 연구가 해결하려는 문제

로봇이 물체를 조작하려면 단순히 현재 이미지가 무엇처럼 보이는지만 알아서는 부족하다.
다음과 같은 변화를 이해해야 한다.

- 로봇 손이 어느 물체에 접근했는가?
- 물체와 접촉했는가?
- 물체를 실제로 들고 있는가?
- 물체가 어디에서 어디로 이동했는가?
- 물체 사이의 `left`, `right`, `on`, `near` 같은 관계가 어떻게 변했는가?
- 수행한 action이 어떤 상태 변화를 일으켰는가?

기존 CLaD는 관측에서 얻은 semantic feature와 action history를 이용해 미래 상태를
나타내는 latent foresight를 학습한다. 하지만 semantic vector 하나만 보면 어느 물체의
어떤 관계가 변했는지 명시적으로 분해되어 있지 않다.

이 연구의 핵심 질문은 다음과 같다.

> 장면을 물체와 관계로 이루어진 graph로 표현하고, action에 따른 graph transition을
> 학습하면 기존 semantic transition보다 robot–object interaction과 spatial transition을
> 더 잘 보존할 수 있는가?

## 2. 핵심 용어

### Semantic representation

이미지나 관측 전체를 하나 또는 몇 개의 고차원 vector로 압축한 표현이다. 장면의 의미를
풍부하게 담을 수 있지만 특정 물체와 특정 관계를 직접 찾아보기는 어렵다.

### Object–relation graph

장면을 node와 edge로 표현한다.

- Node: 로봇, 물체, fixture, site.
- Node feature: 위치, gripper 상태, joint 상태, 유효성 mask 등.
- Edge: `robot→object`, `object→object`처럼 두 node의 방향성 있는 쌍.
- Edge feature: 상대 위치, 거리, 접촉, spatial relation 등.

예를 들어 `robot0 → mug` edge에는 로봇 손과 머그컵의 상대 위치, 접촉 여부,
holding 여부를 기록할 수 있다.

### Transition representation

현재 상태 자체가 아니라 현재에서 미래로 무엇이 바뀌는지를 담는 표현이다. 이 연구에서는
`graph_t`, action window, `graph_target`의 관계를 학습한다.

### Latent foresight

미래 관측을 직접 생성하는 대신 미래에 필요한 상태 정보를 latent vector로 예측하는
CLaD Stage 1의 출력이다. Stage 2 policy는 이 foresight를 조건으로 action을 생성한다.

### Diffusion Policy

행동 sequence에 noise를 넣고 제거하는 과정을 학습해 action chunk를 생성하는 policy다.
향후 이 프로젝트에서는 Stage 1을 freeze하고 foresight를 조건으로 사용하는 canonical
DDPM Diffusion Policy를 통제 구현할 예정이다.

## 3. 전체 연구 흐름

```text
LIBERO official demonstration
  -> simulator state replay
  -> 각 frame의 object–relation graph
  -> current graph + action window + future graph sample
  -> architecture/protocol screening
  -> semantic / pair-local / graph representation 비교
  -> CLaD Stage 1 latent foresight 통합
  -> 동일한 Diffusion Policy Stage 2 연결
  -> 환경 rollout success와 robustness 비교
```

연구는 한 번에 최종 policy를 만드는 방식이 아니다. 데이터와 label이 유효한지, graph가
실제로 필요한지, action이 올바른 방식으로 사용되는지를 작은 gate에서 확인한 뒤 다음
단계로 넘어간다.

## 4. 왜 holding을 먼저 보는가

`holding`은 최종 연구 목표가 아니라 architecture probe다. 다음 이유로 중간 검사가
가능하다.

- 로봇과 특정 물체의 상호작용을 요구한다.
- `not holding → holding` onset과 `holding → not holding` release가 명확한 transition이다.
- contact만으로는 충분하지 않아 시간 정보가 필요하다.
- action, gripper closure, object motion이 함께 관련된다.
- false positive가 발생하기 쉬운 hard negative를 만들 수 있다.

따라서 holding을 잘 예측하는지 보면 representation이 robot–object transition을 어느
정도 보존하는지 진단할 수 있다. 그러나 holding 성능만으로 object–relation graph
representation 전체가 우수하다고 결론 내릴 수는 없다.

## 5. Holding label이 weak label인 이유

현재 holding label은 사람이 모든 frame을 직접 판정한 ground truth가 아니다. 다음
trajectory evidence를 조합한 heuristic label이다.

- robot/gripper와 object의 contact.
- gripper가 닫혀 있는지.
- 최근 3 frame 동안 gripper–object 상대 위치가 안정적인지.
- object 움직임이 end effector 움직임을 따라가는지.
- release hysteresis가 충족되는지.

자동 생성 label이므로 contact를 grasp로 오인하거나 작은 움직임을 잘못 해석할 가능성이
있다. 그래서 task 0/1/2 각각 onset 10개, release 10개, hard negative 10개인 총
90-item evidence audit를 별도로 준비했다. Human review 전에는 weak label을 정답이라고
표현하지 않는다.

## 6. 데이터가 만들어지는 과정

### 6.1 원천 데이터

LIBERO official demonstration의 HDF5 action/state와 BDDL task description을 사용한다.
Scripted trajectory는 graph extractor를 진단할 때만 사용하고 main training data에는
섞지 않는다.

### 6.2 Episode split을 먼저 고정

하나의 demonstration episode에서 인접한 여러 window가 생성될 수 있다. Window를 만든
뒤 무작위로 train/test를 나누면 같은 episode의 거의 같은 장면이 양쪽에 들어가 leakage가
발생한다. 따라서 episode ID 기준 split을 먼저 고정한 뒤 window를 만든다.

### 6.3 Exact state replay

HDF5의 simulator state를 LIBERO environment에 복원해 frame별 observation, robot state,
object pose, contact 정보를 다시 얻는다. 단순 저장 vector만 읽는 것이 아니라 simulator
상태를 복원해 graph에 필요한 정보를 수집한다.

### 6.4 Graph sample

한 학습 sample의 핵심은 다음과 같다.

```text
graph_t          현재 frame의 node/edge graph
action_window    t부터 t+tau까지의 action chunk
graph_target     target frame의 graph
metadata         task, episode, start_step, target_step, tau, split
```

현재 primary horizon은 `tau=6`이다. Stable sample ID는 suite, task ID, episode ID,
start step, target step, tau의 조합으로 만든다.

### 6.5 Natural dataset과 target-aligned view

- Natural dataset: held-out episode의 transition을 인위적으로 균형화하지 않은 데이터.
- Target-aligned dataset: holding onset/release, future holding positive, hard negative 등을
  더 잘 관찰할 수 있도록 선택한 학습/stress용 view.

Natural test가 primary다. Challenge라고 부르던 view는 natural held-out episode에서
future event를 사용해 고른 부분집합이므로 독립 generalization dataset이 아니다.
현재 문서에서는 challenge stress view라고 부른다.

## 7. Fold가 의미하는 것

이 프로젝트의 fold는 sample을 무작위로 세 조각 낸 것이 아니라 held-out task를 바꾸는
outer split이다.

| Fold | 학습/검증 task | 테스트 task |
|---|---|---|
| `test_task0` | task 1, 2 | task 0 |
| `test_task1` | task 0, 2 | task 1 |
| `test_task2` | task 0, 1 | task 2 |

같은 fold 안의 여러 seed는 같은 test episode를 공유하므로 독립 test 표본이 아니다.
따라서 먼저 task별 seed 평균을 만들고 task fold를 바깥 평가 단위로 해석한다.

## 8. Manifest와 QA

Manifest는 데이터 자체가 아니라 한 실험이 사용할 sample과 split을 고정한 계약서다.
다음을 포함한다.

- Train, natural validation, natural test, stress view sample ID.
- Task와 episode split.
- Category quota와 sampling 결과.
- Natural/stress overlap payload hash.
- Episode leakage 검사.
- Relation label support.
- Source dataset path와 protocol version.

모든 모델이 같은 manifest를 사용해야 모델 차이와 데이터 차이를 분리할 수 있다.
`status=pass`가 아닌 manifest로는 corrected runner가 실행되지 않는다.

## 9. Phase별 연구 목적

### Phase 0 — 제공된 CLaD core 확인

제공받은 `LatentDynamics`, attention, MLP 모듈이 작은 synthetic tensor에서
forward/backward/EMA 계약을 만족하는지 확인한다. 공식 trainer, VLM pipeline,
Stage 2를 재현하는 단계는 아니다.

### Phase 1A — LIBERO state/API 조사

관측 key, robot state, object ID, body/contact mapping, action dimension, simulator API를
확인한다. Graph를 만들기 전에 실제로 읽을 수 있는 정보와 없는 정보를 구분한다.

### Phase 2A — 정적 graph 계약

한 frame의 snapshot을 deterministic object–relation graph로 바꾼다. Node identity에
runtime body index를 쓰지 않고 logical ID를 사용한다. 누락된 값은 0으로 위장하지 않고
validity mask를 함께 둔다.

### Phase 2R — Scripted diagnostic

간단한 scripted trajectory로 graph extractor와 relation handler를 검사한다. Main
training data가 아니며 이 결과만으로 연구 가설을 결론 내리지 않는다.

### Phase 2D — Official-demo temporal dataset

Official demonstration을 exact replay하고 multi-horizon graph sample, holding weak label,
fixed episode split, natural/target-aligned release를 만든다. 실패한 demo는 QA에 남기고
완료된 shard만 merge한다.

### Phase 3A — Dataset과 label QA

Sample/episode leakage, category quota, payload hash, relation support를 검사한다.
Weak-label 90-item audit package도 이 단계에 속한다.

### Phase 3B — Architecture와 action gate

Holding transition을 사용해 어떤 representation 구조가 후보인지 검사한다. Pair MLP,
GNN, no-action, shuffled-action, pair-local history/action 모델을 같은 protocol과 비슷한
parameter 수로 비교한다.

### Phase 3C — CLaD-aligned foresight bridge

아직 미구현이다. 미래 action이나 미래 graph를 encoder 입력으로 사용하지 않고 현재와
과거 정보만으로 future transition에 유용한 latent를 만들 수 있는지 검사한다. Semantic,
pair-local, graph encoder를 freeze한 뒤 같은-capacity probe로 holding, displacement,
source→destination, spatial transition을 비교한다.

### Phase 4 — Graph-CLaD Stage 1 통합

Phase 3C를 통과한 representation을 CLaD latent foresight 구조에 연결한다. 기존 semantic
foresight와 같은 데이터, horizon, latent/probe 조건에서 비교한다.

### Phase 5–7 — Stage 2와 rollout

Stage 1을 freeze하고 canonical DDPM Diffusion Policy를 연결한다. 최종 비교는 다음 세
조건을 동일한 policy capacity와 training/rollout budget으로 수행한다.

1. Policy-only baseline.
2. 기존 semantic foresight policy.
3. 선택된 pair-local 또는 graph foresight policy.

이 단계가 완료돼야 graph/pair representation이 실제 robot action과 task success에
도움이 되는지 평가할 수 있다.

### Phase 8 — 통계와 재현성 보고

Task별 결과, paired difference, hierarchical bootstrap, ablation, failure case, claim
limit를 함께 보고하고 config, manifest, checkpoint, prediction, result, runtime manifest,
code snapshot을 묶는다.

## 10. 모델이 발전한 과정

### B1-v2 pair MLP

Robot/object pair와 상대 geometry를 독립적으로 처리하고 action은 late feature로 넣는다.
단순하지만 강한 non-graph baseline이다.

### G1 late-action sparse GNN

Robot–object sparse topology에서 message passing을 수행한 뒤 action을 prediction head
가까이에 넣는다. Action-conditioned edge message model은 아니다. Corrected 3 folds ×
3 seeds에서 B1보다 일관되게 좋지 않아 깊은 GNN 확대를 중단했다.

### H0–H3 pair-local factorial

Object–object global message passing 대신 robot–object pair를 독립적으로 처리한다.

| 모델 | 과거 causal history | Action |
|---|---:|---:|
| H0 | 없음 | 없음 |
| H1 | 있음 | 없음 |
| H2 | 없음 | 있음 |
| H3 | 있음 | 있음 |

History는 오직 `<=t` frame에서 상대 위치 변화, 상대 속도, contact persistence,
gripper closure velocity, object-following residual과 validity mask를 계산한다.
H3는 action embedding으로 pair token을 FiLM modulation한다.

Three-fold seed-0에서 H3의 mean natural PR-AUC가 가장 높았지만, 올바른 action alignment가
중요한지 확인하기 위해 episode-disjoint matched train-shuffled H3 control이 현재 gate다.

## 11. Corrected evaluation protocol

기존 protocol에서 action이 current auxiliary head에 섞이고 event-enriched validation을
사용하는 confound가 있었다. Corrected protocol은 다음을 고정한다.

- Current relation head는 action-free pair head.
- Checkpoint는 natural validation holding event PR-AUC로 선택.
- Threshold는 natural validation에서 한 번 선택.
- 같은 frozen threshold를 natural test, stress view, control에 적용.
- Natural event PR-AUC를 primary로 사용.
- F1, onset/release, hard-negative FPR, Brier score, ECE를 함께 저장.
- 표본별 probability, target, episode/task/sample ID를 gzip prediction으로 저장.

## 12. 지표를 읽는 법

### PR-AUC

Positive event가 드문 상황에서 threshold를 고정하지 않고 precision–recall tradeoff를
평가한다. 현재 architecture gate의 primary metric이다.

### F1

선택된 threshold에서 precision과 recall의 조화 평균이다. Threshold 선택에 민감하므로
secondary metric으로 본다.

### Onset / Release F1

- Onset: 현재 not-holding이고 미래 holding인 변화.
- Release: 현재 holding이고 미래 not-holding인 변화.

Release가 특히 낮아 현재 주요 병목이다.

### Hard-negative FPR

접촉이나 접근 때문에 holding처럼 보이지만 실제 holding transition이 아닌 sample을
positive로 잘못 예측한 비율이다. 낮을수록 좋다.

### Brier score / ECE

예측 확률이 실제 빈도와 얼마나 잘 맞는지 보는 calibration 지표다. Threshold가 매우
높거나 모델 간 확률 scale이 다를 때 해석에 필요하다.

## 13. Oracle-current event metric의 한계

기존 compatibility event metric은 ground-truth current holding과 predicted future
holding을 XOR해 변화를 계산한다. 즉 현재 symbolic holding 상태를 알고 있다는 조건부
metric이다. End-to-end 환경에서는 current도 예측해야 하므로 predicted current와
predicted future를 함께 쓴 end-to-end event metric도 저장한다.

논문이나 보고서에서는 conditional/oracle-current와 end-to-end를 구분해야 한다.

## 14. Action control이 필요한 이유

모델에 action 입력이 있다고 해서 action 의미를 제대로 사용한다고 볼 수 없다. Action
branch가 capacity만 늘리거나 task shortcut을 제공할 수 있다. 따라서 다음 control을
사용한다.

- No-action model.
- Evaluation-time global shuffled action.
- Train-time episode-disjoint matched shuffled action.
- Constant action.
- Action magnitude/state가 지나치게 다르지 않은 donor matching.

Aligned action이 matched shuffled action보다 같은 fold/seed에서 좋아야 action alignment의
증거가 된다. 그래도 이것만으로 물리적 causal effect 전체를 입증했다고 주장하지 않는다.

## 15. 통계 해석

3 tasks × 3 seeds를 9개의 독립 표본으로 취급하지 않는다. 같은 task의 seeds는 같은 test
episode를 공유한다. 보고 순서는 다음과 같다.

1. Fold/task별 seed 평균.
2. 같은 fold/seed의 paired difference.
3. Task fold를 바깥 단위로 하고 episode/event cluster를 안쪽 단위로 한 hierarchical bootstrap.
4. Mean뿐 아니라 task별 부호와 failure case.

## 16. 최종적으로 무엇을 주장하려는가

최종 representation 비교에서 확인하려는 것은 단순 training loss가 아니다.

- Holding onset/release downstream 성능.
- Object displacement와 source→destination 변화.
- Valid spatial relation transition.
- 적은 labeled data에서의 sample efficiency.
- Hard-negative robustness.
- Action/edge perturbation sensitivity.
- Held-out task의 paired improvement.
- 동일한 Stage 2 policy에서 rollout success 향상.

Graph가 항상 우월하다는 결론이 나오지 않을 수도 있다. Pair-local temporal encoder가 더
좋다면 “holding transition에서는 complete scene message passing보다 pair-local temporal
state가 적합하다”는 방어 가능한 연구 결과가 된다.

## 17. 현재 주장할 수 없는 것

- GNN이 pair MLP보다 일반적으로 우월하다는 주장.
- G1이 action-conditioned temporal edge model이라는 주장.
- Action shuffle 하락만으로 인과 효과가 입증됐다는 주장.
- Stress view를 독립 generalization test라고 부르는 것.
- Weak label을 human ground truth라고 부르는 것.
- 아직 구현되지 않은 Stage 2를 공식 CLaD 재현이라고 부르는 것.
- 현재 3-task protocol을 LIBERO-LONG 공식 결과와 직접 비교하는 것.

## 18. 신규 연구자가 바로 확인할 순서

1. `CURRENT_STATUS.md`에서 live 실행과 다음 gate를 확인한다.
2. 이 문서로 전체 연구 질문과 Phase 구조를 이해한다.
3. `CODEBASE_GUIDE_FOR_BEGINNERS.md`로 관련 구현 파일을 찾는다.
4. `phase3_pair_local_temporal_threefold_seed0_result.md`에서 실제 수치를 확인한다.
5. `01-plan/features/graph-clad-integrated-research-v4.plan.md`에서 아직 gated인 단계와
   성공·중단 기준을 확인한다.
6. `research_log.md`에서 결정이 내려진 시간순 근거를 확인한다.
