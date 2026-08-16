# CLaD 통제 재구현: 확인된 사항과 미확인 사항

이 저장소는 통제된 재구현 연구다. 제공된 Python file 세 개는 Stage 1 model core를
정의하지만 전체 training/evaluation pipeline을 정의하지 않는다. 확인되지 않은 내용을
임의로 채우지 않고 여기에 기록한다.

## 제공 코드에서 확인한 사항

- `LatentDynamics.forward`는 visual history, proprioception history, action tensor 하나,
  language feature, 선택적인 future target을 입력으로 받는다.
- Visual view는 `s_backbone` 전에 flatten된다.
- Semantic/proprioceptive transition block은 설정된 hidden dimension의 4-head
  cross-attention 다섯 layer를 사용한다.
- Proprioceptive transition token이 semantic transition token을 query한다.
- Evaluation은 `[B, H]` shape의 `pred_p_emb`, `pred_s_emb`를 반환한다.
- Training은 `loss_p`, `loss_s`, `loss_p_recon`, `loss_v_recon`을 반환한다.
- EMA target encoder는 있지만 caller가 `update_ema()`를 호출해야 한다.

## Phase 0 가정

- 논문 설정은 `H=1024`, semantic feature dimension 1024, view 2개, action horizon 6,
  EMA momentum 0.995, reconstruction weight 0.1을 사용한다.
- `s_backbone`의 `input_dim=vl_dim * 2`이므로 제공 코드는 `V * Dv = 2 * vl_dim`을 요구한다.
- Semantic backbone output과 `s_query`를 더할 때 `vl_dim = hidden_dim`이어야 한다.
- Smoke test는 CPU forward/backward 확인을 위해 작은 `H=64`를 사용한다. 이는 code
  path 검사이지 production feature dimension의 타당성 검증이 아니다.

## 여전히 검증되지 않은 사항

- VLM preprocessing, frozen encoder, language representation, view order.
- 실제 proprioception/action dimension.
- Action history를 `prev_action`에 묶는 방법.
- 실제 history 길이와 temporal sampling.
- Trainer loss weighting, optimizer, scheduler, checkpoint, EMA 호출 순서.
- 연결된 transition sequence의 masking이 action만 가리려는 의도인지 여부. 제공 함수는
  첫 token만 보호한다.
- 제공 코드의 비대칭 target normalization이 의도적인지 여부.
- View가 여러 개인데 visual reconstruction이 `v_next[:, 0]`만 쓰는 이유.
- 모든 Stage 2 diffusion policy와 rollout 세부사항.

## Phase 2D demonstration data에서 확인해야 했던 사항

공식 demonstration을 조사하기 전에는 다음을 scripted Phase 2R capture에서 추측하면
안 됐다.

- Task 0/1/2 HDF5의 정확한 local/Drive 경로, checksum, task mapping.
- HDF5 `states[t]`가 `actions[t]` 전인지 후인지에 대한 timing.
- 받은 LIBERO/robomimic 버전의 environment metadata와 reset/state restore API.
- Simulator forward 이후 qpos, EEF pose, object pose의 replay tolerance.
- Runtime의 left/right finger, palm, wrist geom/body 이름.
- Task/split별 자연 holding/on/inside event support.
- Train-only holding parameter: contact persistence, K-frame relative-pose stability,
  follow/lift margin, release hysteresis.
- Task 2의 holding-positive demonstration 존재 여부. 없으면 test-dependent threshold로
  고치지 말고 unsupported coverage로 보고해야 한다.

이 항목의 실행 결정 기록은 `docs/revised_research_roadmap_v3.md`다.

## 2026-08-07 이후 상태

범위를 정한 `libero_spatial` release에서는 task 0/1/2 HDF5 mapping, persistent Drive
경로, exact state replay, robot/gripper contact mapping, fixed split, holding-positive
support를 확인했다. Task 2에도 target-aligned artifact에서 holding-positive와
holding-changed support가 존재했다. 남은 질문은 data access가 아니라 validity와
evaluation design으로 이동했다.

- Heuristic holding onset/release label의 event-level precision.
- 현재 hard negative가 실제로 grasp와 혼동하기 어려운 사례인지 여부.
- Natural prevalence와 target-conditioned stress view 성능 차이.
- Task family가 세 개뿐일 때의 방어 가능한 held-out-task protocol.
- P0가 node ordering이나 다른 scene shortcut을 쓰는지 여부.
- Train-time control에서도 action 의존성이 유지되는지 여부.
- Target-centric holder-object와 action-conditioned temporal edge가 target-object-only,
  flat baseline보다 나은지 여부.

현재 위험은 `docs/01-plan/features/graph-clad-integrated-research-v4.plan.md`,
`docs/revised_research_roadmap_v3.md`, `docs/CURRENT_STATUS.md`,
`docs/phase3_holder_object_action_graph_design.md`에서 추적한다.

## 2026-08-16 Stage 2 구현 결정과 남은 불확실성

공식 Stage 2 source는 여전히 없지만 제출용 통제 비교의 구현 방향은 정했다. Stage 1을
freeze하고 current observation과 predicted foresight를 modality별 FiLM으로 결합한
canonical DDPM Diffusion Policy를 사용하며 action horizon `tau=6`, epsilon-prediction
objective를 우선 적용한다. 이 선택은 공식 코드 확인이 아니라 논문 설명에 따른
CLaD-compatible 재구현 가정이다. Network 세부 폭/깊이, noise schedule, inference step,
rollout wrapper, checkpoint 선택은 아직 config로 확정해야 한다.

## 통제 연구 원칙

Baseline 완성에 필요한 가정은 config로 고정하고 baseline CLaD와 Graph-CLaD에 동일하게
적용한다. Official CLaD metric은 참고값일 뿐이며 이 저장소는 통제된 재구현 환경
내부의 비교를 보고한다.
