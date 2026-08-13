# Phase 1A: LIBERO state 접근과 oracle graph 경계

> **2026-08-06 roadmap v3 참고:** Phase 1A의 live state-access 결과는 유지한다.
> 다음 data 단계는 scripted action 수집이 아니라 official demonstration HDF5의
> state/action timing을 확인하고 저장된 MuJoCo state를 exact replay하는 Phase 2D다.
> `docs/revised_research_roadmap_v3.md`를 참고한다.

> **2026-08-05 문헌 재평가:** 이 문서는 완료된 Phase 1A state-access gate를 기록한다.
> Action trajectory, episode split, capability-aware relation label은 다시 열린 Phase 1B
> dataset gate에서 추적했으며 이후 Phase 2D로 개정했다.

## 목적

Simulator에서 접근 가능한 LIBERO 값을 기록하고 Phase 2 graph dataset을 만들기 전에
raw-state 조사 계약을 고정한다. 이 단계에서는 graph edge를 만들거나 distance
threshold를 정하거나 GNN을 학습하지 않는다. 해당 결정은 live state capture 이후
Phase 2에서 고정한다.

## Source 조사 결과

Official LIBERO environment wrapper는 `env.sim`으로 simulator를, `env.robots`로 robot
목록을, `get_sim_state()`로 flatten된 MuJoCo state를 제공한다. Task environment는
`obj_of_interest`도 제공한다.

LIBERO task base는 `objects_dict`, `fixtures_dict`, `object_sites_dict`,
`object_states_dict`, `obj_body_id`를 만든다. 일반 object와 fixture의 state wrapper는
`sim.data.body_xpos`, `sim.data.body_xquat`를 읽고 site object는 site position과
rotation을 읽는다. 같은 interface가 contact, containment, on-top, open, close 같은
task predicate를 제공한다.

따라서 inspection script는 다음을 함께 기록한다.

1. Raw observation key와 shape.
2. Privileged simulator object state와 runtime body index.

## Identity 규칙

`logical_id`를 graph identity로 사용한다. 이는 `red_cube`, `target_zone` 같은
task/environment object 이름이다. `body_id`는 현재 MuJoCo runtime의 index일 뿐이다.
Debug에는 쓸 수 있지만 episode 간 object identity로 사용하면 안 된다.

## Live 검증 대상 state 계약

| Field | 내용 | 상태 |
|---|---|---|
| `robot0_eef_pos` | end-effector position | inspection script가 조사 |
| `robot0_eef_quat` | end-effector orientation | inspection script가 조사 |
| `robot0_gripper_qpos` | gripper joint position | inspection script가 조사 |
| `obj_of_interest` | task 관련 logical name | LIBERO task env가 제공 |
| `obj_body_id` | logical name에서 runtime body index로의 mapping | LIBERO task env가 제공 |
| `body_xpos/body_xquat` | MuJoCo object pose | object/fixture body에서 제공 |
| `object_states_dict` | predicate와 joint-state interface | LIBERO task env가 제공 |
| `sim.get_state()` | 전체 privileged simulator state | wrapper가 제공 |

Script는 robot key가 없을 때 임의 fallback state vector를 만들지 않는다. 실제
observation 계약을 runtime에서 확인해야 하므로 missing key는 `null`로 기록한다.

## 좌표와 quaternion 안전 규칙

- 먼저 raw pose를 저장하고 live frame convention 확인 후 normalize한다.
- World, robot base, task-local 중 어느 coordinate frame인지 명시한다.
- Raw MuJoCo body quaternion ordering을 policy-facing convention과 분리해 기록한다.
- Normalization statistics는 training split에서만 계산한다.
- Stage 1 input window에 future state, success, terminal 정보가 들어가지 않게 한다.

## Phase 1A 파일

- `scripts/phase1/state_inspection.py`: live LIBERO task inspector.
- `tests/test_phase1_inspection.py`: LIBERO 비의존 schema 검사.
- `configs/phase1_state_inspection.json`: 기본 inspection 설정.
- `data/phase1_libero_state_sample.json`: schema example.
- `data/phase1_libero_state_capture.json`: Colab live capture의 compact manifest.

## LIBERO 설치 후 실행

```powershell
python -m scripts.phase1.state_inspection `
  --suite libero_spatial `
  --task-id 0 `
  --init-state-id 0 `
  --steps 1 `
  --controller JOINT_POSITION `
  --output data/phase1_libero_state_capture.json
```

이 명령으로 `libero_spatial` task 0, initial state 0을 live capture했다. Snapshot 2개,
snapshot당 object/fixture/site entry 23개, 92-dimensional flattened simulator state,
통제 graph 연구에 필요한 raw robot field가 포함됐다. Compact manifest는 측정 schema와
대표 pose를 기록하며 전체 capture는 명령을 다시 실행해 생성한다.

## 현재 한계

로컬 project runtime에는 LIBERO, robosuite, MuJoCo가 없어 live capture는 Colab에서
실행했다. 당시 환경은 robosuite 1.4.0의 MuJoCo mass-matrix wrapper를 위한 runtime-only
compatibility patch와 신뢰한 official LIBERO init-state file을 읽을 때 명시적인
`weights_only=False`가 필요했다. Baseline model code에는 적용하지 않았으며 향후
environment lock에 기록해야 한다.

Generic predicate probe에서 일부 object-specific `get_joint_state`, `is_open`,
`is_close` 호출은 shape 또는 구현 오류를 반환했다. Phase 2 graph extractor는
capability-aware predicate를 사용하고 unavailable value를 기본값으로 채우지 말아야 한다.

Diagnostic controller 기본값은 `JOINT_POSITION`이다. LIBERO `MountedPanda`에 IK solver를
요구하지 않으며 Stage 2 policy 결정과 무관하다.

## 참고자료

- Official repository: https://github.com/Lifelong-Robot-Learning/LIBERO
- Environment wrapper: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py
- Task state interface: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/bddl_base_domain.py
- Object state predicates: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/object_states/base_object_states.py
