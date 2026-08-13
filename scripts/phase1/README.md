# Phase 1

- `state_inspection.py`: LIBERO observation, robot/object state, logical ID와 simulator
  metadata를 조사한다.
- `runtime_compat.py`: Colab의 robosuite 1.4.1과 MuJoCo 3 mass-matrix API 차이를
  점검한다. 기본은 dry-run이며 `--apply` 때만 백업 후 수정한다.

