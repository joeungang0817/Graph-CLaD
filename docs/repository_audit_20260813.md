# 저장소 및 Colab 현황 조사 — 2026-08-13

## 안전 경계

실행 중이던 Colab PID 12616과 해당 output root는 읽기 전용으로 취급했다. 프로세스
signal, 재시작, checkpoint 수정, 출력 삭제, 실행 코드 교체를 하지 않았다. 대용량
dataset과 model artifact도 로컬로 복사하지 않았다.

## 로컬 현황

- Canonical code: `scripts/phase0`, `phase1`, `phase2a`, `phase2r`, `phase2d`, `phase3`.
- 제공 baseline: `baseline_code/`의 세 모듈.
- 실험 설정: legacy와 corrected 버전을 포함한 `configs/*.json`.
- 검사 코드: pure Python, synthetic graph, tensor/model, manifest, analysis,
  weak-label audit 계약을 담은 `tests/`.
- 정리 전 notebook: 저장 출력이 없는 19-cell legacy notebook
  `notebooks/graph_clad_phase0_to3.ipynb` 한 개. 과거 Phase 3 smoke/controlled-run
  경로와 여러 `/content` 절대 경로를 사용했다.
- 소규모 저장 데이터: schema fixture와 compact summary만 존재한다.
- 대형 로컬 참고자료: PDF 두 개. 로컬 training checkpoint나 official-demo 전체
  dataset은 발견되지 않았다.
- 생성 cache: `scripts/**/__pycache__`, `tests/__pycache__`.
- 임시 전송 자료: `.tmp_pair_local_sync_*`, corrected bundle, 추출 stage 폴더.

## 중복 파일 분류

1. `scripts/phase3_offline_probe.py` 같은 `scripts/*.py` 파일은 경쟁 구현이 아니라
   의도적인 compatibility wrapper다.
2. Source of truth는 `scripts/phase*/`다.
3. `.tmp_pair_local_sync_v2_stage`, `_v3_stage`는 Colab으로 전달한 pair-local source의
   중복본이다. 삭제하지 않고 보관 후보로만 분류했다.
4. `archive/legacy_staging`은 역사적 실행본을 정확히 보존한다. 현재 import는 이
   경로에서 이루어지면 안 된다.

## Colab과 로컬 비교

실행 중이던 `/content/Graph-CLaD-corrected-v2`와 완료된 three-fold
`code_snapshot`을 파일명, 크기, SHA256으로만 조사했다.

| Source | SHA256 | 로컬 상태 |
|---|---|---|
| `build_eval_manifest.py` | `3e5c26ad156c...` | 정확히 일치 |
| `offline_probe.py` | `4c3bde2edb87...` | 정확히 일치 |
| `pair_local_temporal.py` | `642a1787e1ce...` | 정확히 일치 |
| `run_holder_action_smoke.py` | `a0e6d0d31a19...` | 정확히 일치 |
| `run_topology_action_followup.py` | `6f5fadefa6b5...` | 정확히 일치 |
| 완료 snapshot runner | `64d089947a5d...` | 로컬이 문서화된 상위 호환본 |
| live stage runner | `973878c21c08...` | alignment protocol만 추가한 실행본 |
| live alignment config | `f706b44c8a75...` | 의미와 byte content 일치; 로컬은 마지막 newline만 추가 |

로컬 runner에는 three-fold 단일 seed screen을 one-fold smoke로 잘못 표시하지 않도록
claim-limit 문구를 바로잡은 변경도 있다. 이 변경은 model training이나 evaluation
계산을 바꾸지 않는다.

Notebook에만 있던 launch/recovery/status cell은 그대로 승격하지 않았다. 재사용할
내용은 `scripts/research_paths.py`, versioned alignment config, 공식 phase별 notebook에
반영했다. 일회성 base64 transfer 조각과 process status cell은 source가 아니라 운영
기록으로 남겼다.

## 경로 조사

과거 Phase 2/3 summary와 config의 `/content`, Drive 절대 경로는 실행 근거이자 고정된
계약이므로 수정하지 않았다. 새 notebook은 `scripts.research_paths`와 환경 변수를 쓴다.

`scripts/phase2d/build_demo_dataset.py`는 CLI 기본값으로 `/content/LIBERO`를 유지한다.
로컬 실행에서는 `--bddl-root`를 전달하거나 `GRAPH_CLAD_LIBERO_ROOT`를 설정한다.

로컬 checkpoint는 발견되지 않았다. 전체 산출물의 canonical 위치는 다음과 같다.

`/content/drive/MyDrive/Graph-CLaD/artifacts`

## 의존성 조사

- Phase 0/model: PyTorch, einops, timm.
- Phase 2 real replay: LIBERO, robosuite, MuJoCo, h5py, numpy.
- Phase 3 training: PyTorch, numpy.
- Audit/plot: viewer 또는 plot 경로에서 matplotlib 사용.

현재는 Phase 0 의존성만 고정되어 있다. Simulator runtime 설치는 환경별로 다르므로
로컬에서 검증하지 않은 부분을 검증된 것처럼 기록하지 않았다.

## 구조 정리 결정

- `src/` migration을 하지 않음: 기존 phase module과 compatibility wrapper가 과거
  snapshot에서 안정적으로 참조된다.
- 기존 notebook을 rename하지 않음: 당시 결정 시점에는 ordered notebook을 추가하고
  legacy runbook을 보존했다. 이후 공식 계획서와 번호 충돌을 확인해 새 runbook만
  공식 Phase 0/1A/2A/2R/2D/3A/3B 명칭으로 교정했다.
- Artifact를 복사하지 않음: persistent root와 hash만 기록한다.
- 임시 파일을 삭제하지 않음: 별도의 보관 후보로만 분류한다.

## 검증 결과

- Python syntax: `baseline_code/`, `scripts/`, `tests/`가 `compileall`을 통과했다.
- JSON: 확인한 모든 config, compact data artifact, 문서 JSON, notebook이 정상 파싱됐다.
- Notebook 계약: 공식 runbook은 nbformat 4이고 목적/입력/설정/실행/출력/검증/다음
  단계 구성을 가지며 저장된 실행 출력이 없다. 모든 code cell은 Python으로 파싱된다.
- 저비용 집중 검사: 환경 독립 경로, notebook 구조, corrected-protocol pure function,
  corrected analysis, weak-label audit 계약 검사 10개가 통과했다.
- 경로 사전 점검: project와 local output root가 정상 해석됐다. 로컬 LIBERO root는
  없으므로 simulator replay는 미검증이다.
- 새 진입 문서의 Markdown link가 모두 로컬에서 유효했다.
- 활성 Python/notebook/JSON/Markdown에서 API key, access token, password, private key,
  client secret 패턴이 발견되지 않았다.
- 전체 test discovery를 통과했다고 기록하지 않는다. 이 workstation에는 Phase 0
  smoke에 필요한 PyTorch와 tensor/model 계약 하나에 필요한 `pytest`가 없다.

이번 검증에서는 training, dataset 재구축, simulator replay, checkpoint load를
실행하지 않았다.
