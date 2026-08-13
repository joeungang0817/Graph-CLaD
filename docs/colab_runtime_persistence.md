# Colab runtime 영구 저장 정책

## 영구 저장 root

모든 Phase 2D source input과 재현성 metadata는 임시 Colab runtime 밖의 다음 경로에
저장한다.

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase2d`

대응하는 local project source는 이 저장소의 `docs/`, `scripts/`, `configs/`에 둔다.

## 보존 대상

- `libero_spatial_hdf5/`: task 0/1/2 official HDF5 input.
- `data/phase2d_demo_split_manifest.json`: 고정 150-demo split manifest.
- `project/`: 복구 가능한 Phase 2D script와 config.
- `artifact_manifest.json`: 저장 input의 size와 SHA256.
- `runtime/`: restore instruction과 status record.

생성한 graph JSONL과 relation-audit output은 성공한 run 직후 같은 Drive root에 복사한다.
산출물이 `/content`에만 존재하게 두지 않는다.

## Runtime reset 후 복구 순서

1. Google Drive를 mount한다.
2. `artifact_manifest.json`을 읽고 HDF5 checksum을 검증한다.
3. `project/`, `data/`, `libero_spatial_hdf5/`를 새 runtime workspace로 copy/link한다.
4. 고정한 runtime dependency를 설치/import하고 짧은 preflight를 실행한다.
5. Persistent input에서 generation 또는 audit을 재개한다.
6. Task 완료 때마다 compressed dataset과 QA JSON을 Drive로 복사하고 completion marker와
   checksum을 쓴다.

Notebook 자체도 Google Drive에 있지만 notebook history를 dataset backup으로 간주하지
않는다. Drive artifact bundle이 rerun의 authoritative source다.

## 복구 코드

Colab backup/restore cell의 재사용 부분은 `scripts/phase2d/persistence.py`에 있다. SHA256
manifest를 만들고 restore 전에 검사하며 관련 없는 runtime state를 삭제하지 않고
overlay한다. Robosuite/MuJoCo preflight patch는 `scripts/phase1/runtime_compat.py`로
분리했으며 기본값은 dry-run이다.

## Clean-runtime compatibility 기록 — 2026-08-06

작동한 Colab preflight는 `robosuite==1.4.1`, MuJoCo 3.11.0을 사용했다. LIBERO import
path에는 구 robosuite package가 필요했고 작은 runtime-only patch로 mass-matrix call을
MuJoCo 3 API에 맞췄다. Runtime reset마다 다시 적용해야 하며 research model이나 source
HDF5 수정이 아니다.

Reset 순서는 Drive mount → persistent bundle restore → pinned robosuite 설치 →
compatibility patch → LIBERO import → one-demo state-replay/graph preflight다. Full graph
file은 Drive에 복사하고 manifest/checksum marker를 쓴 뒤에만 durable로 간주한다.

## 현재 durable Phase 2D release

Corrected full-demo release:

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase2d/data/phase2d_full_demo_v2_splitfixed_stream1`

Manifest:

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase2d/phase2d_full_demo_splitfixed_stream1_manifest.json`

Task 0/1/2 official demonstration 150개를 포함한다. Final QA에서 null split 0,
split mismatch 0, exact state replay, task error 0을 확인했다. 이전
`phase2d_full_demo_v2`는 split repair 전 intermediate artifact이므로 최종 training
input으로 사용하지 않는다.

## Phase 3 controlled-run bundle

경로: `/content/drive/MyDrive/Graph-CLaD/artifacts/phase3`

- `phase3_controlled_taskfamily_report.json`: 3 folds, 45 model/seed runs, control, relation metric.
- `phase3_controlled_taskfamily_config.json`: fold와 training config.
- `phase3_controlled_taskfamily_summary.json`: completion summary.
- `phase3_controlled_taskfamily_checkpoint.json`: latest fold checkpoint.
- `code/offline_probe.py`, `code/run_controlled_taskfamily.py`: 당시 실행 source snapshot.

Runtime reset 후 Drive를 mount하고 snapshot 또는 현재 repository의 versioned runner를
사용한다. Runner는 각 fold 뒤 checkpoint를 쓰며 이전 notebook variable에 의존하지 않는다.

## Holding target artifact

- Target-aligned dataset:
  `/content/drive/MyDrive/Graph-CLaD/artifacts/phase2d/data/phase2d_holding_target_v2_inputclean_stream1`
- Episode-cap result: `/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holding_target_v2`
- Category-aware v3 result:
  `/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holding_target_balanced_v3`

현재 source of truth는 과거 Drive result 아래의 patched code가 아니라 local repository의
`scripts/phase2d/`, `scripts/phase3/`다. 기존 balanced-v3 sampler는
`configs/phase3_holding_target_balanced_v3.json`으로 재현한다. Corrected sampler config
`configs/phase3_holding_target_balanced_v4_samplingfix.json`의 output을 v3 root에 덮어쓰거나
병합하지 않는다.

## 현재 corrected Phase 3 root

`/content/drive/MyDrive/Graph-CLaD/artifacts/phase3_holder_action_v1/corrected_protocol_v2`

새 결과마다 versioned config, code snapshot, manifest, checkpoint, per-sample prediction,
aggregate analysis를 같은 실험 root 아래 보존한다. 실행 전 GPU, Drive mount, source hash,
manifest path, 새 output path를 확인한다.
