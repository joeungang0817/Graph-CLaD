# 보관 후보 — 2026-08-13

이번 조사에서는 어떤 파일도 이동하거나 삭제하지 않았다. 아래 목록은 사용자 확인을
받은 뒤 `archive/`로 옮기거나 재생성 가능한 cache로 처리할 수 있는 후보일 뿐이다.

## 재생성 가능한 후보

- `scripts/**/__pycache__/`
- `tests/__pycache__/`

위 항목은 Python bytecode cache다. 연구 근거가 아니며 소스에서 다시 생성할 수 있다.

## Colab 전송 과정에서 생긴 후보

- `.tmp_pair_local_sync_v1.zip`
- `.tmp_pair_local_sync_v2.zip`
- `.tmp_pair_local_sync_v3.zip`
- `.tmp_pair_local_sync_v2_stage/`
- `.tmp_pair_local_sync_v3_stage/`
- `.tmp_phase3_corrected_v2_bundle.zip`
- `.tmp_phase3_holder_action_bundle.zip`
- `.tmp_phase3_holder_action_bundle_v3.zip`

이 자료는 Colab으로 코드를 전달하기 위해 만든 bundle 또는 추출 stage다. 현재
`scripts/phase3/`, `configs/`, `tests/`에 동일하거나 상위 호환인 canonical source가
있지만, 실제 이동 전에는 다시 SHA256과 파일 목록을 비교해야 한다.

## 현재 위치에 유지할 항목

- `scripts/` 루트의 compatibility wrapper: 과거 명령과 snapshot 재현에 필요하다.
- `archive/legacy_staging/`: 역사적 실행 코드의 의도적인 보존본이다.
- `notebooks/graph_clad_phase0_to3.ipynb`: 과거 통합 Colab 실행 기록이다.
- 연구 계획서와 논문 PDF: 설계 근거와 출처이므로 임시 파일로 취급하지 않는다.
- `data/*.json`: 소규모 fixture와 요약은 regression 및 provenance에 필요하다.

승인 후 이동할 경우 `RESEARCH_GUIDE.md`, 관련 문서 링크, `.gitignore`, checksum
기록을 함께 갱신해야 한다.
