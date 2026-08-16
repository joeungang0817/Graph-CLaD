# Archive 안내

이 폴더는 현재 실행에서 import하거나 학습 입력으로 사용하지 않는 과거 bundle,
staging copy, 생성 cache를 보존한다. 재현 근거일 수 있으므로 임의로 삭제하지 않는다.

## 분류

- `legacy_bundles/`: Phase 0–3의 과거 전달 zip.
- `legacy_staging/`: 과거 bundle의 추출본과 PDF 검사 산출물.
- `generated_python_cache/`: 이전 정리에서 회수한 Python cache.
- `pre_korean_translation_20260813.zip`: 문서 한국어화 전 Markdown 원본.
- `temporary_transfers_20260816/`: 루트에서 정리한 Colab 동기화용 임시 bundle과 stage.

`temporary_transfers_20260816/`의 파일은 활성 `scripts/phase*/`보다 이전 snapshot이다.
루트 가독성을 위해 2026-08-16에 삭제 없이 이동했다. 현재 source of truth는 항상
`../scripts/phase*/`, 실행 config는 `../configs/`다.

| 이동된 항목 | 크기/형태 |
|---|---|
| `.tmp_pair_local_sync_v1.zip` | 37,380 bytes |
| `.tmp_pair_local_sync_v2.zip` | 45,857 bytes |
| `.tmp_pair_local_sync_v3.zip` | 52,011 bytes |
| `.tmp_phase3_corrected_v2_bundle.zip` | 62,849 bytes |
| `.tmp_phase3_holder_action_bundle.zip` | 11,527 bytes |
| `.tmp_phase3_holder_action_bundle_v3.zip` | 11,821 bytes |
| `.tmp_pair_local_sync_v2_stage/` | extracted staging directory |
| `.tmp_pair_local_sync_v3_stage/` | extracted staging directory |
