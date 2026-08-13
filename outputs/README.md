# 로컬 출력 폴더

이 폴더는 가벼운 로컬 실행의 기본 산출물 루트다. `scripts.research_paths`는 사용자가
명시적으로 요청했을 때만 `checkpoints/`, `logs/`, `figures/`, `metrics/`,
`predictions/` 하위 폴더를 만든다. 각 하위 폴더의 내용은 Git 추적 대상에서 제외한다.

official-demo 전체 데이터셋, 체크포인트, 표본별 예측, 전체 실험 결과는 영구 저장소에
보존한다. 현재 Colab 영구 저장 경로는 다음과 같다.

`/content/drive/MyDrive/Graph-CLaD/artifacts`

Drive의 대용량 artifact를 이 저장소로 복사하지 않는다. 대신 config, manifest,
result 경로, checksum, code snapshot을 기록한다.
