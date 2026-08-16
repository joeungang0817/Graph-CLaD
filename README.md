# Graph-CLaD 연구 작업공간

이 저장소는 CLaD의 semantic transition을 object–relation graph transition으로
구조화할 수 있는지 검증한다. 원 연구자의 CLaD 모듈은 `baseline_code/`에 보존하며,
현재 실험은 LIBERO official demonstration에서 만든 graph dynamics dataset을 사용한다.

현재 결론은 제한적이다. Near-parameter-matched corrected gate에서 late-action G1은
pair MLP를 일관되게 이기지 못해 깊은 GNN 확대를 중단했고, causal pair-local
temporal encoder의 H0–H3 history/action factorial로 전환했다. Three-fold seed-0
screen에서 H3가 primary PR-AUC는 가장 높았지만, action-alignment control과
weak-label QA가 끝나기 전에는 representation 우월성이나 인과적 action 효과를
주장하지 않는다.

## 처음 시작하기

1. [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)에서 연구 상태와 데이터·실험 경로를 읽는다.
2. [notebooks/README.md](notebooks/README.md)의 공식 Phase 0→1A→2A→2R→2D→3A→3B 순서대로 노트북을 연다.
3. `python -m scripts.research_paths`로 로컬/Colab 경로를 확인한다.
4. 가벼운 계약 검사는 `python -m unittest tests.test_research_paths tests.test_notebook_structure`로 실행한다. 전체 test discovery는 PyTorch와 `pytest`를 포함한 실험 의존성이 필요하다.
5. 학습 전에는 config, manifest, output root, code snapshot 경로가 새 버전인지 확인한다.

KCloudVPN Linux에서 실행할 때는 [SSH 실행 안내서](docs/kcloudvpn_linux_ssh_runbook_ko.md)를
먼저 읽는다. 접속 주소는 `ubuntu@172.10.5.118`이며, 서버의 영구 디스크 경로는
`GRAPH_CLAD_ARTIFACT_ROOT` 환경 변수로 지정한다.

핵심 구현의 source of truth는 `scripts/phase*/`다. 루트 `scripts/*.py` 중 일부는
과거 명령을 보존하는 compatibility wrapper이며 새 코드를 추가하지 않는다. Colab
노트북에는 재사용 가능한 구현을 복사하지 않고 저장소 모듈을 import한다.

대용량 데이터·체크포인트·per-sample prediction은 Git에 넣지 않는다. 현재 persistent
artifact root는 다음과 같다.

`/content/drive/MyDrive/Graph-CLaD/artifacts`

세부 실행법, 폴더 역할, legacy 대응표, 검증 범위는
[RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)에 정리되어 있다.
