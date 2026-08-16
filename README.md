# Graph-CLaD 연구 작업공간

이 저장소는 CLaD의 semantic transition을 object–relation graph transition으로
구조화할 수 있는지 검증한다. 원 연구자의 CLaD 모듈은 `baseline_code/`에 보존하며,
현재 실험은 LIBERO official demonstration에서 만든 graph dynamics dataset을 사용한다.

현재 결론은 제한적이다. Near-parameter-matched corrected gate에서 late-action G1은
pair MLP를 일관되게 이기지 못해 깊은 GNN 확대를 중단했고, causal pair-local
temporal encoder의 H0–H3 history/action factorial로 전환했다. Three-fold seed-0
screen에서 H3가 primary PR-AUC는 가장 높았지만, action-alignment gate에서 aligned
PR-AUC 우세가 1/3 tasks에 그쳐 실패했다. H3의 인과적 action 효과와 seed 확대를
중단하고 Phase 3C CLaD-causal representation 비교로 이동한다. Core 비교는 controlled
semantic CLaD, SceneSet, pair-local, GeomMPNN, RelPool, RelMPNN이며 primary는 RelMPNN 대
semantic의 sample-level valid spatial-relation any-change PR-AUC다. RelPool은 동일 relation
token에서 message passing만 제거한 필수 control이다. Holding은 audit 전 Phase 3C
checkpoint/gate에서 제외하고, Transformer 2×2는 core 결과 후 secondary로 실행한다.

## 처음 시작하기

1. [현재 상태와 실행 인계](docs/CURRENT_STATUS.md)에서 지금 실행할 작업을 확인한다.
2. [통합 연구계획서 v4](docs/01-plan/features/graph-clad-integrated-research-v4.plan.md)에서
   연구 질문, 비교 조건, gate와 최종 완료 기준을 확인한다.
3. 다음 구현은 [Phase 3C 계획](docs/01-plan/features/phase3c-oracle-graph-clad-core.plan.md)과
   [기술 설계](docs/02-design/features/phase3c-oracle-graph-clad-core.design.md)를 따른다.
4. 처음 접한다면 [연구 입문서](docs/RESEARCH_WORKFLOW_FOR_BEGINNERS.md)와
   [코드 입문서](docs/CODEBASE_GUIDE_FOR_BEGINNERS.md)를 순서대로 읽는다.
5. [문서 색인](docs/README.md)에서 목적에 맞는 설계·결과 문서를 찾는다.
6. [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)에서 전체 데이터·코드·실행 구조를 읽는다.
7. [notebooks/README.md](notebooks/README.md)의 공식 Phase 순서대로 노트북을 연다.
8. `python -m scripts.research_paths`로 환경 경로를 확인하고, 학습 전에는 config,
   manifest, output root, code snapshot이 새 version인지 확인한다.

현재 학습 기본 환경은 KCloudVPN Linux의 RTX 3090 서버다. 실행할 때는
[SSH 실행 안내서](docs/kcloudvpn_linux_ssh_runbook_ko.md)를 따른다. 서버 artifact
root는 `/home/ubuntu/graphclad-artifacts`이며 환경 변수
`GRAPH_CLAD_ARTIFACT_ROOT`로 지정한다.

핵심 구현의 source of truth는 `scripts/phase*/`다. 루트 `scripts/*.py` 중 일부는
과거 명령을 보존하는 compatibility wrapper이며 새 코드를 추가하지 않는다. Colab
노트북에는 재사용 가능한 구현을 복사하지 않고 저장소 모듈을 import한다.

대용량 데이터·체크포인트·per-sample prediction은 Git에 넣지 않는다. 현재 새 실행은
KCloudVPN의 `/home/ubuntu/graphclad-artifacts`, 기존 Colab 결과는
`/content/drive/MyDrive/Graph-CLaD/artifacts`에 version별로 보존한다.

세부 실행법, 폴더 역할, legacy 대응표, 검증 범위는
[RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)에 정리되어 있다.
