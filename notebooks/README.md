# 공식 연구 단계별 노트북

이 폴더의 번호는 단순 실행 순번이 아니라
`docs/01-plan/features/graph-clad-integrated-research-v4.plan.md`의 공식 연구 단계를
따른다. `docs/revised_research_roadmap_v3.md`는 v4 이전 단계 개정의 근거로 보존한다.
재사용 구현은 `scripts/phase*/`에 두고 노트북은 경로 확인, 모듈 호출, 산출물 검사만
담당한다.

## 권장 실행 순서

| 순서 | 공식 단계 | 노트북 | 역할 |
|---:|---|---|---|
| 0 | 환경 준비 | `00_environment_and_paths.ipynb` | 로컬/Colab 경로와 GPU 사전 점검 |
| 1 | Phase 0 | `phase_0_clad_baseline_smoke.ipynb` | 제공된 CLaD baseline 실행 계약 |
| 2 | Phase 1A | `phase_1a_state_api_audit.ipynb` | LIBERO state/API 조사 |
| 3 | Phase 2A | `phase_2a_static_graph_contract.ipynb` | 정적 GraphSpec과 extractor 계약 |
| 4 | Phase 2R | `phase_2r_scripted_diagnostics.ipynb` | scripted diagnostic 전용; 주 학습 제외 |
| 5 | Phase 2D | `phase_2d_official_demo_dataset.ipynb` | official-demo temporal graph dataset |
| 6 | Phase 3A | `phase_3a_dataset_and_label_qa.ipynb` | corrected manifest, leakage, label QA |
| 7 | Phase 3B | `phase_3b_corrected_architecture_gate.ipynb` | G1 및 pair-local architecture gate 학습 |
| 8 | Phase 3B | `phase_3b_evaluation_and_controls.ipynb` | 통계, action control, weak-label 판정 |

현재 진행 단계와 서버 output은 `../docs/CURRENT_STATUS.md`를 먼저 확인한다. 공식
Phase 3B action-alignment gate는 실패로 완료됐다. Phase 3C는 아직 실행
노트북으로 만들지 않았고, Phase 4 Stage 1 및 Phase 5~8도 gate 통과 전까지 차단한다.

장시간 또는 상태를 변경하는 셀은 기본적으로 `RUN_* = False`다. 하나의 플래그를
활성화하기 전에 config와 output 경로가 새 버전인지 확인한다. 기존 결과 디렉터리에
새 실행을 덮어쓰지 않는다.

## 기존 통합 노트북 대응표

`graph_clad_phase0_to3.ipynb`는 과거 통합 Colab 실행 기록으로 그대로 보존한다.
기존 파일을 현재 실험 실행에 사용하지 않는다.

| 기존 통합 노트북 내용 | 현재 권장 노트북 |
|---|---|
| 환경, clone, 경로 설정 | `00_environment_and_paths.ipynb` |
| baseline smoke | `phase_0_clad_baseline_smoke.ipynb` |
| state 조사 | `phase_1a_state_api_audit.ipynb` |
| snapshot graph 구성 | `phase_2a_static_graph_contract.ipynb` |
| scripted probe | `phase_2r_scripted_diagnostics.ipynb` |
| official-demo replay | `phase_2d_official_demo_dataset.ipynb` |
| balanced holding manifest | `phase_3a_dataset_and_label_qa.ipynb`의 corrected v2 protocol |
| GNN smoke와 controlled training | `phase_3b_corrected_architecture_gate.ipynb` |
| metric 출력과 결과 검사 | `phase_3b_evaluation_and_controls.ipynb` |

앞서 임시로 만들었던 순번형 `phase_01`~`phase_05` 명칭은 공식 연구 Phase와 충돌해
위 이름으로 교정했다. 이 변경은 노트북 파일에만 적용되며 실험 config, 결과 경로,
checkpoint, code snapshot에는 영향을 주지 않는다.
