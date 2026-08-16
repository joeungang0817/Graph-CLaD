# KCloudVPN Linux SSH 실행 안내서

이 문서는 Colab 대신 KCloudVPN 내부 Linux 서버에서 Graph-CLaD corrected
protocol을 실행하기 위한 재현 절차다. 현재 Colab 런타임과 그 결과 디렉터리는
중단·삭제하지 않으며, KCloudVPN 실행은 별도 output 디렉터리에 저장한다.

## 1. 접속

KCloudVPN에 먼저 연결한 뒤 로컬 터미널에서 다음처럼 접속한다.

```bash
ssh ubuntu@172.10.5.118
```

비밀번호, SSH 키, 포트가 연구실 정책과 다르면 조교님 또는 서버 관리자가 제공한
값을 사용한다. 이 저장소에서는 SSH 접속을 대신 수행하지 않는다.

## 2. 저장소와 가상환경

서버에 저장소가 이미 있으면 해당 경로로 이동하고, 없다면 연구실의 Git 주소를
사용해 clone한다. 아래의 `<REPOSITORY_URL>`은 실제 주소로 바꾼다.

```bash
cd ~
git clone <REPOSITORY_URL> Graph-CLaD
cd ~/Graph-CLaD
git status --short
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

PyTorch는 서버의 CUDA 드라이버와 연구실 설치 정책에 맞는 build를 먼저 설치한다.
CUDA wheel을 임의로 고정하지 않는다. 이후 최소 의존성을 설치한다.

```bash
python -m pip install -r requirements-phase0.txt
```

`requirements-phase0.txt`의 `torch` 항목은 서버에 맞는 PyTorch로 이미 설치된
경우 다시 설치하지 않아도 된다. 설치 후 다음을 확인한다.

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY
nvidia-smi
```

## 3. 환경 변수와 입력 artifact

실행 경로는 config에 절대경로로 박지 않고 다음 환경 변수로 정한다.

```bash
export GRAPH_CLAD_PROJECT_ROOT="$HOME/Graph-CLaD"
export GRAPH_CLAD_ARTIFACT_ROOT="$HOME/graphclad-artifacts"
export GRAPH_CLAD_LIBERO_ROOT="/path/to/LIBERO"
cd "$GRAPH_CLAD_PROJECT_ROOT"
```

`GRAPH_CLAD_ARTIFACT_ROOT`는 서버 재시작 후에도 유지되는 연구실 디스크 경로로
바꾼다. `/tmp`, 컨테이너 임시 디렉터리, 세션 종료 시 사라지는 scratch는 쓰지
않는다. `GRAPH_CLAD_LIBERO_ROOT`는 현재 corrected pair-local gate에서 직접
필수 입력이 아닐 수 있지만, 후속 Phase와 preflight의 일관성을 위해 설정한다.

현재 config가 요구하는 입력은 다음 네 가지다. 대용량 자연 데이터는 Git에 넣지
말고, 연구실 저장소·공유 디스크·승인된 전송 방법 중 하나로 복사한다.

```text
$GRAPH_CLAD_ARTIFACT_ROOT/phase3_holder_action_v1/corrected_protocol_v2/phase3B_R1_eval_manifest_v2.json
$GRAPH_CLAD_ARTIFACT_ROOT/phase2d/data/phase2d_full_demo_v2_inputclean_stream1/
$GRAPH_CLAD_ARTIFACT_ROOT/phase2d/data/phase2d_holding_target_v2_inputclean_stream1/
$GRAPH_CLAD_ARTIFACT_ROOT/phase2d/data/phase2d_demo_split_manifest.json
$GRAPH_CLAD_ARTIFACT_ROOT/phase3_holder_action_v1/corrected_protocol_v2/ai_assisted_full_cluster_audit_v1/review/ai_assisted_sensitivity_groups_v1.json
```

`phase3B_R1_eval_manifest_v2.json`은 Colab에서 그대로 복사하면 source root가
`/content/drive`를 가리키므로 서버 경로로 바꿔야 한다. 처음에는 아래 portable
config로 재생성하는 방법을 준비했지만, 현재 builder는 압축 해제된 graph payload
전체를 메모리에 유지하며 KCloudVPN에서 memory OOM으로 추정되는 `Killed`가 발생했다.

```bash
mkdir -p "$GRAPH_CLAD_ARTIFACT_ROOT/phase3_holder_action_v1/corrected_protocol_v2"
python -m scripts.phase3.build_eval_manifest \
  --config configs/phase3_kcloudvpn_linux_eval_manifest_v2.json \
  --output "$GRAPH_CLAD_ARTIFACT_ROOT/phase3_holder_action_v1/corrected_protocol_v2/phase3B_R1_eval_manifest_v2.json"
```

따라서 streaming builder가 구현되기 전에는 이 명령을 큰 Phase2D 입력에 반복하지
않는다. 현재 권장 방법은 기존 Colab `status=pass` manifest를 복사하고 원본을 별도
보존한 뒤 `/content/drive/MyDrive/Graph-CLaD/artifacts` 문자열만
`$GRAPH_CLAD_ARTIFACT_ROOT`로 바꾸는 것이다. 변환 후 `status=pass`, folds=3,
source roots를 확인한다. 경로 변환으로 SHA256은 바뀌므로 원본과 변환본을 함께 보존한다.
Colab의 `/content`는 임시 공간이며
현재 실행 결과의 원본은 Google Drive에 있으므로, KCloudVPN에서 자동으로 보인다고
가정하지 않는다.

### Drive artifact의 우선순위

| Drive 항목 | 새 학습 실행 | 용도 |
|---|---:|---|
| `phase2d_full_demo_v2_inputclean_stream1/` | 필수 | natural sample과 causal history 재구성 |
| `phase2d_holding_target_v2_inputclean_stream1/` | 필수 | train/challenge용 target-aligned sample |
| `phase2d_demo_split_manifest.json` | 필수 | episode split 고정 및 manifest 재생성 |
| `phase3B_R1_eval_manifest_v2.json` | 필수 | Colab pass manifest의 source root를 서버 경로로 바꾼 portable copy 사용 |
| `ai_assisted_sensitivity_groups_v1.json` | 실행에는 선택 | weak-label sensitivity 사후 분석 |
| 기존 `checkpoints/`, `predictions/`, result JSON | 새 학습에는 불필요 | 기존 결과 비교·보고서·bootstrap 재분석 |
| 기존 `code_snapshot/`, runtime manifest | 새 학습에는 불필요 | 과거 실행의 정확한 재현 및 감사 |

따라서 처음에는 표에서 “필수” 세 항목과 저장소 코드/config만 전송하면 된다.
기존 checkpoint를 새 output에 복사하거나 resume하려고 하지 않는다. 현재 runner는
새 version output을 만드는 gate이며, 과거 checkpoint를 자동으로 이어 학습하는
resume 계약이 아니다.

## 4. 사전 점검

```bash
cd "$GRAPH_CLAD_PROJECT_ROOT"
python -m scripts.research_paths
test -f "$GRAPH_CLAD_ARTIFACT_ROOT/phase3_holder_action_v1/corrected_protocol_v2/phase3B_R1_eval_manifest_v2.json"
test -d "$GRAPH_CLAD_ARTIFACT_ROOT/phase2d/data/phase2d_full_demo_v2_inputclean_stream1"
test -d "$GRAPH_CLAD_ARTIFACT_ROOT/phase2d/data/phase2d_holding_target_v2_inputclean_stream1"
test -f "$GRAPH_CLAD_ARTIFACT_ROOT/phase2d/data/phase2d_demo_split_manifest.json"
```

위 `test` 중 하나라도 실패하면 학습을 시작하지 않는다. config 안의 `${...}`는
runner가 읽을 때 확장되며, 환경 변수가 비어 있으면 경로가 유효하지 않다.
Weak-label sensitivity 파일은 사후 분석 시에만 별도로 확인한다.

## 5. 먼저 실행할 protocol

권장 순서는 다음과 같다.

1. `configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json`로
   action-shuffled 3-fold alignment control을 실행한다.
2. 기존 H0/H1/H2/H3 three-fold seed-0 12-run은 Colab에서 이미 완료됐으므로 단순
   환경 전환을 이유로 다시 실행하지 않는다. Alignment 완료 후 기존 aligned H3
   result/prediction과 paired 비교한다.

KCloudVPN config는 기존 Colab 경로를 사용하지 않고 `GRAPH_CLAD_ARTIFACT_ROOT`
아래의 새 output 디렉터리를 사용한다. 기존 Colab 결과와 섞거나 같은 디렉터리에
재실행하지 않는다.

## 6. 중단에 강한 실행

GPU 작업은 `tmux` 안에서 실행한다.

```bash
tmux new -s graphclad-align
cd "$GRAPH_CLAD_PROJECT_ROOT"
source .venv/bin/activate
export GRAPH_CLAD_PROJECT_ROOT="$HOME/Graph-CLaD"
export GRAPH_CLAD_ARTIFACT_ROOT="$HOME/graphclad-artifacts"
python -u -m scripts.phase3.run_corrected_architecture_gate \
  --config configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json
```

실행 중에는 `Ctrl-b`를 누른 뒤 `d`로 detach한다. 다시 접속할 때는 다음을 쓴다.

```bash
tmux attach -t graphclad-align
```

Factorial screen은 portability 회귀를 별도 검증해야 한다는 연구적 이유가 생긴 경우에만
새 output version으로 실행한다. 기존 12-run을 자동 후속 작업으로 반복하지 않는다.

```bash
tmux new -s graphclad-factorial
cd "$GRAPH_CLAD_PROJECT_ROOT"
source .venv/bin/activate
python -u -m scripts.phase3.run_corrected_architecture_gate \
  --config configs/phase3_kcloudvpn_linux_pair_local_temporal_threefold_seed0_v1.json
```

`nohup`을 쓸 경우에는 표준출력을 별도 로그로 저장하되, runner가 만드는
`runtime_manifest.json`, result JSON, checkpoint, prediction, code snapshot을
삭제하거나 덮어쓰지 않는다.

## 7. 완료 확인

각 실행의 output root 아래에 다음이 있어야 한다.

```text
phase3_*.json
runtime_manifest.json
checkpoints/
predictions/
code_snapshot/
```

Result JSON의 `status`가 `completed`이고 action alignment는 `results` 배열 길이가 3,
factorial은 12인지 확인한다. `runtime_manifest.json`에서 GPU 이름, CUDA 여부,
config/manifest SHA256, causal history QA를 확인한다. `predictions/`는 이후
hierarchical bootstrap과 calibration 분석에 필요한 per-sample artifact다.

## 8. GPU와 protocol 해석

이전 Colab 세션의 T4는 그 세션에서만 사용한 일시적 상태다. KCloudVPN 서버의 실제
GPU는 매번 `nvidia-smi`와 runtime manifest로 기록한다. 새 서버에서 OOM이 발생하면
기존 결과를 수정하지 말고 `..._batch32_v2.json`처럼 새 config version을 만들어
batch size를 32로 낮춘다. GPU가 달라지거나 batch size가 달라진 결과를 기존 A100/64
결과와 같은 protocol 평균에 합치지 않는다.

현재 확인된 KCloudVPN 장치는 NVIDIA GeForce RTX 3090 24GB이며 driver는
`595.71.05`, `nvidia-smi`의 CUDA compatibility 표시는 `13.2`다. 이 값은 현재
runtime 기록이며, 이후 서버 재생성이나 driver 변경 시 다시 측정한다.

또한 한 fold/seed 또는 seed-0 three-fold 결과는 architecture/protocol gate다.
자연 test PR-AUC, release F1, hard-negative FPR, action shuffle 하락을 함께 본 뒤에만
추가 seed 확대를 결정한다.

## 9. 보존 및 전송 원칙

- 현재 Colab process와 Drive output은 이 전환 때문에 중단하지 않는다.
- 대용량 dataset/checkpoint는 Git에 커밋하지 않는다.
- 서버 output root와 입력 artifact의 백업 정책을 연구실 기준에 맞춘다.
- SSH 키·토큰·비밀번호를 config, notebook, 로그에 기록하지 않는다.
- 실행 전 `git status --short`, config 경로, 입력 경로, output root를 기록한다.

관련 설정:

- `configs/phase3_kcloudvpn_linux_pair_local_temporal_action_alignment_seed0_v1.json`
- `configs/phase3_kcloudvpn_linux_pair_local_temporal_threefold_seed0_v1.json`
- `scripts/research_paths.py`
- `scripts/phase3/run_corrected_architecture_gate.py`
