# Colab MCP 다음 세션 연결 메모

> 상태: legacy. 현재 학습 환경은 KCloudVPN Linux로 이전했다. 새 세션은
> `docs/NEXT_SESSION_PROMPT.md`와 `docs/CURRENT_STATUS.md`를 사용한다. 이 파일은 기존
> Colab/Drive artifact에 접근해야 할 때만 참고한다.

이 문서는 Codex가 브라우저에 열린 Google Colab notebook의 런타임을 제어하고,
Google Drive의 연구 artifact를 읽고 쓰기 위한 다음 세션용 연결 기록이다.

## 현재 연결 방식

- 로컬 PC의 GPU를 사용하는 것이 아니다.
- 사용자가 브라우저에서 Colab notebook을 열어 둔다.
- Codex의 `colab-mcp` 서버가 열린 Colab 탭과 런타임을 연결한다.
- Colab 런타임의 GPU와 실행 상태를 확인한 뒤 notebook 셀을 실행한다.
- Drive persistent root는 다음과 같다.

```text
/content/drive/MyDrive/Graph-CLaD
```

새 런타임에서는 필요할 때 다음 셀을 실행하고, 사용자가 Google Drive 권한을 승인한다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## MCP 설정

공식 서버 저장소:

```text
https://github.com/googlecolab/colab-mcp
```

프로젝트 설정 파일:

```text
C:\Users\User\Graph-CLaD\.codex\config.toml
```

현재 프로젝트 설정은 다음 형태를 사용한다.

```toml
[mcp_servers.colab]
command = '<현재 uvx.exe의 절대 경로>'
args = ['git+https://github.com/googlecolab/colab-mcp']
startup_timeout_sec = 300
tool_timeout_sec = 300
```

`uvx.exe`는 Codex runtime 교체에 따라 경로가 바뀔 수 있다. 서버 시작 오류가
`지정된 파일을 찾을 수 없음`이면 다음 위치에서 실제 파일을 다시 찾는다.

```text
C:\Users\User\.cache\codex-runtimes\*\dependencies\python\Scripts\uvx.exe
```

찾은 경로를 프로젝트 `.codex/config.toml`의 `command`에 반영하고 Codex를
완전히 재시작한다. MCP 설정은 현재 Codex 프로세스가 시작될 때 읽히므로,
설정 파일만 수정하고 재시작하지 않으면 이전 경로가 계속 사용될 수 있다.

공식 서버는 `pip install colab-mcp` 방식보다 다음 `uvx` 실행 방식을 기준으로 한다.

```text
uvx git+https://github.com/googlecolab/colab-mcp
```

## 다음 세션 시작 순서

1. `C:\Users\User\Graph-CLaD` workspace를 연다.
2. 이 문서와 `docs/colab_runtime_persistence.md`를 읽는다.
3. Colab notebook을 브라우저에서 열고 runtime이 연결되어 있는지 확인한다.
4. Codex를 재시작해 MCP 설정을 다시 로드한다.
5. Colab MCP 연결 후 runtime 상태와 GPU를 읽기 전용으로 확인한다.
6. Drive를 mount하고 `/content/drive/MyDrive/Graph-CLaD`의 manifest/checksum을 확인한다.
7. 필요한 경우에만 persistent artifact를 runtime workspace로 복원한다.

## 현재 연구에서의 주의사항

- Phase 3 controlled experiment 45 runs는 이미 완료되었다.
- `phase3_runtime_manifest.json`과 기존 분석 결과를 먼저 확인한다.
- 현재 다음 목표는 official demo 기반 `holding` positive와 hard-negative event
  dataset 보강이다.
- `holding`은 primary relation이다.
- 현재 task 0·1·2 input-clean demo dataset에는 holding positive가 없다고 기록되어 있다.
- `inside`는 valid label support가 생길 때까지 deferred로 유지한다.

## 주요 persistent 경로

```text
/content/drive/MyDrive/Graph-CLaD/artifacts/phase2d
/content/drive/MyDrive/Graph-CLaD/artifacts/phase3
```

Phase 3 결과를 확인할 때 우선 볼 파일:

```text
/content/drive/MyDrive/Graph-CLaD/artifacts/phase3/phase3_runtime_manifest.json
/content/drive/MyDrive/Graph-CLaD/artifacts/phase3/phase3_controlled_taskfamily_report.json
/content/drive/MyDrive/Graph-CLaD/artifacts/phase3/phase3_controlled_taskfamily_analysis.json
/content/drive/MyDrive/Graph-CLaD/artifacts/phase3/phase3_controlled_taskfamily_summary.json
```
