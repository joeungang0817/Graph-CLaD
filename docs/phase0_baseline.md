# Phase 0: CLaD baseline smoke test

## 목적

LIBERO나 graph data를 도입하기 전에 제공된 Stage 1 code가 synthetic train/eval
경로를 실행할 수 있는지 확인한다.

## 통과 기준

- 제공된 baseline source file을 수정하지 않는다.
- 입력 dimension 계약을 명시한다.
- Training forward가 예상한 scalar loss 네 개를 반환한다.
- Loss와 output이 모두 finite다.
- Backward에서 finite gradient가 생성된다.
- Evaluation이 `[B, H]` 형태의 foresight embedding 두 개를 반환한다.
- EMA 초기화가 online parameter를 복사하고 다음 update가 설정한 momentum 규칙을 따른다.
- 아직 검증하지 못한 data-pipeline 사항을 `docs/unknowns.md`에 기록한다.

## 실행

저장소 root에서 다음을 실행한다.

```powershell
python tests/test_phase0_smoke.py
```

검사는 CPU에서 실행 가능한 작은 설정을 사용한다. Production dimension 가정은
`configs/phase0_synthetic.json`에 기록되어 있으며 Stage 1 실제 학습 전에 real VLM과
dataset pipeline에 맞는지 확인해야 한다.

## 의존성

PyTorch, `einops`, `timm`이 필요하다. Phase 0 조사 당시 Codex bundled Python에는
PyTorch가 없어서 project runtime이 제공되기 전에는 그 환경에서 실행할 수 없었다.
