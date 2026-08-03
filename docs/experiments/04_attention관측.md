# 2단계 — attention 관측 (NIAR)

**질문:** 1단계에서 확인한 행동(위반 코드가 준수율을 낮춤)이 일어날 때,
모델이 실제로 **위반 코드 구간을 더 참조(attention 배분)** 하는가?
관련 계획서 절: 3.4(꺼내기), 3.7(지표).

> **이건 상관 관측이지 인과 주장이 아니다.** "attention을 많이 줬다"가 "그게 원인이다"는 아니다
> (Jain & Wallace 2019 등). 인과 주장은 전적으로 3단계 개입에 근거한다.
> **오픈 모델(Qwen)에서만 가능** — 가중치·내부 접근이 필요.

---

## 방법론

### attention을 어떻게 꺼내나 (3.4)

전체 attention matrix 저장은 불가(16K에서 층당 ~15GB). 필요한 건 prefill matrix가 아니라
**"새로 생성되는 토큰이 context의 어느 구간을 참조하나"** = decode 단계 query 한 행뿐(층당 ~0.9MB).

- `AttentionInterface`에 **query 길이 분기** 함수 등록:
  - `q_len > 1`(prefill) → SDPA/FlashAttention에 위임, attention 미계산.
  - `q_len == 1`(decode) → 명시적 softmax → **구간별 합으로 즉시 축약, 축약값만 저장.**
- prefill 메모리 폭증 회피, `output_attentions=True` 불필요(FlashAttention 비호환 회피).
- **검증:** 512 토큰 짧은 시퀀스에서 전 구간 eager 결과와 수치 일치 확인(prefill/decode 커널이 달라 필수).

### 지표 — 길이 정규화 (3.7)

단순 attention 합 비율은 context가 길어지면 지침 구간이 **참조 정도와 무관하게 자동 감소**한다.
그래서 길이로 정규화:

```
NIAR = (지침 구간 attention 합 ÷ 지침 토큰 수) ÷ (전체 attention 합 ÷ 전체 토큰 수)
  NIAR = 1 → 평균 수준 참조,  > 1 → 평균보다 많이 참조
```

- 동일 방식으로 **NSCAR**(선행 코드), **NVCAR**(위반 코드) 정의.
- **핵심 성질:** 균등 분포일 때 NIAR은 context 길이와 무관하게 1 → "지침이 짧아서"라는 반론 차단.
- **보조:** 지침과 같은 토큰 수의 코드 구간 100회 무작위 추출해 attention 합 직접 비교(정규화 불필요).
- **보완 지표:** `‖α·v‖`(Kobayashi 2020) — attention이 높아도 value가 작으면 실제 기여 작음.

### 교란 통제 (3.7)

| 요인 | 처리 |
|---|---|
| attention sink (Xiao 2024) | 앞 4개 토큰 제외. 제외 여부 민감도는 부록 |
| 위치 편향 (Liu 2024) | 지침을 전 조건 동일 위치 고정, 조건 간 prefix 길이를 토큰 수준 정렬 |

### 예상 관측

위반 코드가 준수율을 낮추는 조건에서 **NVCAR(위반 코드 참조)이 상대적으로 높고 NIAR(지침 참조)이 낮으면**,
"규칙보다 선행 코드를 더 참조한다"는 행동과 정합하는 관측이 된다. (여기까지는 상관.)

---

## 실행

- 하네스: [`src/stage2_attention.py`](../../src/stage2_attention.py)
  - `--validate` — 512토큰 eager 대조 검증(계획서 3.4). **관측의 게이트.**
  - `--observe` — exp1 조건(compliant_remaining)·seed를 그대로 재사용해 NIAR/NSCAR/NVCAR 관측.
- Colab 진입점: [`notebooks/stage2_colab.ipynb`](../../notebooks/stage2_colab.ipynb) (검증 → 관측 → 집계).
- 관측 결과: `results/stage2_niar.jsonl` (세션 단위 append, `(model, condition, seed)`로 재개).
- 구현 메모:
  - `AttentionInterface`에 query 길이 분기 함수 등록 — prefill은 SDPA 위임, decode만 명시적 softmax → 구간별 즉시 축약. GQA `repeat_kv` 처리.
  - 관측 context는 exp1_main의 `build_prefix`/`system_prompt`를 그대로 import(정본과 assert로 대조) → 행동↔관측 조건 동일성 보장.
  - attention sink 앞 `SINK=4` 토큰은 구간·분모 양쪽에서 제외.

## 결과

*(미실행 — Colab에서 `--validate` PASS 후 `--observe` 실행하고 채운다.)*

- NIAR / NSCAR / NVCAR 조건별 분포:
- 동일 길이 구간 대조:
- `‖α·v‖` 보완 지표:
- attention sink 민감도:

---

## 관찰 기록 (2단계)

*(실행 시 날짜별로 추가)*
