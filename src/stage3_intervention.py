"""
3단계 — attention/KV 개입 (P1a·P2). 계획서 3.5(점수화) · 3.6(개입).

주 가설 H3: attention/KV 개입이 준수 선호 점수를 **인과적으로** 바꾼다.
2단계는 "attention을 많이 준다"는 상관까지만 본다. 3단계는 그 attention/KV를
**직접 치환·조정**해 준수 결정이 바뀌는지 봄으로써 인과 기여를 확립한다.
→ 논문의 진짜 새로움. 오픈 모델(Qwen)에서만 가능 — 내부 접근 필요.

CLAUDE.md "절대 어기면 안 되는 것"을 코드로 강제한다:
  (1) 준수 선호 점수 값으로 표본을 선별하지 않는다.
      context pair 선별은 토큰 수 일치·함수명 속성 등 점수와 무관한 사전 기준만.
      A/B 분할은 고정 seed 셔플로, 어떤 점수도 참조하지 않는다.
  (2) 후보 쌍은 모든 조건에서 **동일한 문자열**을 채점한다(DECISION_PAIR 고정).
      실제 생성 이름과 절대 섞지 않는다(그건 행동 판정용, 여기선 안 씀).
  (3) 개입 pair는 토큰 수 정확 일치. ±3 허용 없음.
      두 조건 프롬프트를 토큰 정렬 assert로 검증, 이름 토큰 밖에서 다르면 폐기.
  (4) **P1a는 KV group 단위(=KV head 4개), P2는 query head 단위(28개).**
      GQA(7:1 공유)라 P1a를 query head로 짜면 7개를 건드리며 1개로 보고하게 된다.
      → P1a는 pre-repeat key/value(b, kvh, kv, d)의 kvh 축을 인덱싱(=KV group).
        P2는 post-softmax α(b, H, q, kv)의 H 축을 인덱싱(=query head).
  (5) results/ 사전 등록 파일 수정 금지 → 새 파일(stage3_intervention.jsonl).
  (6) 주 가설은 H1a·H2a·H3 뿐. P1b/P3는 탐색적(이 v1 미구현, 아래 주석 참조).

개입 메커니즘 (계획서 3.6)
  - 조건 간 실제로 다른 건 ① 선행 코드 구간 K/V, ② 결정 지점 query 둘뿐
    (지침 K/V는 causal mask 때문에 두 조건에서 비트 동일 → 교체 무의미).
  - **P1a(주):** 위반 조건의 선행 코드 K/V를 준수 조건 값으로 치환 → α 재계산.
    prefix KV 캐시는 위반본 하나만 만들고, 후보 채점 forward에서 코드 구간 열을
    준수본 donor로 덮어쓴다(= 모든 teacher-forcing step에 동일 적용).
  - **P2(보조):** 위반 조건 내부에서 지침 구간 α에 λ∈{0.5,1,2,4,8} 배율(PASTA 계열).
    질량 보존(재정규화) 변형이 주 결과, 비보존은 총량 변화 공변량.

준수 선호 점수 (계획서 3.5)
    score = (1/|Y_c|)·logP(Y_c | X) − (1/|Y_v|)·logP(Y_v | X)      (teacher forcing)
  Y_c=준수 표기 후보(camel), Y_v=위반 표기 후보(snake). **조건 공통 고정 문자열.**
  개입 효과 = score(개입본) − score(손상본).  개입은 X(context)만 바꾸므로
  같은 개입 아래 Y_c·Y_v를 함께 채점한다.

판정 (계획서 3.6)
  - 표본 분할 A(기저 gap 추정 전용)·B(개입 실행) 분리 → Recovery Ratio = B의 개입
    변화량 ÷ A의 기저 gap(준수 baseline − 손상 baseline). 분자·분모 상관 회피.
  - 음성 대조(효과 없어야 함): no-op(donor=self), 지침 구간 patch(비트 동일),
    무작위 단위 patch. 이들의 변화량 분포를 귀무분포로, P1a가 상위 꼬리인지 검정(주 판정).

모델 (CLAUDE.md 미결정 사항 — 개입용 크기 미확정)
  - 개입: 기본 3B fp16(2단계 관측과 연속). **양자화 금지**(activation 흔들림 → 노이즈).
    크기는 계획서 5.5 게이트로 Step 1에서 정하는 미결정 사항 → --model로 재정의 가능.
  - 검증: 커널 등가·불변식만 보므로 크기 무관 → 기본 1.5B fp32.

Colab 제약(CLAUDE.md)
  - prefix KV 캐시를 (pair, seed)당 1회 만들고 전 config 재사용. 매번 prefill 재계산 금지.
  - 결과는 세션 단위 JSONL append, (design, model, split, config_key, pair_idx, seed)로 재개.
  - 개입 실험에 양자화 모델 쓰지 않는다.

usage:
  python src/stage3_intervention.py --validate                 # 불변식 자기검증 (게이트, 먼저)
  python src/stage3_intervention.py --run-a --n-seeds 3        # A분할: 기저 gap
  python src/stage3_intervention.py --run-b --p1a --p2         # B분할: 개입 + 음성 대조
  python src/stage3_intervention.py --summary-only             # 집계(Recovery Ratio·귀무검정)
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_main import system_prompt, INSTRUCTION            # noqa: E402
from exp1_pilot import INSTRUCTION_RULE, PREFIX_FUNCS        # noqa: E402
from stage2_attention import _char_span_to_tokens, _repeat_kv  # noqa: E402  (정본 재사용)

DESIGN_VERSION = "stage3_intervention_v2"   # v1(초안)과 안 섞이게 — L25 조건·규모일치 대조 추가
ATTN_NAME = "stage3_intervene"        # AttentionInterface 등록 이름

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
DEFAULT_OUT = os.path.join(RESULTS_DIR, "stage3_intervention.jsonl")
PAIRS_PATH = os.path.join(RESULTS_DIR, "matched_pairs.json")

RUN_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"       # 개입 기본(2단계 관측과 동일). 양자화 금지.
VALIDATE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # 불변식 검증(크기 무관) → 작은 것 fp32

LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]   # P2 배율 (계획서 3.6)
SPLIT_SEED = 20240803                  # A/B 분할 고정 seed (점수 무관 — 규칙 1)
N_SPLIT = 50                           # A·B 각 50쌍 (계획서 3.6)

# 결정 지점 후보(조건 공통 고정 문자열, 규칙 2) — **작업 지시와 의미가 일치해야 한다.**
#   작업="값을 표시용으로 포맷" ↔ 후보 이름 formatValue/format_value. (사전순 임의 선택 폐기)
#   camel=준수(Python에 camelCase 요구 = 비관용 방향), snake=위반(관용). exp1/2단계 매핑과 동일.
#   채점은 (1/|Y|) 정규화라 두 후보의 토큰 수 일치는 불필요(토큰 정확 일치는 context pair 규칙).
DECISION_SPEC = "formats a value for display"
DECISION_PAIR = {"camel": "formatValue", "snake": "format_value"}

# 2단계 split-half로 재현된 개입 후보 층(docs/experiments/04b). **L25 우선(주 개입).**
#   나머지는 보조 분석. --main-layer / --aux-layers로 재정의 가능.
MAIN_LAYER = 25
AUX_LAYERS = [7, 31, 27, 15, 20, 34, 33]

SINK = 4                               # attention sink 앞 토큰 — 대조 위치에서 제외(2단계와 동일)


# ─────────────────────────────────────────────────────────────
# 개입 컨텍스트 — 커스텀 attention 함수가 읽는 전역 상태.
#   config마다 iv_set()으로 완전히 초기화한다(세션 전역이지만 매 채점 리셋).
# ─────────────────────────────────────────────────────────────

class _IV:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False        # False면 커스텀 forward는 표준 SDPA passthrough(비트 동일)
        self.mode = "none"         # "none" | "p1a" | "p2"
        # P1a: 코드 구간 열을 donor로 덮어쓴다 (KV group 단위)
        self.donor = None          # {layer_idx: (k_seg (b,kvh,m,d), v_seg (b,kvh,m,d))}
        self.code_pos = None       # LongTensor — donor를 넣을 kv 위치(prefix 내부)
        self.groups = None         # 덮어쓸 KV group 인덱스 리스트(kvh 축). None=전체
        self.layers = None         # 개입할 층 집합. None=전체
        # P2: 지침 구간 α에 배율
        self.instr_pos = None      # LongTensor — 지침 열 위치
        self.heads = None          # 배율 걸 query head 인덱스 리스트(H 축). None=전체
        self.lam = 1.0             # 배율 λ
        self.conserve = True       # 질량 보존(재정규화) 여부
        # 검증용 훅
        self.assert_rowsum = False
        self.last_rowsum_err = 0.0


_IVX = _IV()


def iv_off():
    _IVX.reset()


def iv_p1a(donor, code_pos, groups=None, layers=None):
    _IVX.reset()
    _IVX.active = True
    _IVX.mode = "p1a"
    _IVX.donor = donor
    _IVX.code_pos = code_pos
    _IVX.groups = groups
    _IVX.layers = layers


def iv_p2(instr_pos, lam, heads=None, layers=None, conserve=True):
    _IVX.reset()
    _IVX.active = True
    _IVX.mode = "p2"
    _IVX.instr_pos = instr_pos
    _IVX.lam = lam
    _IVX.heads = heads
    _IVX.layers = layers
    _IVX.conserve = conserve


# ─────────────────────────────────────────────────────────────
# 커스텀 attention 함수 (GQA 단위 주의 — 규칙 4)
# ─────────────────────────────────────────────────────────────

def intervene_attention_forward(module, query, key, value, attention_mask,
                                scaling, dropout=0.0, **kwargs):
    """AttentionInterface용.

    query: (b, H, q, d), key/value: (b, kvh, kv, d) — GQA라 아직 복제 전(= KV group 축).
      · 비활성/prefill: 표준 SDPA로 위임 → 개입 없는 경로는 비트 동일(검증 불변식).
      · P1a: pre-repeat key/value의 kvh 축(=KV group)에서 code_pos 열을 donor로 치환 → SDPA.
             donor 치환 후 α는 SDPA가 재계산한다(계획서 3.6 "α 재계산").
      · P2: post-softmax α(H 축=query head)에서 지침 열에 λ 배율 → (질량 보존시) 재정규화.
    """
    import torch
    import torch.nn.functional as F

    layer_idx = int(getattr(module, "layer_idx", -1))
    n_rep = query.shape[1] // key.shape[1]
    q_len = query.shape[2]

    active = _IVX.active and (_IVX.layers is None or layer_idx in _IVX.layers)

    # ── P1a: key/value(pre-repeat) 코드 구간 열 치환 → 표준 SDPA ──
    if active and _IVX.mode == "p1a":
        k = key.clone()
        v = value.clone()
        pos = _IXV_pos(torch, _IVX.code_pos, k.device)
        dk, dv = _IVX.donor[layer_idx]
        groups = _IVX.groups if _IVX.groups is not None else range(k.shape[1])
        for g in groups:                       # KV group 단위(규칙 4) — 최대 4개
            k[:, g].index_copy_(1, pos, dk[:, g].to(k.dtype).to(k.device))
            v[:, g].index_copy_(1, pos, dv[:, g].to(v.dtype).to(v.device))
        key_states = _repeat_kv(k, n_rep)
        value_states = _repeat_kv(v, n_rep)
        mask = (attention_mask[:, :, :, : key_states.shape[-2]]
                if attention_mask is not None else None)
        out = F.scaled_dot_product_attention(
            query, key_states, value_states, attn_mask=mask,
            dropout_p=0.0, scale=scaling, is_causal=mask is None and q_len > 1)
        return out.transpose(1, 2).contiguous(), None

    # ── P2: 명시적 softmax → 지침 열 α에 λ 배율 (query head 단위, 규칙 4) ──
    if active and _IVX.mode == "p2":
        key_states = _repeat_kv(key, n_rep)
        value_states = _repeat_kv(value, n_rep)
        kvlen = key_states.shape[-2]
        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # (b,H,q,kv)
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :kvlen]
        elif q_len > 1:
            # mask가 없으면 SDPA는 is_causal로 마스킹한다. 명시적 경로는 직접 causal 처리 —
            # 안 하면 후보 토큰이 미래를 봐 λ=1도 항등이 아니게 된다(V6 버그). end-정렬 causal.
            qi = torch.arange(q_len, device=scores.device).view(q_len, 1)
            ki = torch.arange(kvlen, device=scores.device).view(1, kvlen)
            causal = (ki <= (kvlen - q_len + qi)).view(1, 1, q_len, kvlen)
            scores = scores.masked_fill(~causal, float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)             # (b,H,q,kv)
        pos = _IXV_pos(torch, _IVX.instr_pos, probs.device)
        pos = pos[pos < probs.shape[-1]]
        heads = _IVX.heads if _IVX.heads is not None else range(probs.shape[1])
        for h in heads:                        # query head 단위(규칙 4) — 최대 28개
            col = probs[:, h].index_select(-1, pos)     # (b,q,m)
            probs[:, h].index_copy_(-1, pos, col * _IVX.lam)
        if _IVX.conserve:                      # 질량 보존: 행합 1로 재정규화
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
        if _IVX.assert_rowsum:
            _IVX.last_rowsum_err = float((probs.sum(-1) - 1).abs().max().item())
        probs = probs.to(query.dtype)
        out = torch.matmul(probs, value_states).transpose(1, 2).contiguous()
        return out, None

    # ── 비활성/prefill: 표준 SDPA passthrough (개입 없는 경로는 비트 동일) ──
    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)
    mask = (attention_mask[:, :, :, : key_states.shape[-2]]
            if attention_mask is not None else None)
    out = F.scaled_dot_product_attention(
        query, key_states, value_states, attn_mask=mask,
        dropout_p=0.0, scale=scaling, is_causal=mask is None and q_len > 1)
    return out.transpose(1, 2).contiguous(), None


def _IXV_pos(torch, pos_list, device):
    """위치 리스트/텐서를 device 텐서로."""
    if pos_list is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if hasattr(pos_list, "to"):
        return pos_list.to(device)
    return torch.tensor(pos_list, dtype=torch.long, device=device)


def register_attention():
    """등록 API가 버전마다 달라 두 경로 시도(stage2와 동일)."""
    try:
        from transformers import AttentionInterface
        AttentionInterface.register(ATTN_NAME, intervene_attention_forward)
        return
    except Exception:
        pass
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[ATTN_NAME] = intervene_attention_forward


# ─────────────────────────────────────────────────────────────
# KV 캐시 추출/재구성 (transformers 버전 견고화)
#   prefix 캐시를 legacy 튜플로 보관 → config마다 새 캐시로 재구성해 재사용(불변).
# ─────────────────────────────────────────────────────────────

def _extract_kv(cache):
    try:
        leg = cache.to_legacy_cache()
        return [(k, v) for (k, v) in leg]
    except Exception:
        pass
    if getattr(cache, "key_cache", None) is not None:
        return list(zip(cache.key_cache, cache.value_cache))
    if getattr(cache, "layers", None) is not None:
        return [(l.keys, l.values) for l in cache.layers]
    raise RuntimeError("알 수 없는 KV 캐시 형식")


def _make_cache(kv_list):
    from transformers import DynamicCache
    try:
        return DynamicCache.from_legacy_cache(tuple(kv_list))
    except Exception:
        c = DynamicCache()
        for i, (k, v) in enumerate(kv_list):
            c.update(k, v, i)
        return c


def _slice_kv(legacy, pos_tensor):
    """legacy 캐시 각 층 K/V에서 pos 열만 뽑아 donor 형태 {layer:(k_seg,v_seg)}로.

    pos_tensor 순서 = 나중에 forward에서 index_copy_로 되쓸 위치 순서와 반드시 동일해야 한다.
    """
    d = {}
    for li, (k, v) in enumerate(legacy):
        p = pos_tensor.to(k.device)
        d[li] = (k.index_select(2, p).detach(), v.index_select(2, p).detach())
    return d


# ─────────────────────────────────────────────────────────────
# 프롬프트·구간·후보 구성
# ─────────────────────────────────────────────────────────────

def _load_pairs(single_token_only=False):
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)
    if single_token_only:
        pairs = [p for p in pairs if p.get("single_token_divergence")]
    return pairs


def choose_decision_pair(pairs):
    """결정 지점 후보(규칙 2) — **작업 지시와 의미가 일치하는 고정 쌍**(DECISION_PAIR).

    사전순 임의 선택은 작업(`formats a value for display`)과 무관한 이름(applyInvoice)이 뽑혀
    부적합했다. 이제 작업과 맞는 formatValue/format_value로 고정한다. context pool에선 제외된다.
    """
    return dict(DECISION_PAIR)


def split_context_pairs(pairs, decision):
    """주 context pair(= 클러스터 라벨)를 A/B로 분할. **점수 무관**(규칙 1).

    고정 seed 셔플 후 order[:50]=A, [50:100]=B. decision 쌍 제외.
    (동반 함수 pool은 order[100:]에서 별도로 뽑아 A/B와 겹치지 않게 한다.)
    """
    pool = [p for p in pairs if p["camel"] != decision["camel"]]
    rng = random.Random(SPLIT_SEED)
    order = list(range(len(pool)))
    rng.shuffle(order)
    a = [pool[i] for i in order[:N_SPLIT]]
    b = [pool[i] for i in order[N_SPLIT:2 * N_SPLIT]]
    return a, b


def companion_pairs(pairs, decision, n):
    """동반(companion) 토글 함수 — 조작 강도를 위한 **고정 다수 위반**용.

    exp1에서 효과는 위반이 다수(8~12/12)일 때 났다. 여기선 주 context pair 하나로는
    준수·위반 gap이 0에 가깝다(camelCase 지침이 강해 함수 1개론 결정이 안 바뀜).
    → 주 pair + (n−1)개의 **고정 동반 matched pair**를 조건에 따라 **함께 토글**한다
    (준수=전부 camel, 위반=전부 snake). 전부 토큰정렬 pair라 정렬 유지.
    동반 pool은 A/B(order[:100])와 겹치지 않게 order[100:]에서 고정 추출 → 전 세션 공통.
    """
    pool = [p for p in pairs if p["camel"] != decision["camel"]]
    rng = random.Random(SPLIT_SEED)
    order = list(range(len(pool)))
    rng.shuffle(order)
    comp_idx = order[2 * N_SPLIT: 2 * N_SPLIT + max(0, n - 1)]
    return [pool[i] for i in comp_idx]


def build_prompt(tokenizer, ctx_pair, ctx_style, seed, companions):
    """주 context pair + 동반 pair들을 ctx_style(camel=준수/snake=위반)로 **함께 토글**.

    반환: dict(prompt_text, input_ids, prompt_len, code_pos, ctxfunc_pos, ctxname_pos, instr_pos).
      code_pos    = 선행 코드 블록 전체(모든 토글 함수) → P1a 주 치환 구간.
      ctxfunc/ctxname = **주 context pair** 함수/이름 span(단계적 범위·전파 대조 기준점).
    프롬프트는 "```python\\ndef"로 프라이밍(끝 공백 없음) → 다음 토큰이 새 함수의 결정 지점.
    후보 이름은 선행 공백으로 붙인다(BPE 경계 고정, `decision_strings`).
    """
    funcs = [ctx_pair] + list(companions)               # [0]=주 pair
    bodies = ["def {n}(value):\n    return _process(value)".format(n=f[ctx_style])
              for f in funcs]
    order = random.Random(seed).sample(range(len(bodies)), len(bodies))
    parts = [bodies[i] for i in order]
    prefix = "\n\n\n".join(parts)
    primary_body = bodies[0]                             # 주 pair 본문(현재 style)
    primary_name = ctx_pair[ctx_style]

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": ("Here is the existing code in this project:\n\n"
                                     f"```python\n{prefix}\n```\n\n"
                                     f"Now add a function that {DECISION_SPEC}.")},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    prompt_text += "```python\ndef"                     # 끝 공백 없음 — 이름은 선행 공백으로 붙임

    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets, input_ids = enc["offset_mapping"], enc["input_ids"]

    cp = prompt_text.find(prefix)
    code_pos = _char_span_to_tokens(offsets, cp, cp + len(prefix)) if cp != -1 else []

    # 주 context 함수 span + 그 안의 이름 토큰 span(단계적 범위·pre/post 대조 기준).
    cf = prompt_text.find(primary_body, cp) if cp != -1 else -1
    ctxfunc_pos = _char_span_to_tokens(offsets, cf, cf + len(primary_body)) if cf != -1 else []
    cn = prompt_text.find(primary_name, cf) if cf != -1 else -1
    ctxname_pos = _char_span_to_tokens(offsets, cn, cn + len(primary_name)) if cn != -1 else []

    instr = INSTRUCTION_RULE[INSTRUCTION].strip()
    ci = prompt_text.find(instr)
    instr_pos = _char_span_to_tokens(offsets, ci, ci + len(instr)) if ci != -1 else []

    return dict(prompt_text=prompt_text, input_ids=input_ids, prompt_len=len(input_ids),
                code_pos=code_pos, ctxfunc_pos=ctxfunc_pos, ctxname_pos=ctxname_pos,
                instr_pos=instr_pos)


def decision_strings(decision):
    """결정 지점 후보 문자열 — primer가 무공백 "def"로 끝나므로 이름에 선행 공백을 붙인다.

    반환: (" applyInvoice", " apply_invoice") 형태. 두 문자열의 첫 토큰(" apply…")은 공유되고
    표기 분기 지점부터 갈린다. 조건 무관 고정 문자열(규칙 2).
    """
    return " " + decision["camel"], " " + decision["snake"]


def candidate_ids(tokenizer, base_prompt_text, name):
    """base_prompt_text + name 을 토큰화해 name 토큰 id만 뽑는다.

    base가 full의 접두이면(경계 병합 없음) full[len(base):]가 후보 토큰.
    아니면 None(경계 병합 → 폐기).
    """
    base = tokenizer(base_prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(base_prompt_text + name, add_special_tokens=False)["input_ids"]
    if full[:len(base)] != base:
        return None
    return full[len(base):]


# ─────────────────────────────────────────────────────────────
# 준수 선호 점수 (teacher forcing) — 계획서 3.5
# ─────────────────────────────────────────────────────────────

def _logprob(model, torch, prefix_kv, prefix_len, trigger_id, cand_ids):
    """logP(cand | prefix, trigger). prefix 캐시는 불변(config마다 새 캐시로 재구성).

    입력 = [trigger] + cand[:-1]  → 로짓 T개가 cand[0..T-1]을 예측.
    개입은 _IVX가 켜져 있으면 이 forward의 모든 query 위치(결정 지점 + teacher-forcing
    step 전부)에 동일 적용된다(계획서 3.6).
    """
    import torch.nn.functional as F
    cache = _make_cache(prefix_kv)
    inp_list = [trigger_id] + cand_ids[:-1]
    inp = torch.tensor([inp_list], dtype=torch.long, device=model.device)
    T = len(cand_ids)
    cache_pos = torch.arange(prefix_len, prefix_len + T, device=model.device)
    pos_ids = cache_pos.unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=inp, past_key_values=cache, use_cache=True,
                    position_ids=pos_ids, cache_position=cache_pos)
    logits = out.logits[0]                        # (T, V)
    lp = 0.0
    for i, t in enumerate(cand_ids):
        lp += float(F.log_softmax(logits[i].float(), dim=-1)[t].item())
    return lp


def pref_score(model, torch, prefix_kv, prefix_len, trigger_id, yc_ids, yv_ids):
    """준수 선호 점수 = (1/|Y_c|)logP(Y_c) − (1/|Y_v|)logP(Y_v). 같은 개입 아래 둘 다 채점."""
    lpc = _logprob(model, torch, prefix_kv, prefix_len, trigger_id, yc_ids)
    lpv = _logprob(model, torch, prefix_kv, prefix_len, trigger_id, yv_ids)
    return lpc / len(yc_ids) - lpv / len(yv_ids)


# ─────────────────────────────────────────────────────────────
# 한 (context pair, seed) 세션 준비 — prefix 캐시·donor 1회 생성 후 재사용
# ─────────────────────────────────────────────────────────────

def prepare_session(model, tokenizer, torch, ctx_pair, decision, seed, companions):
    """위반/준수 두 조건의 prefix 캐시를 만들고 정렬을 검증한다.

    주 pair + 동반 pair들을 함께 토글(준수=전부 camel / 위반=전부 snake).
    반환: dict 또는 (None, 사유).
    """
    P = {s: build_prompt(tokenizer, ctx_pair, s, seed, companions)
         for s in ("camel", "snake")}
    a, b = P["camel"], P["snake"]

    # 규칙 3: 토큰 정확 일치 — 이름 토큰 밖에서 다르면 K/V 치환이 정의 안 됨 → 폐기.
    if len(a["input_ids"]) != len(b["input_ids"]):
        return None, "len_mismatch"
    diff = [i for i, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y]
    code_set = set(a["code_pos"])
    if set(diff) - code_set:
        return None, "align_out_of_code"          # 코드 구간 밖에서 토큰 상이 → 폐기
    if a["code_pos"] != b["code_pos"] or a["instr_pos"] != b["instr_pos"]:
        return None, "seg_mismatch"

    # 결정 지점 후보(규칙 2: 조건 공통 고정 문자열).
    # 두 context 프롬프트는 context 함수명(camel/snake) 때문에 서로 다르다 — 그게 조건이다.
    # 하지만 후보는 "```python\ndef " 프라이밍 **뒤**에 붙는 동일 문자열이라, 두 조건에서
    # 같은 토큰열이어야 한다. 양쪽에서 뽑아 일치를 검증(어긋나면 폐기).
    dc, dv = decision_strings(decision)
    yc_a = candidate_ids(tokenizer, a["prompt_text"], dc)
    yv_a = candidate_ids(tokenizer, a["prompt_text"], dv)
    yc_b = candidate_ids(tokenizer, b["prompt_text"], dc)
    yv_b = candidate_ids(tokenizer, b["prompt_text"], dv)
    if None in (yc_a, yv_a, yc_b, yv_b):
        return None, "cand_boundary_merge"
    if yc_a != yc_b or yv_a != yv_b:
        return None, "cand_context_dependent"     # 후보 토큰이 조건에 의존 → 채점 부적격
    yc, yv = yc_a, yv_a

    # trigger("def " 마지막 토큰)도 두 조건에서 같아야 한다(공통 프라이밍).
    if a["input_ids"][-1] != b["input_ids"][-1]:
        return None, "trigger_mismatch"
    ids = a["input_ids"]
    prefix_len = len(ids) - 1
    trigger_id = ids[-1]

    def clip(ps):
        return [p for p in ps if SINK <= p < prefix_len]

    code_pos = clip(a["code_pos"])
    ctxfunc_pos = clip(a.get("ctxfunc_pos", []))
    ctxname_pos = clip(a.get("ctxname_pos", []))
    instr_pos = clip(a["instr_pos"])
    # 대조용 위치 — **다중 토글에선 코드 블록 전체가 조건 간 다르다**(모든 함수가 토글).
    #   그래서 "진짜 무변화" 대조는 조건 간 **동일한 영역**뿐: 지침·코드 블록 밖 보일러플레이트.
    #   (단일 토글 때의 prectx/postctx는 다른 토글 함수에 오염되므로 폐기.)
    code_start = min(code_pos) if code_pos else prefix_len
    seg_set = set(code_pos) | set(instr_pos)
    boiler_pos = [p for p in range(SINK, code_start) if p not in seg_set][:len(code_pos)]

    def T(ps):
        return torch.tensor(ps, dtype=torch.long)

    pos = {
        "code": T(code_pos),            # 선행 코드 전체(모든 토글 함수 = 주 개입 범위)
        "ctxfunc": T(ctxfunc_pos),      # 주 context 함수 전체(중간 범위 — 단계적 국소)
        "ctxname": T(ctxname_pos),      # 주 context 함수 이름 토큰만(최소 범위 — 2단계 이름 신호)
        "instr": T(instr_pos),          # 지침(조건 간 비트 동일 → ≈0 대조)
        "boiler": T(boiler_pos),        # 코드 블록 밖 보일러플레이트(조건 간 동일 → 진짜 ≈0)
    }

    # prefix(마지막 trigger 토큰 제외) prefill → 캐시. 개입 없이(iv_off) 표준 경로.
    iv_off()
    kv = {}
    for s in ("camel", "snake"):
        inp = torch.tensor([P[s]["input_ids"][:-1]], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model(input_ids=inp, use_cache=True)
        kv[s] = _extract_kv(out.past_key_values)

    return dict(
        ctx_pair=ctx_pair, seed=seed, n_ctx=len(companions) + 1,
        prefix_len=prefix_len, trigger_id=trigger_id,
        pos=pos, instr_pos=pos["instr"],
        yc_ids=yc, yv_ids=yv,
        kv_viol=kv["snake"], kv_comp=kv["camel"],   # 손상(위반)·준수 prefix 캐시(불변, 재사용)
        n_diff=len(diff), n_layers=len(kv["snake"]),
        n_kv_heads=kv["snake"][0][0].shape[1],
    ), None


def _score_baselines(model, torch, sess):
    """개입 없는 두 baseline: 손상본(위반 prefix) / 준수본(준수 prefix)."""
    iv_off()
    damaged = pref_score(model, torch, sess["kv_viol"], sess["prefix_len"],
                         sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    compliant = pref_score(model, torch, sess["kv_comp"], sess["prefix_len"],
                           sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    return damaged, compliant


# ─────────────────────────────────────────────────────────────
# 재개 / 저장
# ─────────────────────────────────────────────────────────────

def load_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("design_version") != DESIGN_VERSION:
                continue
            done.add((r["split"], r["config_key"], r["pair_idx"], r["seed"], r["model"]))
    return done


def append_jsonl(out_path, record):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# 실행: A분할(기저 gap) / B분할(개입 + 음성 대조)
# ─────────────────────────────────────────────────────────────

def _load_model(model_id, dtype_str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    register_attention()
    dtype = {"fp16": torch.float16, "fp32": torch.float32,
             "bf16": torch.bfloat16}[dtype_str]
    print(f"[{model_id}] 로드 중 (attn_implementation={ATTN_NAME}, {dtype_str}, 양자화 없음)...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
        attn_implementation=ATTN_NAME).eval()
    return model, tok, torch


def run_a(args):
    """A분할: 개입 없이 손상/준수 baseline만 기록 → 기저 gap 추정 전용."""
    model, tok, torch = _load_model(args.model, args.dtype)
    pairs = _load_pairs()
    decision = choose_decision_pair(pairs)
    a_pairs, _ = split_context_pairs(pairs, decision)
    companions = companion_pairs(pairs, decision, args.n_ctx)
    print(f"[A] n_ctx={args.n_ctx} (주 pair + 동반 {len(companions)}개 함께 토글)")
    if args.max_pairs:
        a_pairs = a_pairs[:args.max_pairs]
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    for pi, ctx in enumerate(a_pairs):
        for seed in seeds:
            key = ("A", "baseline", pi, seed, args.model)
            if key in done:
                continue
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        companions)
            if sess is None:
                append_jsonl(args.out, _skip_rec("A", "baseline", pi, seed, ctx,
                                                 decision, args.model, why))
                continue
            damaged, compliant = _score_baselines(model, torch, sess)
            append_jsonl(args.out, {
                "design_version": DESIGN_VERSION, "model": args.model,
                "split": "A", "config_key": "baseline",
                "pair_idx": pi, "seed": seed,
                "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                "decision_camel": decision["camel"], "decision_snake": decision["snake"],
                "score_damaged": damaged, "score_compliant": compliant,
                "gap": compliant - damaged,        # 기저 gap = 완전 회복 목표
                "n_diff": sess["n_diff"],
            })
            print(f"  A pair{pi} seed{seed}: gap={compliant - damaged:+.4f}")
    print_summary(args.out)


def _configs_b(args, n_layers, n_kv_heads, n_q_heads):
    """B분할 개입 config 목록. (config_key, kind, params).

    설계 핵심(리뷰 반영):
      · **P1a 주 조건 = L25 단독 × 전 KV group**(코드 K/V 이식). 2단계에서 재현된 L25가
        원인인지 검정.  전 층 각각을 같은 규모(1층×전 group)로 돌려, **L25 아닌 층들의 효과
        분포가 규모-일치 귀무분포**가 된다(무작위 층 대조를 결정적·포괄적으로 대체).
      · `p1a_full`(전 층×전 group)은 회복 상한선·양성 대조.
      · 음성 대조는 전부 **L25와 동일 규모**(1층×전 group), 위치만 바꿔 특이성 확인:
        no-op(donor=self)·지침 구간·무관 코드(filler)·코드 외 구간 → 모두 Δ≈0 기대.
      · P2 = 지침 α×λ. 전 층×전 head λ 곡선(단조성) + 전 층 국소(층별 귀무).
    """
    main = args.main_layer
    aux = args.aux_layers
    layers_all = list(range(n_layers))
    cfgs = []

    if args.p1a:
        cfgs.append(("p1a_full", "p1a",
                     dict(pos="code", donor="comp", groups=None, layers=None)))
        # 층별 단독(전 층) — 주(L25)·보조(aux)·귀무(나머지)가 한 규모로 나온다.
        for li in layers_all:
            cfgs.append((f"p1a_L{li}_allG", "p1a",
                         dict(pos="code", donor="comp", groups=None, layers=[li])))
        # L25 단계적 범위(보조) — 원인이 이름 토큰인지 이후 전파 표현인지 구분.
        #   name(이름만) ⊂ func(함수 전체) ⊂ code(선행 코드 전체=주).
        cfgs.append((f"p1a_L{main}_name", "p1a",
                     dict(pos="ctxname", donor="comp", groups=None, layers=[main])))
        cfgs.append((f"p1a_L{main}_func", "p1a",
                     dict(pos="ctxfunc", donor="comp", groups=None, layers=[main])))
        # L25 규모 음성 대조(위치만 교체) — 전부 조건 간 동일 영역이라 Δ≈0 기대:
        #   noop=자기이식(정확0)·instr=지침(비트동일)·boiler=코드 밖 보일러플레이트(동일).
        for pos_name, tag, donor in (("code", "noop", "self"),
                                     ("instr", "instr", "comp"),
                                     ("boiler", "boiler", "comp")):
            cfgs.append((f"ctl_{tag}_L{main}", "p1a",
                         dict(pos=pos_name, donor=donor, groups=None, layers=[main])))
        if args.sweep:
            # group별 국소화 — 후보 층에서 KV group 4개 단독.
            for li in [main] + aux:
                for g in range(n_kv_heads):
                    cfgs.append((f"p1a_L{li}_G{g}", "p1a",
                                 dict(pos="code", donor="comp", groups=[g], layers=[li])))

    if args.p2:
        for lam in LAMBDAS:                                  # 전 층×전 head λ 곡선(단조성)
            cfgs.append((f"p2_all_lam{lam}", "p2",
                         dict(lam=lam, heads=None, layers=None, conserve=True)))
            cfgs.append((f"p2_all_lam{lam}_noncons", "p2",
                         dict(lam=lam, heads=None, layers=None, conserve=False)))
        for li in layers_all:                                # 전 층 국소(층별 귀무)
            for lam in LAMBDAS:
                cfgs.append((f"p2_L{li}_lam{lam}", "p2",
                             dict(lam=lam, heads=None, layers=[li], conserve=True)))
        if args.sweep:
            # 층 × query head 28개 순회 — 후보 층에서.
            for li in [main] + aux:
                for h in range(n_q_heads):
                    for lam in LAMBDAS:
                        cfgs.append((f"p2_L{li}_H{h}_lam{lam}", "p2",
                                     dict(lam=lam, heads=[h], layers=[li], conserve=True)))
    return cfgs


def _apply_config(torch, sess, kind, params):
    """config에 맞춰 _IVX를 세팅. 반환: 개입에 쓸 prefix 캐시(위반본, 불변 재사용)."""
    if kind == "p1a":
        src = sess["kv_comp"] if params["donor"] == "comp" else sess["kv_viol"]
        positions = sess["pos"][params["pos"]]
        donor = _slice_kv(src, positions)                    # 준수/위반 캐시의 해당 위치 열
        iv_p1a(donor, positions, groups=params.get("groups"), layers=params.get("layers"))
    elif kind == "p2":
        iv_p2(sess["instr_pos"], params["lam"], heads=params.get("heads"),
              layers=params.get("layers"), conserve=params.get("conserve", True))
    return sess["kv_viol"]


def run_b(args):
    """B분할: 개입본 준수 선호 점수 변화량 = score(개입) − score(손상)."""
    model, tok, torch = _load_model(args.model, args.dtype)
    n_q = model.config.num_attention_heads
    pairs = _load_pairs()
    decision = choose_decision_pair(pairs)
    _, b_pairs = split_context_pairs(pairs, decision)
    companions = companion_pairs(pairs, decision, args.n_ctx)
    print(f"[B] n_ctx={args.n_ctx} (주 pair + 동반 {len(companions)}개 함께 토글)")
    if args.max_pairs:
        b_pairs = b_pairs[:args.max_pairs]
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    for pi, ctx in enumerate(b_pairs):
        for seed in seeds:
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        companions)
            if sess is None:
                append_jsonl(args.out, _skip_rec("B", "prep", pi, seed, ctx,
                                                 decision, args.model, why))
                continue
            damaged, compliant = _score_baselines(model, torch, sess)
            cfgs = _configs_b(args, sess["n_layers"], sess["n_kv_heads"], n_q)
            for ckey, kind, params in cfgs:
                key = ("B", ckey, pi, seed, args.model)
                if key in done:
                    continue
                prefix_kv = _apply_config(torch, sess, kind, params)
                s_iv = pref_score(model, torch, prefix_kv, sess["prefix_len"],
                                  sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
                iv_off()
                append_jsonl(args.out, {
                    "design_version": DESIGN_VERSION, "model": args.model,
                    "split": "B", "config_key": ckey, "kind": kind,
                    "pair_idx": pi, "seed": seed,
                    "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                    "decision_camel": decision["camel"],
                    "decision_snake": decision["snake"],
                    "score_damaged": damaged, "score_compliant": compliant,
                    "score_intervened": s_iv,
                    "delta": s_iv - damaged,        # 주 지표: 준수 선호 점수 변화량
                    "pos": params.get("pos"), "layers": params.get("layers"),
                    "groups": params.get("groups"), "heads": params.get("heads"),
                    "conserve": params.get("conserve", None),
                    "lam": params.get("lam", None),
                })
            print(f"  B pair{pi} seed{seed}: {len(cfgs)} configs")
    print_summary(args.out)


def _skip_rec(split, ckey, pi, seed, ctx, decision, model, why):
    return {"design_version": DESIGN_VERSION, "model": model, "split": split,
            "config_key": ckey, "pair_idx": pi, "seed": seed, "skipped": why,
            "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
            "decision_camel": decision["camel"], "decision_snake": decision["snake"]}


# ─────────────────────────────────────────────────────────────
# 불변식 자기검증 (--validate) — 실측 개입의 게이트
# ─────────────────────────────────────────────────────────────

def run_validate(args):
    """개입 하네스의 불변식을 검증한다. 통과가 실측(run-a/run-b)의 게이트.

    검증 항목:
      V1 SDPA passthrough 정확성 — iv_off 커스텀 forward == reference SDPA(비트 근사).
      V2 no-op 불변 — P1a donor=self → 준수 선호 점수가 손상본과 정확히 동일.
      V3 지침 patch ≈ 0 — 지침 구간 K/V 치환 Δ≈0(causal mask상 비트 동일).
      V4 GQA 단위 — n_q_heads·n_kv_heads 정합, P1a=KV group·P2=query head(규칙 4).
      V5 P2 질량 보존 — 재정규화 후 α 행합 == 1.
      V6 λ=1 항등 — P2 conserve·λ=1은 개입 없음과 동일.
      V7 주 층(L25) 이식이 손상 대비 점수를 움직이는지(하네스 sanity, 부호는 본실험).
    """
    model, tok, torch = _load_model(args.model_validate, "fp32")
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    n_layers = cfg.num_hidden_layers
    print(f"\n[검증] {args.model_validate} n_q_heads={n_q} n_kv_heads={n_kv} "
          f"n_layers={n_layers}")

    pairs = _load_pairs()
    decision = choose_decision_pair(pairs)
    _, b_pairs = split_context_pairs(pairs, decision)
    companions = companion_pairs(pairs, decision, args.n_ctx)
    print(f"[검증] n_ctx={args.n_ctx} (주 pair + 동반 {len(companions)}개 함께 토글)")
    # 첫 pair가 정렬/경계 사유로 폐기될 수 있으니 몇 개 시도(게이트가 단일 pair에 안 걸리게).
    sess = why = ctx = None
    tried = []
    for cand in b_pairs[:12]:
        s, w = prepare_session(model, tok, torch, cand, decision, seed=0,
                               companions=companions)
        tried.append((cand["camel"], w))
        if s is not None:
            sess, ctx = s, cand
            break
    assert sess is not None, (
        f"검증용 세션 준비 실패 — 시도한 pair들: {tried}\n"
        f"(decision={decision['camel']}/{decision['snake']}) "
        "대부분 cand_boundary_merge면 decision 후보 교체 필요.")
    print(f"[검증] 세션 pair = {ctx['camel']}/{ctx['snake']}, "
          f"decision = {decision['camel']}/{decision['snake']}")
    damaged, compliant = _score_baselines(model, torch, sess)

    # ── 진단(gap=0 원인 규명): span 길이·n_diff·캐시 실제 차이 ──
    li = args.main_layer
    kc, kv = sess["kv_comp"], sess["kv_viol"]
    cp = sess["pos"]["code"]

    def _maxdiff(a, b, idx=None):
        if idx is not None:
            if idx.numel() == 0:
                return None
            a = a.index_select(2, idx.to(a.device))
            b = b.index_select(2, idx.to(b.device))
        return float((a.float() - b.float()).abs().max().item())

    full_diff = max(_maxdiff(kc[l][0], kv[l][0]) for l in range(len(kc)))     # 전 층 K 최대차
    code_diff_L = _maxdiff(kc[li][0], kv[li][0], cp)                          # L25 code K 차
    print("\n[진단] gap=0 원인 규명")
    print(f"  n_diff(토큰 상이 수)={sess['n_diff']}  "
          f"len(code_pos)={cp.numel()}  len(instr_pos)={sess['instr_pos'].numel()}")
    print(f"  damaged={damaged:.6f}  compliant={compliant:.6f}  gap={compliant-damaged:.6f}")
    print(f"  준수·위반 prefill K 최대차(전층)={full_diff:.3e}  "
          f"L25 code 구간 K 최대차={code_diff_L}")
    print("  해석: code_pos=0 → span 버그 / L25차=0 but 전층차>0 → span 오배치 / "
          "차>0 but gap≈0 → 조작이 약함(설계)")

    results = {}
    results["diagnostics"] = {
        "n_diff": sess["n_diff"], "len_code_pos": int(cp.numel()),
        "len_instr_pos": int(sess["instr_pos"].numel()),
        "gap": compliant - damaged, "prefill_K_maxdiff_all": full_diff,
        "L25_code_K_maxdiff": code_diff_L,
    }

    # V4 — GQA 단위 (규칙 4)
    results["V4_gqa_units"] = {
        "pass": (n_q % max(n_kv, 1) == 0),        # group 공유가 정합
        "n_q_heads": n_q, "n_kv_heads": n_kv,
        "p1a_unit": "KV group", "p2_unit": "query head",
        "shared_per_group": n_q // max(n_kv, 1),
        "matches_plan_28x4": (n_q == 28 and n_kv == 4),
    }

    # V2 — no-op 불변 (전 층 코드 구간, donor=self)
    prefix_kv = _apply_config(torch, sess, "p1a",
                              dict(pos="code", donor="self", groups=None, layers=None))
    s_noop = pref_score(model, torch, prefix_kv, sess["prefix_len"],
                        sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    iv_off()
    results["V2_noop_invariance"] = {
        "pass": abs(s_noop - damaged) < 1e-4,
        "abs_diff": abs(s_noop - damaged),
    }

    # V3 — 지침 patch ≈ 0 (지침 구간 K/V는 두 조건 비트 동일 → 이식해도 무변화)
    prefix_kv = _apply_config(torch, sess, "p1a",
                              dict(pos="instr", donor="comp", groups=None, layers=None))
    s_instr = pref_score(model, torch, prefix_kv, sess["prefix_len"],
                         sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    iv_off()
    results["V3_instr_patch_zero"] = {
        "pass": abs(s_instr - damaged) < 1e-3,
        "abs_diff": abs(s_instr - damaged),
    }

    # V7 — L25 코드 이식이 실제로 점수를 움직이는가(하네스가 죽어있지 않은지, **부호 무관**).
    #   서로 다른 K/V를 L25에 넣으면 출력은 반드시 바뀐다 — 정확히 0이거나 NaN이면 배관 결함.
    #   기준: 유한 + |Δ| > 수치 바닥(1e-8). 작지만 0 아닌 실제 효과는 통과(효과 크기 판정 아님).
    import math
    delta_main = s_main = None
    prefix_kv = _apply_config(torch, sess, "p1a",
                              dict(pos="code", donor="comp", groups=None,
                                   layers=[args.main_layer]))
    s_main = pref_score(model, torch, prefix_kv, sess["prefix_len"],
                        sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    iv_off()
    delta_main = s_main - damaged
    results["V7_mainlayer_moves"] = {
        "pass": bool(math.isfinite(delta_main) and abs(delta_main) > 1e-8),
        "layer": args.main_layer, "delta_vs_damaged": delta_main,
        "note": "L25 코드 이식이 손상 대비 점수를 이동(유한·비영)시키는지. 부호·크기는 본실험 판정.",
    }

    # V5 — P2 질량 보존 + V6 λ=1 항등
    iv_p2(sess["instr_pos"], lam=4.0, heads=None, layers=None, conserve=True)
    _IVX.assert_rowsum = True
    s_p2 = pref_score(model, torch, sess["kv_viol"], sess["prefix_len"],
                      sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    rowsum_err = _IVX.last_rowsum_err
    iv_off()
    results["V5_p2_mass_conserved"] = {
        "pass": rowsum_err < 1e-4, "max_rowsum_err": rowsum_err,
    }
    iv_p2(sess["instr_pos"], lam=1.0, heads=None, layers=None, conserve=True)
    s_p2_id = pref_score(model, torch, sess["kv_viol"], sess["prefix_len"],
                         sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    iv_off()
    results["V6_lambda1_identity"] = {
        "pass": abs(s_p2_id - damaged) < 1e-3, "abs_diff": abs(s_p2_id - damaged),
    }

    # V1 — SDPA passthrough == reference (개입 없는 경로가 표준 attention과 비트 근사)
    ref_model = None
    try:
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_validate, torch_dtype=torch.float32, device_map="auto",
            attn_implementation="sdpa").eval()
    except Exception as e:
        results["V1_passthrough"] = {"pass": None, "note": f"reference 로드 실패: {e}"}
    if ref_model is not None:
        iv_off()
        full_ids = _validate_full_ids(tok, ctx, decision)
        inp = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            a = model(input_ids=inp).logits.float()
            b = ref_model(input_ids=inp.to(ref_model.device)).logits.float().to(a.device)
        maxerr = float((a - b).abs().max().item())
        results["V1_passthrough"] = {"pass": maxerr < 1e-3, "max_abs_err": maxerr}
        del ref_model

    # ── 출력 ──
    print("\n=== 3단계 개입 불변식 검증 ===")
    ok = True
    for k, v in results.items():
        p = v.get("pass")
        mark = "✅" if p else ("⚠️ " if p is None else "❌")
        ok = ok and (p is not False)
        print(f"  {mark} {k}: {json.dumps(v, ensure_ascii=False)}")
    print(f"\n기저: 손상={damaged:+.4f} 준수={compliant:+.4f} gap={compliant-damaged:+.4f}")
    print("게이트:", "PASS ✅ — 실측 진행 가능" if ok else "FAIL ❌ — 실측 금지")
    return ok


def _validate_full_ids(tok, ctx, decision):
    P = build_prompt(tok, ctx, "snake", 0, [])
    dc, _ = decision_strings(decision)
    yc = candidate_ids(tok, P["prompt_text"], dc)
    return P["input_ids"] + (yc or [])


# ─────────────────────────────────────────────────────────────
# 집계 — Recovery Ratio · 귀무검정
# ─────────────────────────────────────────────────────────────

def _load_rows(out_path):
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("design_version") == DESIGN_VERSION:
                        rows.append(r)
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ── two-way cluster bootstrap (이름쌍·seed 두 margin 독립 재표본) ──
#   config 하나당 (이름쌍, seed) 셀에 관측 1개뿐이라, 단순 (pair,seed) 클러스터를 재표본하면
#   사실상 iid 재표본과 같다(반복 문맥 구조 미반영). 그래서 **이름쌍 50개와 seed 3개를 각각
#   독립 복원추출**하고 그 교차셀 평균을 통계로 쓴다(crossed/two-way cluster bootstrap).

def _cells(rows, value):
    """(이름쌍, seed)→값 셀 표. 같은 셀 다중값은 평균. 반환: (cell, pairs, seeds)."""
    from collections import defaultdict
    acc = defaultdict(list)
    pairs, seeds = set(), set()
    for r in rows:
        v = r.get(value)
        if v is None:
            continue
        p, s = r.get("ctx_camel"), r.get("seed")
        acc[(p, s)].append(v)
        pairs.add(p)
        seeds.add(s)
    cell = {k: sum(v) / len(v) for k, v in acc.items()}
    return cell, sorted(pairs, key=lambda x: (x is None, x)), \
        sorted(seeds, key=lambda x: (x is None, x))


def _twoway_boot(rows, value, n_boot=2000, boot_seed=12345):
    """이름쌍·seed를 각각 복원추출해 교차셀 평균의 부트 분포. 반환: (point, boot_list)."""
    cell, pairs, seeds = _cells(rows, value)
    if not cell:
        return None, []

    def stat(P, S):
        vals = [cell[(p, s)] for p in P for s in S if (p, s) in cell]
        return sum(vals) / len(vals) if vals else None

    point = stat(pairs, seeds)
    rng = random.Random(boot_seed)
    npr, nse = len(pairs), len(seeds)
    boot = []
    for _ in range(n_boot):
        P = [pairs[rng.randrange(npr)] for _ in range(npr)]
        S = [seeds[rng.randrange(nse)] for _ in range(nse)]
        m = stat(P, S)
        if m is not None:
            boot.append(m)
    return point, boot


def _ci(boot, lo=2.5, hi=97.5):
    if not boot:
        return (None, None)
    b = sorted(boot)
    n = len(b)
    return (b[int(lo / 100 * n)], b[min(int(hi / 100 * n), n - 1)])


def _ratio_ci(num_boot, den_boot):
    """Recovery Ratio CI — 분자(B config)·분모(A gap)의 독립 부트 분포를 index로 짝지어 비율."""
    if not num_boot or not den_boot:
        return (None, None, None)
    m = min(len(num_boot), len(den_boot))
    ratios = [num_boot[i] / den_boot[i] for i in range(m) if abs(den_boot[i]) > 1e-9]
    if not ratios:
        return (None, None, None)
    ratios.sort()
    return (sum(ratios) / len(ratios),
            ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))])


def print_summary(out_path, n_boot=2000):
    rows = _load_rows(out_path)
    if not rows:
        print(f"(레코드 없음: {out_path})")
        return
    a = [r for r in rows if r.get("split") == "A" and "gap" in r]
    b = [r for r in rows if r.get("split") == "B" and "delta" in r]
    print(f"\n=== 3단계 개입 집계 [{DESIGN_VERSION}] ===  A={len(a)} B={len(b)}")

    gap_mean, gap_boot = _twoway_boot(a, "gap", n_boot, boot_seed=999)
    if gap_mean is not None:
        glo, ghi = _ci(gap_boot)
        print(f"A분할 기저 gap(준수−손상) = {gap_mean:+.4f}  95%CI[{glo:+.4f},{ghi:+.4f}] "
              f"(완전 회복 목표=분모)  [two-way boot: 이름쌍×seed]")

    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in b:
        by_cfg[r["config_key"]].append(r)

    def stat(ckey):
        m, boot = _twoway_boot(by_cfg[ckey], "delta", n_boot)
        lo, hi = _ci(boot)
        return boot, m, lo, hi

    # ── 규모-일치 귀무분포: 층별 단독(p1a_L*_allG) 중 주 층(L25) 제외한 나머지 층들의
    #    '층 평균 효과' 분포. 주 조건(L25)과 같은 규모(1층×전 group)라 공정 비교. ──
    per_layer = {}
    for ckey, rws in by_cfg.items():
        if ckey.startswith("p1a_L") and ckey.endswith("_allG"):
            li = int(ckey[len("p1a_L"):-len("_allG")])
            per_layer[li] = _mean([r["delta"] for r in rws])
    main_layer = _main_from(rows)
    null_layer_effects = [eff for li, eff in per_layer.items()
                          if li != main_layer and eff is not None]

    def layer_tail_p(eff):
        if eff is None or not null_layer_effects:
            return None
        ge = sum(1 for x in null_layer_effects if x is not None and x >= eff)
        return (ge + 1) / (len(null_layer_effects) + 1)

    # ── P1a 주 결과 ──
    print("\n── P1a (코드 K/V 이식) ──")
    print(f"{'config':26s} {'평균Δ':>9s} {'95%CI':>20s} {'RecovRatio':>11s} {'층귀무p':>8s}")
    p1a_keys = [f"p1a_L{main_layer}_allG", "p1a_full"] + \
               sorted(k for k in by_cfg if k.startswith("p1a_L") and k.endswith("_allG")
                      and k != f"p1a_L{main_layer}_allG")
    for ckey in p1a_keys:
        if ckey not in by_cfg:
            continue
        cl, m, lo, hi = stat(ckey)
        rr = _ratio_ci(cl, gap_boot)
        li = None
        if ckey.startswith("p1a_L") and ckey.endswith("_allG"):
            li = int(ckey[len("p1a_L"):-len("_allG")])
        p = layer_tail_p(per_layer.get(li)) if li is not None else None
        star = "  ← 주(L25)" if li == main_layer else (" (상한)" if ckey == "p1a_full" else "")
        ci = f"[{lo:+.4f},{hi:+.4f}]" if lo is not None else ""
        rr_s = f"{rr[0]:+.2f}" if rr[0] is not None else "-"
        p_s = f"{p:.4f}" if p is not None else "-"
        print(f"{ckey:26s} {m:+.4f} {ci:>20s} {rr_s:>11s} {p_s:>8s}{star}")

    # ── 단계적 범위(보조) — 이름 토큰 vs 전파 표현 구분 ──
    print("\n── L25 단계적 범위 (name ⊂ func ⊂ code) ──")
    for ckey in [f"p1a_L{main_layer}_name", f"p1a_L{main_layer}_func",
                 f"p1a_L{main_layer}_allG"]:
        if ckey in by_cfg:
            cl, m, lo, hi = stat(ckey)
            ci = f"[{lo:+.4f},{hi:+.4f}]" if lo is not None else ""
            print(f"{ckey:26s} {m:+.4f} {ci:>20s}")

    # ── 음성 대조(특이성) — 전부 조건 간 동일 영역 → Δ≈0 기대 ──
    print("\n── 대조 (L25 규모, 조건 간 동일 영역 → Δ≈0 기대) ──")
    for ckey in sorted(k for k in by_cfg if k.startswith("ctl_")):
        cl, m, lo, hi = stat(ckey)
        ci = f"[{lo:+.4f},{hi:+.4f}]" if lo is not None else ""
        print(f"{ckey:26s} {m:+.4f} {ci:>20s}")

    # ── P2 λ 단조 추세 ──
    lam_means = []
    for lam in LAMBDAS:
        k = f"p2_all_lam{lam}"
        if k in by_cfg:
            lam_means.append((lam, _mean([r["delta"] for r in by_cfg[k]])))
    if lam_means:
        print("\n── P2 (지침 α×λ, 전 층·전 head, 질량보존) ──")
        for lam, m in lam_means:
            print(f"  λ={lam:<4} 평균Δ={m:+.4f}" if m is not None else f"  λ={lam}: -")
        rho = _spearman([x for x, _ in lam_means],
                        [y for _, y in lam_means if y is not None])
        print(f"  단조 추세(Spearman ρ, λ vs Δ) = "
              f"{rho:+.3f}" if rho is not None else "  추세: -")

    print("\n※ 주 판정: p1a_L{}_allG 효과가 나머지 층(같은 규모) 귀무분포의 상위 꼬리인가."
          .format(main_layer))
    print("※ 건전성: ctl_noop=0(정확)·ctl_instr·ctl_boiler≈0(조건 간 동일 영역).")
    print("※ CI는 two-way(이름쌍×seed) cluster bootstrap. "
          "RecoveryRatio=config 효과 ÷ A분할 gap(분자·분모 독립 부트).")


def _main_from(rows):
    """레코드에서 주 층을 역추론(없으면 상수). p1a_L{n}_allG 중 주 조건 표시가 없으니 MAIN_LAYER."""
    return MAIN_LAYER


def _spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="3단계 — attention/KV 개입 (P1a·P2)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model", default=RUN_MODEL, help="개입 모델(양자화 금지)")
    ap.add_argument("--model-validate", default=VALIDATE_MODEL)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=8,
                    help="함께 토글할 context 함수 수(주 pair + 동반). 위반=전부 snake. "
                         "1이면 조작 약함(gap≈0). exp1처럼 다수(≈8) 권장")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="파일럿용 — context pair 수 제한(0=전체 50)")
    ap.add_argument("--main-layer", type=int, default=MAIN_LAYER,
                    help="주 개입 층(2단계 재현 L25)")
    ap.add_argument("--aux-layers", type=str, default=",".join(map(str, AUX_LAYERS)),
                    help="보조 분석 층(쉼표 구분)")
    # 모드
    ap.add_argument("--validate", action="store_true", help="불변식 자기검증(게이트)")
    ap.add_argument("--run-a", action="store_true", help="A분할: 기저 gap")
    ap.add_argument("--run-b", action="store_true", help="B분할: 개입 실행")
    ap.add_argument("--summary-only", action="store_true")
    # B분할 개입 선택
    ap.add_argument("--p1a", action="store_true", help="P1a(주) 실행")
    ap.add_argument("--p2", action="store_true", help="P2(보조) 실행")
    ap.add_argument("--sweep", action="store_true", help="group별·층×head 국소화 스윕까지")
    args = ap.parse_args()
    args.aux_layers = [int(x) for x in str(args.aux_layers).split(",") if x.strip() != ""]

    if args.validate:
        ok = run_validate(args)
        sys.exit(0 if ok else 1)
    if args.run_a:
        run_a(args)
    elif args.run_b:
        if not (args.p1a or args.p2):
            args.p1a = args.p2 = True                   # 기본: 둘 다(대조군은 P1a에 포함)
        run_b(args)
    elif args.summary_only:
        print_summary(args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
