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

DESIGN_VERSION = "stage3_intervention_v1"
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

# 새로 명명할 함수의 작업 지시(고정). 조건 간 동일.
DECISION_SPEC = "formats a value for display"


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
        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # (b,H,q,kv)
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
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
    """결정 지점 후보(조건 공통 고정 문자열). 점수 무관·안정 기준으로 하나 고정(규칙 2).

    단일 토큰 분기 쌍 중 이름 문자열 사전순 첫 번째. 이 쌍은 context pool에서 제외한다.
    """
    st = sorted((p for p in pairs if p.get("single_token_divergence")),
                key=lambda p: p["camel"])
    if not st:
        st = sorted(pairs, key=lambda p: p["camel"])
    return st[0]


def split_context_pairs(pairs, decision):
    """context pair를 A/B로 분할. **점수를 전혀 참조하지 않는다**(규칙 1).

    고정 seed 셔플 후 앞 N=A, 다음 N=B. decision 쌍은 제외.
    """
    pool = [p for p in pairs if p["camel"] != decision["camel"]]
    rng = random.Random(SPLIT_SEED)
    order = list(range(len(pool)))
    rng.shuffle(order)
    a = [pool[i] for i in order[:N_SPLIT]]
    b = [pool[i] for i in order[N_SPLIT:2 * N_SPLIT]]
    return a, b


def build_prompt(tokenizer, ctx_pair, ctx_style, seed, n_filler):
    """context pair 함수를 ctx_style(camel=준수 / snake=위반)로 넣은 프롬프트.

    반환: dict(prompt_text, input_ids, prompt_len, code_pos, instr_pos).
      code_pos  = 선행 코드 블록(context 함수 포함) 토큰 위치 → P1a 치환 구간.
      instr_pos = 시스템 프롬프트 꼬리 지침 문장 위치 → P2 배율 구간.
    프롬프트는 "```python\\ndef"로 프라이밍(끝 공백 없음) → 다음 토큰이 새 함수의 결정 지점.
    후보 이름은 **선행 공백**을 달아 붙인다(`decision_strings`): Qwen/GPT BPE는 선행 공백을
    단어에 병합하므로, 끝 공백 primer + 무공백 이름이면 base가 base+name의 토큰 접두가 아니게
    되어(경계 병합) 후보 추출이 깨진다. → primer는 무공백, 이름은 선행 공백으로 경계 고정.
    """
    name = ctx_pair[ctx_style]
    ctx_body = "def {n}(value):\n    return _process(value)".format(n=name)
    pool = [f["camel"] for f in PREFIX_FUNCS]            # filler는 준수(camel) 맥락 고정
    order = random.Random(seed).sample(range(len(pool)), min(n_filler, len(pool)))
    fillers = [pool[i] for i in order]
    slot = len(fillers) // 2
    parts = fillers[:slot] + [ctx_body] + fillers[slot:]
    prefix = "\n\n\n".join(parts)

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

    # 선행 코드 블록 구간(치환 대상). 준수/위반 두 조건에서 이름 토큰만 다르므로,
    # 블록 전체를 치환 구간으로 잡아도 이름 이전은 donor=self(무변화), 이름 이후만 실제 치환.
    cp = prompt_text.find(prefix)
    code_pos = _char_span_to_tokens(offsets, cp, cp + len(prefix)) if cp != -1 else []

    # 지침 구간(P2 대상).
    instr = INSTRUCTION_RULE[INSTRUCTION].strip()
    ci = prompt_text.find(instr)
    instr_pos = _char_span_to_tokens(offsets, ci, ci + len(instr)) if ci != -1 else []

    return dict(prompt_text=prompt_text, input_ids=input_ids, prompt_len=len(input_ids),
                code_pos=code_pos, instr_pos=instr_pos)


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

def prepare_session(model, tokenizer, torch, ctx_pair, decision, seed, n_filler):
    """위반/준수 두 조건의 prefix 캐시를 만들고 정렬을 검증한다.

    반환: dict 또는 (None, 사유). donor는 준수(camel) 캐시의 코드 구간 열 슬라이스.
    """
    P = {s: build_prompt(tokenizer, ctx_pair, s, seed, n_filler)
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
    code_pos_t = torch.tensor([p for p in a["code_pos"] if p < prefix_len],
                              dtype=torch.long)
    instr_pos_t = torch.tensor([p for p in a["instr_pos"] if p < prefix_len],
                               dtype=torch.long)

    # prefix(마지막 trigger 토큰 제외) prefill → 캐시. 개입 없이(iv_off) 표준 경로.
    iv_off()
    kv = {}
    for s in ("camel", "snake"):
        inp = torch.tensor([P[s]["input_ids"][:-1]], dtype=torch.long, device=model.device)
        with torch.no_grad():
            out = model(input_ids=inp, use_cache=True)
        kv[s] = _extract_kv(out.past_key_values)

    # donor(준수=camel)·self(위반=snake) 코드 구간 열 슬라이스 — 층별 (k,v).
    cpos_dev = code_pos_t.to(model.device)

    def slice_donor(legacy):
        d = {}
        for li, (k, v) in enumerate(legacy):
            d[li] = (k.index_select(2, cpos_dev).detach(),
                     v.index_select(2, cpos_dev).detach())
        return d

    return dict(
        ctx_pair=ctx_pair, seed=seed, n_filler=n_filler,
        prefix_len=prefix_len, trigger_id=trigger_id,
        code_pos=code_pos_t, instr_pos=instr_pos_t,
        yc_ids=yc, yv_ids=yv,
        kv_viol=kv["snake"], kv_comp=kv["camel"],
        donor_comp=slice_donor(kv["camel"]),      # 준수 실행 값 (P1a 주입용)
        donor_self=slice_donor(kv["snake"]),      # 위반 자기 값 (no-op 대조용)
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
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    for pi, ctx in enumerate(a_pairs):
        for seed in seeds:
            key = ("A", "baseline", pi, seed, args.model)
            if key in done:
                continue
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        args.n_filler)
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


def _configs_b(args, n_layers, n_kv_heads):
    """B분할에서 돌릴 개입 config 목록. (config_key, kind, params)."""
    cfgs = []
    layers_all = list(range(n_layers))
    if args.p1a:
        # 주 결과: 전 층 × 전 KV group 동시 치환(완전 회복 추정).
        cfgs.append(("p1a_full", "p1a",
                     dict(groups=None, layers=None, donor="comp")))
        # 국소화 스윕: 각 층 × 각 KV group 단독. (계획서: 전 층 × KV group 4)
        if args.sweep:
            for li in layers_all:
                for g in range(n_kv_heads):
                    cfgs.append((f"p1a_L{li}_G{g}", "p1a",
                                 dict(groups=[g], layers=[li], donor="comp")))
    if args.p2:
        # 주 결과 후보: 전 층 × 전 head × λ (질량 보존). λ=1은 no-op 확인.
        for lam in LAMBDAS:
            cfgs.append((f"p2_all_lam{lam}", "p2",
                         dict(lam=lam, heads=None, layers=None, conserve=True)))
            cfgs.append((f"p2_all_lam{lam}_noncons", "p2",
                         dict(lam=lam, heads=None, layers=None, conserve=False)))
        if args.sweep:
            for li in layers_all:
                for lam in LAMBDAS:
                    cfgs.append((f"p2_L{li}_lam{lam}", "p2",
                                 dict(lam=lam, heads=None, layers=[li], conserve=True)))
    if args.controls:
        # 음성 대조(효과 없어야 함) → 귀무분포.
        cfgs.append(("ctl_noop", "p1a",
                     dict(groups=None, layers=None, donor="self")))        # donor=self
        cfgs.append(("ctl_instr_patch", "p1a_instr",
                     dict(groups=None, layers=None, donor="comp")))        # 지침 구간(비트 동일)
        for r in range(args.n_random):
            cfgs.append((f"ctl_random_{r}", "p1a_random",
                         dict(donor="comp", rseed=1000 + r)))              # 무작위 단위
    return cfgs


def _apply_config(torch, sess, kind, params):
    """config에 맞춰 _IVX를 세팅. 반환: 개입에 쓸 prefix 캐시(위반본)."""
    donor = sess["donor_comp"] if params.get("donor") == "comp" else sess["donor_self"]
    if kind == "p1a":
        iv_p1a(donor, sess["code_pos"], groups=params.get("groups"),
               layers=params.get("layers"))
    elif kind == "p1a_instr":
        # 지침 구간에 준수 donor 주입 — causal mask상 지침 K/V는 두 조건 비트 동일 → Δ≈0.
        instr_donor = {}
        cpos = sess["instr_pos"].to(sess["kv_comp"][0][0].device)
        for li, (k, v) in enumerate(sess["kv_comp"]):
            instr_donor[li] = (k.index_select(2, cpos), v.index_select(2, cpos))
        iv_p1a(instr_donor, sess["instr_pos"], groups=None, layers=None)
    elif kind == "p1a_random":
        # 무작위 KV group·층에 준수 donor 주입(코드 구간이 아닌 무작위 위치).
        rng = random.Random(params["rseed"])
        n_layers, n_kv = sess["n_layers"], sess["n_kv_heads"]
        rlayers = sorted(rng.sample(range(n_layers), max(1, n_layers // 4)))
        rgroups = [rng.randrange(n_kv)]
        iv_p1a(donor, sess["code_pos"], groups=rgroups, layers=rlayers)
    elif kind == "p2":
        iv_p2(sess["instr_pos"], params["lam"], heads=params.get("heads"),
              layers=params.get("layers"), conserve=params.get("conserve", True))
    return sess["kv_viol"]


def run_b(args):
    """B분할: 개입본 준수 선호 점수 변화량 = score(개입) − score(손상)."""
    model, tok, torch = _load_model(args.model, args.dtype)
    pairs = _load_pairs()
    decision = choose_decision_pair(pairs)
    _, b_pairs = split_context_pairs(pairs, decision)
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    for pi, ctx in enumerate(b_pairs):
        for seed in seeds:
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        args.n_filler)
            if sess is None:
                append_jsonl(args.out, _skip_rec("B", "prep", pi, seed, ctx,
                                                 decision, args.model, why))
                continue
            damaged, compliant = _score_baselines(model, torch, sess)
            cfgs = _configs_b(args, sess["n_layers"], sess["n_kv_heads"])
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
                    "conserve": params.get("conserve", None),
                    "lam": params.get("lam", None),
                })
            print(f"  B pair{pi} seed{seed}: {len(cfgs)} configs done")
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
      V4 GQA 단위 — n_q_heads=28·n_kv_heads=4 확인, P1a=KV group·P2=query head(규칙 4).
      V5 P2 질량 보존 — 재정규화 후 α 행합 == 1.
      V6 λ=1 항등 — P2 conserve·λ=1은 개입 없음과 동일.
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
    # 첫 pair가 정렬/경계 사유로 폐기될 수 있으니 몇 개 시도(게이트가 단일 pair에 안 걸리게).
    sess = why = ctx = None
    tried = []
    for cand in b_pairs[:12]:
        s, w = prepare_session(model, tok, torch, cand, decision, seed=0,
                               n_filler=args.n_filler)
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

    results = {}

    # V4 — GQA 단위 (규칙 4)
    results["V4_gqa_units"] = {
        "pass": (n_q % max(n_kv, 1) == 0),        # group 공유가 정합
        "n_q_heads": n_q, "n_kv_heads": n_kv,
        "p1a_unit": "KV group", "p2_unit": "query head",
        "shared_per_group": n_q // max(n_kv, 1),
        "matches_plan_28x4": (n_q == 28 and n_kv == 4),
    }

    # V2 — no-op 불변 (donor=self)
    iv_p1a(sess["donor_self"], sess["code_pos"], groups=None, layers=None)
    s_noop = pref_score(model, torch, sess["kv_viol"], sess["prefix_len"],
                        sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    iv_off()
    results["V2_noop_invariance"] = {
        "pass": abs(s_noop - damaged) < 1e-4,
        "abs_diff": abs(s_noop - damaged),
    }

    # V3 — 지침 patch ≈ 0
    prefix_kv = _apply_config(torch, sess, "p1a_instr", dict(donor="comp"))
    s_instr = pref_score(model, torch, prefix_kv, sess["prefix_len"],
                         sess["trigger_id"], sess["yc_ids"], sess["yv_ids"])
    iv_off()
    results["V3_instr_patch_zero"] = {
        "pass": abs(s_instr - damaged) < 1e-3,
        "abs_diff": abs(s_instr - damaged),
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
    P = build_prompt(tok, ctx, "snake", 0, 6)
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


def print_summary(out_path):
    rows = _load_rows(out_path)
    if not rows:
        print(f"(레코드 없음: {out_path})")
        return
    a = [r for r in rows if r.get("split") == "A" and "gap" in r]
    b = [r for r in rows if r.get("split") == "B" and "delta" in r]
    print(f"\n=== 3단계 개입 집계 [{DESIGN_VERSION}] ===  A={len(a)} B={len(b)}")

    gap_a = _mean([r["gap"] for r in a])
    if gap_a:
        print(f"A분할 기저 gap(준수−손상) 평균 = {gap_a:+.4f}   (완전 회복 목표)")

    # config별 개입 변화량
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in b:
        by_cfg[r["config_key"]].append(r["delta"])

    # 귀무분포 = 무작위 단위 대조들의 변화량
    null = [d for k, v in by_cfg.items() if k.startswith("ctl_random")
            for d in v]

    def tail_p(delta_mean):
        if not null or delta_mean is None:
            return None
        ge = sum(1 for d in null if d >= delta_mean)
        return (ge + 1) / (len(null) + 1)

    print("\nconfig                         평균Δ      RecoveryRatio   상위꼬리p(vs무작위)")
    for ckey in sorted(by_cfg):
        dm = _mean(by_cfg[ckey])
        rr = (dm / gap_a) if (gap_a and dm is not None) else None
        p = tail_p(dm)
        rr_s = f"{rr:+.3f}" if rr is not None else "   -  "
        p_s = f"{p:.4f}" if p is not None else "  -   "
        print(f"  {ckey:28s} {dm:+.4f}    {rr_s:>10s}    {p_s:>8s}")

    print("\n※ 주 판정: P1a(p1a_full) 변화량이 무작위 대조 귀무분포의 상위 꼬리인가.")
    print("※ no-op·지침 patch 대조는 Δ≈0이어야 한다(하네스 건전성).")


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
    ap.add_argument("--n-filler", type=int, default=6)
    ap.add_argument("--n-random", type=int, default=8, help="무작위 단위 대조 수")
    # 모드
    ap.add_argument("--validate", action="store_true", help="불변식 자기검증(게이트)")
    ap.add_argument("--run-a", action="store_true", help="A분할: 기저 gap")
    ap.add_argument("--run-b", action="store_true", help="B분할: 개입 실행")
    ap.add_argument("--summary-only", action="store_true")
    # B분할 개입 선택
    ap.add_argument("--p1a", action="store_true", help="P1a(주) 실행")
    ap.add_argument("--p2", action="store_true", help="P2(보조) 실행")
    ap.add_argument("--controls", action="store_true", help="음성 대조 실행")
    ap.add_argument("--sweep", action="store_true", help="층×단위 국소화 스윕까지")
    args = ap.parse_args()

    if args.validate:
        ok = run_validate(args)
        sys.exit(0 if ok else 1)
    if args.run_a:
        run_a(args)
    elif args.run_b:
        if not (args.p1a or args.p2 or args.controls):
            args.p1a = args.p2 = args.controls = True   # 기본: 전부
        run_b(args)
    elif args.summary_only:
        print_summary(args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
