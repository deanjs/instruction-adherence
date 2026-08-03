"""
2단계 — attention 관측 (NIAR). 계획서 3.4(꺼내기) · 3.7(지표).

1단계(실험 1)에서 확인한 행동 — context 내 위반 코드가 후속 함수의 준수율을 낮춘다 —
이 일어날 때, 모델이 실제로 **위반 코드 구간을 더 참조(attention 배분)** 하는가를 관측한다.

  ※ 이건 상관 관측이지 인과 주장이 아니다(Jain & Wallace 2019).
     2단계 가설 H5/H6은 **탐색적 보고**다 — 주 가설은 H1a·H2a·H3 뿐(CLAUDE.md 규칙 6).
     인과 주장은 전적으로 3단계 개입에 근거한다.
  ※ 오픈 모델(Qwen)에서만 가능 — 내부 접근 필요.

이 스크립트가 하는 일 (로드맵 Step 2)
  1. `AttentionInterface`에 **query 길이 분기** attention 함수를 등록(계획서 3.4).
       q_len > 1 (prefill) → SDPA에 위임, attention 미계산(메모리 폭증 회피).
       q_len == 1 (decode) → 명시적 softmax → **구간별 합으로 즉시 축약, 축약값만 저장.**
  2. **512 토큰 eager 대조 검증**(--validate). prefill/decode 커널이 달라 필수.
       짧은 시퀀스에서 `output_attentions=True` eager 결과를 같은 방식으로 축약해
       캡처값과 수치가 일치하는지 확인한다. 통과가 관측(--observe)의 게이트다.
  3. **관측**(--observe). exp1 조건(compliant_remaining)을 그대로 재사용해
       지침/선행코드/위반코드 구간의 NIAR/NSCAR/NVCAR을 잰다.

지표 (계획서 3.7)
    NIAR = (지침 구간 attention 합 ÷ 지침 토큰 수) ÷ (전체 attention 합 ÷ 전체 토큰 수)
  균등 분포면 길이와 무관하게 1 → "지침이 짧아서"라는 반론 차단.
  NSCAR(선행 코드), NVCAR(위반 코드)도 같은 방식. attention sink(앞 4토큰)는 제외.
  보완 지표 ‖α·v‖(Kobayashi 2020): attention이 높아도 value가 작으면 실제 기여 작음.

모델
  - 관측: exp1 확증과 동일한 Qwen2.5-Coder-3B-Instruct(fp16, 양자화 없음) 재사용 —
    행동↔관측 연속성. (개입용 최종 모델 크기는 CLAUDE.md 미결정, 여기서 새로 정하지 않음.)
  - 검증: 커널 등가성만 보므로 크기 무관 → 기본 1.5B, fp32로 tight tolerance.

Colab 제약(CLAUDE.md): 관측 결과는 세션 단위로 JSONL append, (model, condition, seed)로 재개.

usage:
  python src/stage2_attention.py --validate                 # 512토큰 eager 대조 (먼저)
  python src/stage2_attention.py --observe --n-seeds 20      # 관측 실행
  python src/stage2_attention.py --observe --summary-only    # 집계만
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_main import (  # noqa: E402  (동일 context 문자열을 보장하려 그대로 재사용)
    INSTRUCTION,
    build_prefix as exp1_build_prefix,
    system_prompt,
    target_specs,
    user_prompt_first,
)
from exp1_pilot import INSTRUCTION_RULE, PREFIX_FUNCS  # noqa: E402

DESIGN_VERSION = "stage2_niar_v1"
ATTN_NAME = "niar_capture"          # AttentionInterface 등록 이름
SINK = 4                            # attention sink 제외 토큰 수 (Xiao 2024, 계획서 3.7)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
DEFAULT_OUT = os.path.join(RESULTS_DIR, "stage2_niar.jsonl")
OBSERVE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"      # exp1 확증과 동일
VALIDATE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"   # 검증은 크기 무관 → 작은 것


# ─────────────────────────────────────────────────────────────
# 캡처 컨텍스트 — attention 함수가 읽는 전역 상태.
#   실험용 전역이지만, 세션마다 ctx_begin으로 완전히 초기화한다.
# ─────────────────────────────────────────────────────────────

class _Ctx:
    def __init__(self):
        self.reset()

    def reset(self):
        self.enabled = False
        self.seg_positions = {}   # {seg_name: [토큰 인덱스,...]}  (sink 제외, prompt 영역만)
        self.seg_ntok = {}        # {seg_name: 토큰 수}
        self.prompt_len = 0
        self.value_norm = False
        self.capture_prefill_last = False  # prefill 마지막 query 행도 캡처(naming step 측정용)
        self.records = []         # [{layer, kv_len, total_kept, seg{...}, seg_vnorm{...}}]
        self._pos_cache = {}      # device별 index tensor 캐시


_CTX = _Ctx()


def ctx_begin(seg_positions, prompt_len, value_norm, capture_prefill_last=False):
    _CTX.reset()
    # sink 제외 + prompt 영역 안으로 클리핑
    _CTX.seg_positions = {
        name: [p for p in pos if SINK <= p < prompt_len]
        for name, pos in seg_positions.items()
    }
    _CTX.seg_ntok = {name: len(pos) for name, pos in _CTX.seg_positions.items()}
    _CTX.prompt_len = prompt_len
    _CTX.value_norm = value_norm
    _CTX.capture_prefill_last = capture_prefill_last
    _CTX.enabled = True


def ctx_end():
    _CTX.enabled = False
    return _CTX.records


def _pos_tensors(torch, device):
    key = str(device)
    if key not in _CTX._pos_cache:
        _CTX._pos_cache[key] = {
            name: torch.tensor(pos, dtype=torch.long, device=device)
            for name, pos in _CTX.seg_positions.items()
            if pos
        }
    return _CTX._pos_cache[key]


def reduce_attn_row(torch, row, pos_tensors, value_row=None):
    """decode query 한 행(H, kv)을 구간별로 축약. 캡처와 eager 검증이 공유(핵심).

    row: (H, kv) — head별 softmax 확률.  value_row: (H, kv, d) 또는 None.
    반환: {"kv_len", "total_kept"(head 평균 스칼라), "seg":{name:합}, "seg_vnorm":{...}}
      - total_kept = sink 제외 전 구간 확률 합의 head 평균.
      - seg[name]  = 그 구간 확률 합의 head 평균.
    """
    kv = row.shape[-1]
    kept = row[:, SINK:kv]                      # (H, kv-sink)
    rec = {"kv_len": int(kv),
           "total_kept": float(kept.sum(-1).mean().item()),
           "seg": {}, "seg_vnorm": {}}
    for name, idx in pos_tensors.items():
        idx = idx[idx < kv]                     # decode 진행 중 안전(항상 참이지만 방어)
        s = row.index_select(-1, idx).sum(-1)   # (H,)
        rec["seg"][name] = float(s.mean().item())
        if value_row is not None:
            w = row.index_select(-1, idx)                    # (H, m)
            vv = value_row.index_select(1, idx)              # (H, m, d)
            weighted = (w.unsqueeze(-1) * vv).sum(1)         # (H, d) — Σ α·v
            rec["seg_vnorm"][name] = float(weighted.norm(dim=-1).mean().item())
    return rec


# ─────────────────────────────────────────────────────────────
# query 길이 분기 attention 함수 (계획서 3.4)
# ─────────────────────────────────────────────────────────────

def _repeat_kv(hidden, n_rep):
    """GQA: K/V head를 query head 수만큼 복제. (b, kvh, s, d) → (b, kvh*n_rep, s, d)."""
    b, kvh, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    hidden = hidden[:, :, None, :, :].expand(b, kvh, n_rep, s, d)
    return hidden.reshape(b, kvh * n_rep, s, d)


def niar_attention_forward(module, query, key, value, attention_mask,
                           scaling, dropout=0.0, **kwargs):
    """AttentionInterface용. q_len으로 prefill/decode를 가른다.

    query: (b, H, q, d), key/value: (b, kvh, kv, d) — GQA라 아직 복제 전.
    반환: (attn_output (b, q, H, d), attn_weights or None) — HF 관례.
    """
    import torch
    import torch.nn.functional as F

    n_rep = query.shape[1] // key.shape[1]
    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)
    q_len = query.shape[2]

    if q_len > 1:
        # prefill — SDPA에 위임. attention 값 미계산(메모리 폭증·FlashAttention 비호환 회피).
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # naming step 측정: 마지막 query 행(= 다음 토큰을 예측하는 위치)만 명시적으로 축약.
        # 한 행뿐이라 메모리 폭증 없음. 첫 생성 토큰(이름)의 참조를 잡는다.
        if _CTX.enabled and _CTX.capture_prefill_last:
            q_last = query[:, :, -1:, :]                                   # (b,H,1,d)
            s = torch.matmul(q_last, key_states.transpose(2, 3)) * scaling  # (b,H,1,kv)
            if attention_mask is not None:
                s = s + attention_mask[:, :, -1:, : key_states.shape[-2]]
            p = F.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
            row = p[0, :, 0, :]                                             # (H, kv)
            vrow = value_states[0] if _CTX.value_norm else None            # (H, kv, d)
            rec = reduce_attn_row(torch, row, _pos_tensors(torch, p.device), vrow)
            rec["layer"] = int(getattr(module, "layer_idx", -1))
            rec["step"] = 0                                                 # naming step
            _CTX.records.append(rec)
        attn_output = F.scaled_dot_product_attention(
            query, key_states, value_states,
            attn_mask=mask,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
            is_causal=mask is None and q_len > 1,
        )
        return attn_output.transpose(1, 2).contiguous(), None

    # decode (q_len == 1) — 명시적 softmax → 구간별 합으로 즉시 축약, 원본 폐기.
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling   # (b,H,1,kv)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(probs, value_states).transpose(1, 2).contiguous()

    if _CTX.enabled:
        row = probs[0, :, 0, :]                              # (H, kv), batch=1 가정
        vrow = value_states[0] if _CTX.value_norm else None  # (H, kv, d)
        rec = reduce_attn_row(torch, row, _pos_tensors(torch, probs.device), vrow)
        rec["layer"] = int(getattr(module, "layer_idx", -1))
        rec["step"] = int(rec["kv_len"] - _CTX.prompt_len)   # 1,2,... (첫 decode=1)
        _CTX.records.append(rec)                             # 축약값만 저장

    return attn_output, probs


def register_attention():
    """등록 API가 버전마다 다르므로 두 경로를 시도."""
    try:
        from transformers import AttentionInterface
        AttentionInterface.register(ATTN_NAME, niar_attention_forward)
        return
    except Exception:
        pass
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[ATTN_NAME] = niar_attention_forward


# ─────────────────────────────────────────────────────────────
# 구간(segment) 매핑 — 문자열 span → 토큰 index
# ─────────────────────────────────────────────────────────────

def _char_span_to_tokens(offsets, c0, c1):
    """offset_mapping에서 문자 구간 [c0,c1)과 겹치는 토큰 index 목록."""
    out = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:            # 특수 토큰 등 빈 span
            continue
        if a < c1 and b > c0:  # 겹침
            out.append(i)
    return out


def _find_all(haystack, needle, start=0):
    idxs, i = [], haystack.find(needle, start)
    while i != -1:
        idxs.append(i)
        i = haystack.find(needle, i + 1)
    return idxs


def prefix_with_spans(compliant_remaining, seed):
    """exp1_main.build_prefix를 그대로 재현하되, 함수별 (텍스트, 위반 여부)까지 돌려준다.

    exp1의 정본 문자열과 반드시 일치해야 하므로(조건 동일성) 끝에서 assert로 검증한다.
    """
    n = len(PREFIX_FUNCS)
    perm = random.Random(seed).sample(range(n), n)
    compliant_idx = set(perm[:compliant_remaining])
    parts, is_viol = [], []
    for i in perm:
        if i in compliant_idx:
            parts.append(PREFIX_FUNCS[i]["camel"])
            is_viol.append(False)
        else:
            parts.append(PREFIX_FUNCS[i]["snake"])
            is_viol.append(True)
    prefix = "\n\n\n".join(parts)
    assert prefix == exp1_build_prefix(compliant_remaining, seed)[0], \
        "prefix가 exp1 정본과 불일치 — 조건 동일성이 깨졌다"
    return prefix, parts, is_viol


def compute_segments(tokenizer, prompt_text, prefix, parts, is_viol):
    """prompt_text에서 지침/선행코드/위반코드 구간의 토큰 index를 계산.

    반환: {"instruction":[...], "prefix_code":[...], "violation_code":[...]},
          그리고 (input_ids, prompt_len).
    """
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    prompt_len = len(input_ids)

    seg = {}

    # 지침 구간 — 시스템 프롬프트 꼬리의 규칙 문장.
    instr = INSTRUCTION_RULE[INSTRUCTION].strip()
    ci = prompt_text.find(instr)
    seg["instruction"] = (_char_span_to_tokens(offsets, ci, ci + len(instr))
                          if ci != -1 else [])

    # 선행 코드 구간 — 코드블록 안 prefix 전체.
    cp = prompt_text.find(prefix)
    seg["prefix_code"] = (_char_span_to_tokens(offsets, cp, cp + len(prefix))
                          if cp != -1 else [])

    # 위반 코드 구간 — prefix 안에서 snake 함수들의 span 합집합.
    viol = []
    if cp != -1:
        for text, bad in zip(parts, is_viol):
            if not bad:
                continue
            # 같은 본문이 여러 번 나올 수 있으니 prefix 범위 안의 첫 등장만.
            occ = _find_all(prompt_text, text, cp)
            occ = [o for o in occ if cp <= o <= cp + len(prefix)]
            if occ:
                viol += _char_span_to_tokens(offsets, occ[0], occ[0] + len(text))
    seg["violation_code"] = sorted(set(viol))

    return seg, input_ids, prompt_len


# ─────────────────────────────────────────────────────────────
# NIAR 집계
# ─────────────────────────────────────────────────────────────

def niar_from_records(records, seg_ntok):
    """캡처 records → 층별·전체 NIAR/…AR. 균등분포면 1.

    per-step per-layer: xAR = (seg합/seg토큰수) ÷ (total_kept/(kv-sink)).
    """
    per_layer = {}   # {seg: {layer: [step별 값]}}
    overall = {}     # {seg: [모든 step·layer 값]}
    for r in records:
        kv = r["kv_len"]
        denom = r["total_kept"] / max(kv - SINK, 1)
        if denom <= 0:
            continue
        for name, s in r["seg"].items():
            ntok = seg_ntok.get(name, 0)
            if ntok == 0:
                continue
            val = (s / ntok) / denom
            per_layer.setdefault(name, {}).setdefault(r["layer"], []).append(val)
            overall.setdefault(name, []).append(val)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    layer_profile = {
        name: {lyr: mean(vs) for lyr, vs in sorted(layers.items())}
        for name, layers in per_layer.items()
    }
    return {
        "overall": {name: mean(vs) for name, vs in overall.items()},
        "n_records": len(records),
        "layer_profile": layer_profile,
    }


# ─────────────────────────────────────────────────────────────
# 512 토큰 eager 대조 검증 (계획서 3.4) — Step 2의 핵심 게이트
# ─────────────────────────────────────────────────────────────

def _validation_messages():
    """검증 전용 짧은 프롬프트(<512 토큰). 커널 등가성만 보므로 내용은 최소.

    지침 + 코드 2줄을 넣어 구간 매핑까지 함께 검증한다.
    """
    code = ("def get_user_orders(user_id):\n"
            "    return db.query(user_id)\n\n\n"
            "def parse_iso_date(text):\n"
            "    return datetime.fromisoformat(text)")
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user",
         "content": ("Here is the existing code in this project:\n\n"
                     f"```python\n{code}\n```\n\n"
                     "Now add a function that validates an email address.")},
    ]
    return messages, code


def _reduce_eager(torch, attentions_step, pos_tensors, value=None):
    """eager outputs.attentions의 한 decode step(층 tuple)을 캡처와 동일 방식으로 축약."""
    out = []
    for layer, attn in enumerate(attentions_step):
        row = attn[0, :, 0, :]             # (H, kv)
        rec = reduce_attn_row(torch, row, pos_tensors, None)
        rec["layer"] = layer
        out.append(rec)
    return out


def run_validate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    register_attention()
    model_id = args.model or VALIDATE_MODEL
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[검증] {model_id} dtype={dtype} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    messages, code = _validation_messages()
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # 구간: 지침 + 코드블록.
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets, input_ids = enc["offset_mapping"], enc["input_ids"]
    prompt_len = len(input_ids)
    if prompt_len > args.max_prompt_tokens:
        print(f"경고: 프롬프트 {prompt_len} 토큰 > {args.max_prompt_tokens}. "
              "검증 프롬프트를 줄여라.", file=sys.stderr)
    instr = INSTRUCTION_RULE[INSTRUCTION].strip()
    ci, cc = prompt_text.find(instr), prompt_text.find(code)
    seg = {
        "instruction": _char_span_to_tokens(offsets, ci, ci + len(instr)),
        "prefix_code": _char_span_to_tokens(offsets, cc, cc + len(code)),
    }
    print("  구간 토큰수:", {k: len(v) for k, v in seg.items()}, "prompt_len:", prompt_len)

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)

    # ── run 1: 캡처 커널 ──
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=ATTN_NAME).to(device).eval()
    ctx_begin(seg, prompt_len, value_norm=False)
    with torch.no_grad():
        model.generate(**{k: v.to(device) for k, v in inputs.items()},
                       max_new_tokens=args.gen_tokens, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    captured = ctx_end()
    pos_t = _pos_tensors(torch, torch.device(device))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── run 2: eager 정답 ──
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    with torch.no_grad():
        out = model.generate(**{k: v.to(device) for k, v in inputs.items()},
                             max_new_tokens=args.gen_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id,
                             return_dict_in_generate=True, output_attentions=True)

    # eager: attentions[step]는 층 tuple. step 0은 prefill(q=prompt_len) → 캡처 대상 아님.
    # decode step(q==1)만 골라 kv_len으로 캡처 record와 정렬.
    eager_by_key = {}
    for step_attn in out.attentions:
        if step_attn[0].shape[2] != 1:   # q_len != 1 → prefill, 건너뜀
            continue
        for rec in _reduce_eager(torch, step_attn, pos_t):
            eager_by_key[(rec["kv_len"], rec["layer"])] = rec

    # 비교
    max_diff = 0.0
    n_cmp, n_missing = 0, 0
    for cap in captured:
        key = (cap["kv_len"], cap["layer"])
        ref = eager_by_key.get(key)
        if ref is None:
            n_missing += 1
            continue
        for name in cap["seg"]:
            d = abs(cap["seg"][name] - ref["seg"].get(name, float("nan")))
            max_diff = max(max_diff, d)
            n_cmp += 1
        max_diff = max(max_diff, abs(cap["total_kept"] - ref["total_kept"]))

    tol = args.tol
    ok = (max_diff <= tol) and n_cmp > 0 and n_missing == 0
    print(f"\n  비교 {n_cmp}쌍, 정렬실패 {n_missing}, 최대 절대오차 = {max_diff:.2e} "
          f"(tol {tol:.0e})")
    print(f"  판정: {'PASS ✅ — 캡처=eager, 관측 진행 가능' if ok else 'FAIL ❌ — 커널 불일치'}")
    if not ok:
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# 관측 (--observe)
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
            try:
                r = json.loads(line)
                done.add((r["design_version"], r["model"],
                          r["compliant_remaining"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(out_path, record):
    with open(out_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def observe_session(model, tokenizer, torch, compliant_remaining, seed, args):
    prefix, parts, is_viol = prefix_with_spans(compliant_remaining, seed)
    spec = target_specs(seed, 1)[0]     # 주 결과 = prefix 직후 첫 함수(위치1)

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt_first(prefix, spec)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    seg, input_ids, prompt_len = compute_segments(
        tokenizer, prompt_text, prefix, parts, is_viol)

    inputs = tokenizer(prompt_text, return_tensors="pt",
                       add_special_tokens=False).to(model.device)

    # 대응 설계: 같은 seed를 조건 간 공유(CLAUDE.md).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ctx_begin(seg, prompt_len, value_norm=args.value_norm)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.gen_tokens,
            do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id)
    records = ctx_end()
    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    seg_ntok = {name: len(pos) for name, pos in seg.items()}
    niar = niar_from_records(records, seg_ntok)

    return {
        "design_version": DESIGN_VERSION,
        "model": model.name_or_path,
        "instruction": INSTRUCTION,
        "compliant_remaining": compliant_remaining,
        "violation_count": len(PREFIX_FUNCS) - compliant_remaining,
        "seed": seed,
        "prompt_len": prompt_len,
        "seg_ntok": seg_ntok,
        "sink": SINK,
        "decode": {"do_sample": args.do_sample, "temperature": args.temperature,
                   "top_p": args.top_p, "gen_tokens": args.gen_tokens},
        "niar": niar["overall"],            # {instruction, prefix_code, violation_code}
        "n_decode_steps_x_layers": niar["n_records"],
        "layer_profile": niar["layer_profile"],
        "value_norm": args.value_norm,
        "generated_head": gen_text[:80],
    }


def _load_observe_rows(out_path):
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


def print_observe_summary(out_path, levels):
    rows = _load_observe_rows(out_path)
    if not rows:
        print(f"(design_version={DESIGN_VERSION} 레코드 없음: {out_path})")
        return

    cell = {}
    for r in rows:
        cell.setdefault(r["compliant_remaining"], []).append(r)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    print(f"\n=== 2단계 관측 요약 [{DESIGN_VERSION}] / 조건별 평균 NIAR (>1 = 평균보다 많이 참조) ===")
    print("열 = 남은 compliant 예시 수(4=대부분 준수 … 0=전부 위반)\n")
    names = ["instruction", "prefix_code", "violation_code"]
    labels = {"instruction": "지침 NIAR", "prefix_code": "선행코드 NSCAR",
              "violation_code": "위반코드 NVCAR"}
    header = "  compliant 남음 :" + "".join(f"  {cr:^7}" for cr in levels)
    print(header)
    for nm in names:
        line = f"  {labels[nm]:<14s}:"
        for cr in levels:
            v = mean([s["niar"].get(nm) for s in cell.get(cr, [])])
            line += f"  {v:^7.2f}" if v is not None else f"  {'--':^7}"
        print(line)
    print("\n  ※ H5(위반 조건에서 지침 NIAR↓)/H6(Violation-style에서 NSCAR↑)은 탐색적 보고.")
    print("  ※ 유의성은 JSONL을 조건 대비로 별도 검정. attention 크기 ≠ 인과(계획서 3.7).")


def print_layer_summary(out_path, levels, top_n):
    """층별 진단 — 전체 평균이 평탄해도 특정 층에서 조건이 갈리는지 본다.

    각 구간에 대해 층별로 조건(levels) 값을 세션 평균하고,
    Δ = (가장 위반 많은 조건) − (가장 준수 많은 조건) 로 정렬해 |Δ| 큰 층을 보여준다.
      - 지침(NIAR):   H5면 위반↑에서 낮아짐 → Δ 음수 기대
      - 위반코드(NVCAR): H6 방향이면 위반↑에서 높아짐 → Δ 양수 기대
    층 전부 평탄(|Δ|≈0)이면 coarse 수준에서 신호 없음.
    """
    rows = _load_observe_rows(out_path)
    if not rows:
        print(f"(design_version={DESIGN_VERSION} 레코드 없음: {out_path})")
        return

    lo, hi = max(levels), min(levels)   # lo=가장 준수(4), hi=가장 위반(0)
    cell = {}
    for r in rows:
        cell.setdefault(r["compliant_remaining"], []).append(r)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    # 등장하는 층 번호 수집
    layers = set()
    for r in rows:
        for seg_map in r.get("layer_profile", {}).values():
            layers.update(int(k) for k in seg_map)
    layers = sorted(layers)

    def layer_val(cr, seg, layer):
        vals = []
        for s in cell.get(cr, []):
            lp = s.get("layer_profile", {}).get(seg, {})
            v = lp.get(str(layer), lp.get(layer))
            if v is not None:
                vals.append(v)
        return mean(vals)

    names = ["instruction", "prefix_code", "violation_code"]
    labels = {"instruction": "지침 NIAR", "prefix_code": "선행코드 NSCAR",
              "violation_code": "위반코드 NVCAR"}
    print(f"\n=== 층별 진단 [{DESIGN_VERSION}] / Δ = 조건{hi}(위반多) − 조건{lo}(준수多) ===")
    print(f"세션수: " + ", ".join(f"cr{cr}={len(cell.get(cr, []))}" for cr in levels))
    for nm in names:
        deltas = []
        for L in layers:
            a, b = layer_val(hi, nm, L), layer_val(lo, nm, L)
            if a is not None and b is not None:
                deltas.append((L, a - b, b, a))
        if not deltas:
            continue
        deltas.sort(key=lambda t: abs(t[1]), reverse=True)
        max_abs = deltas[0][1]
        print(f"\n  [{labels[nm]}] |Δ| 최대 층 = L{deltas[0][0]} (Δ={max_abs:+.3f})  "
              f"— 전 층 평탄이면 이 값이 0 근처")
        print(f"    {'층':>4} {'조건'+str(lo):>8} {'조건'+str(hi):>8} {'Δ(hi−lo)':>9}")
        for L, d, b, a in deltas[:top_n]:
            print(f"    L{L:<3d} {b:8.3f} {a:8.3f} {d:+9.3f}")
    print("\n  해석: 어느 구간이든 |Δ| 큰 층이 있으면 그 층에 조건 신호 → 정밀 재측정 대상.")
    print("        전 구간·전 층에서 |Δ|≈0 이면 coarse 수준 null (측정창=naming step 검토).")


def run_observe(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    max_cr = len(PREFIX_FUNCS)
    if any(cr < 0 or cr > max_cr for cr in args.levels):
        sys.exit(f"--levels 는 0~{max_cr} 범위: {args.levels}")

    if args.layer_summary:
        print_layer_summary(args.out, args.levels, args.top_layers)
        return
    if args.summary_only:
        print_observe_summary(args.out, args.levels)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = args.model or OBSERVE_MODEL     # --model 미지정이면 exp1과 동일한 3B
    if not torch.cuda.is_available():
        sys.exit("CUDA 없음 — Colab을 GPU 런타임(T4)으로 바꿔라. "
                 "attention 관측은 GPU 필수(fp16, CPU 불가).")
    if "int8" in model_id.lower() or "4bit" in model_id.lower():
        sys.exit("양자화 모델 금지 — activation 노이즈로 관측이 흔들린다(CLAUDE.md).")

    register_attention()
    seeds = [args.base_seed + k for k in range(args.n_seeds)]
    done = load_done(args.out)
    pending = [(cr, s) for cr in args.levels for s in seeds
               if (DESIGN_VERSION, model_id, cr, s) not in done]
    print(f"완료 {len(done)}, 남은 {len(pending)} 세션 "
          f"(조건 {len(args.levels)} × seed {args.n_seeds})")
    if not pending:
        print_observe_summary(args.out, args.levels)
        return

    print(f"[{model_id}] 로드 중 (attn_implementation={ATTN_NAME}, fp16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        attn_implementation=ATTN_NAME).eval()

    for cr, s in pending:
        print(f"  compliant={cr} seed={s} ...", end=" ", flush=True)
        rec = observe_session(model, tokenizer, torch, cr, s, args)
        append_jsonl(args.out, rec)
        n = rec["niar"]
        print("NIAR지침={:.2f} 위반={:.2f}".format(
            n.get("instruction") or float("nan"),
            n.get("violation_code") or float("nan")))

    print_observe_summary(args.out, args.levels)


# ─────────────────────────────────────────────────────────────
# 내용-분리 측정 (--content) — 2단계를 깨끗이 닫는 대조
#
# 관측(--observe)의 위반 코드 gradient는 조건마다 위반 '구간 크기'가 달라져 생긴 교란이었다.
# 여기선 Step 0의 토큰 수 일치 pair(matched_pairs.json)로 **같은 위치·같은 길이**에서
# 이름만 camel↔snake로 바꿔, 순수 '내용' 효과만 본다(계획서 3.7 보조 검증).
#   - 측정 순간 = naming step: prompt를 "```python\ndef "까지 주고, 다음 토큰(=이름)을
#     예측하는 위치(prefill 마지막 행)의 attention을 target 이름 토큰에 대해 잰다.
#   - 길이가 같으니 정규화 불필요 → raw attention 합을 직접 비교. ‖α·v‖도 함께.
#   - 대응: 같은 seed·같은 filler 배치에서 camel/snake만 교체(pair 내 대응).
# ─────────────────────────────────────────────────────────────

DESIGN_VERSION_CONTENT = "stage2_content_v1"
DEFAULT_CONTENT_OUT = os.path.join(RESULTS_DIR, "stage2_content.jsonl")
PAIRS_PATH = os.path.join(RESULTS_DIR, "matched_pairs.json")

# target 함수 본문(고정). 이름만 pair로 갈아끼운다. camelCase가 규칙(=snake가 위반).
TARGET_BODY = "def {name}(value):\n    return _process(value)"


def _load_pairs(single_token_only):
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)
    if single_token_only:
        pairs = [p for p in pairs if p.get("single_token_divergence")]
    return pairs


def build_content_prefix(pair, style, seed, n_filler):
    """filler(camel, 고정) 사이 가운데 슬롯에 target 함수를 넣는다. style로 이름만 교체.

    반환: (prefix 문자열, target 함수 문자열).
    filler는 exp1 PREFIX_FUNCS의 camel 본문 재사용(준수=camel 맥락).
    """
    name = pair[style]                              # "camel" or "snake"
    target = TARGET_BODY.format(name=name)
    pool = [f["camel"] for f in PREFIX_FUNCS]
    order = random.Random(seed).sample(range(len(pool)), min(n_filler, len(pool)))
    fillers = [pool[i] for i in order]
    slot = len(fillers) // 2                         # 가운데
    parts = fillers[:slot] + [target] + fillers[slot:]
    return "\n\n\n".join(parts), target


def content_session(model, tokenizer, torch, pair, pair_idx, seed, args):
    """한 pair·seed에서 camel/snake 두 조건의 naming-step attention을 target에 대해 측정."""
    spec_desc = "formats a value for display"       # 새로 만들 함수(고정 작업 지시)

    per_style = {}
    align = None
    for style in ("camel", "snake"):
        prefix, target = build_content_prefix(pair, style, seed, args.n_filler)
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": ("Here is the existing code in this project:\n\n"
                                         f"```python\n{prefix}\n```\n\n"
                                         f"Now add a function that {spec_desc}.")},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_text += "```python\ndef "               # naming step 바로 앞까지 프라이밍

        enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets, input_ids = enc["offset_mapping"], enc["input_ids"]
        prompt_len = len(input_ids)

        # target 함수·이름 토큰 위치
        ct = prompt_text.find(target)
        name = pair[style]
        cn = prompt_text.find(name, ct)
        seg = {
            "target_func": _char_span_to_tokens(offsets, ct, ct + len(target)),
            "target_name": _char_span_to_tokens(offsets, cn, cn + len(name)),
        }
        per_style[style] = dict(input_ids=input_ids, prompt_len=prompt_len, seg=seg,
                                prompt_text=prompt_text)

    # ── 정렬 검증: 두 조건의 토큰열이 이름 토큰(들)만 빼고 완전히 같아야 위치·길이 고정 ──
    a, b = per_style["camel"], per_style["snake"]
    if len(a["input_ids"]) != len(b["input_ids"]):
        return None, "len_mismatch"
    diff = [i for i, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y]
    name_pos = set(a["seg"]["target_name"])
    if a["seg"]["target_name"] != b["seg"]["target_name"] or set(diff) - name_pos:
        return None, "align_fail"                    # 이름 위치 밖에서 토큰이 달라짐 → 폐기
    align = {"n_diff": len(diff), "name_tok": a["seg"]["target_name"]}

    # ── 두 조건 각각 단일 forward(prefill)로 naming-step attention 캡처 ──
    out = {}
    for style in ("camel", "snake"):
        st = per_style[style]
        inputs = tokenizer(st["prompt_text"], return_tensors="pt",
                           add_special_tokens=False).to(model.device)
        ctx_begin(st["seg"], st["prompt_len"], value_norm=True, capture_prefill_last=True)
        with torch.no_grad():
            model(**inputs)                          # 생성 없이 1회 forward
        out[style] = ctx_end()

    def by_layer(recs, seg, field):
        return {r["layer"]: r[field].get(seg) for r in recs if seg in r[field]}

    cam_name = by_layer(out["camel"], "target_name", "seg")
    sna_name = by_layer(out["snake"], "target_name", "seg")
    cam_vn = by_layer(out["camel"], "target_name", "seg_vnorm")
    sna_vn = by_layer(out["snake"], "target_name", "seg_vnorm")
    layers = sorted(cam_name)

    def mean(d):
        vs = [v for v in d.values() if v is not None]
        return sum(vs) / len(vs) if vs else None

    return {
        "design_version": DESIGN_VERSION_CONTENT,
        "model": model.name_or_path,
        "pair_idx": pair_idx,
        "camel": pair["camel"], "snake": pair["snake"],
        "total_tokens": pair.get("total_tokens"),
        "seed": seed,
        "n_filler": args.n_filler,
        "align": align,
        # naming step, target 이름 토큰에 대한 raw attention(head 평균) — 길이 동일이라 비정규화
        "attn_name": {"camel": mean(cam_name), "snake": mean(sna_name)},
        "attn_name_by_layer": {"camel": cam_name, "snake": sna_name},
        # 보완 지표 ‖α·v‖ (Kobayashi 2020)
        "vnorm_name": {"camel": mean(cam_vn), "snake": mean(sna_vn)},
        "vnorm_name_by_layer": {"camel": cam_vn, "snake": sna_vn},
    }, None


def print_content_summary(out_path):
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("design_version") == DESIGN_VERSION_CONTENT:
                        rows.append(r)
    if not rows:
        print(f"(design_version={DESIGN_VERSION_CONTENT} 레코드 없음: {out_path})")
        return

    def paired(rows, metric):
        d = []
        for r in rows:
            c, s = r[metric]["camel"], r[metric]["snake"]
            if c is not None and s is not None:
                d.append(s - c)                       # snake(위반) − camel(준수)
        return d

    def stat(xs):
        if not xs:
            return None
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        se = (var / len(xs)) ** 0.5
        return m, se, len(xs)

    print(f"\n=== 2단계 내용-분리 측정 [{DESIGN_VERSION_CONTENT}] "
          f"/ 같은 위치·같은 길이, 이름만 snake↔camel ===")
    print("naming step에서 target 이름 토큰에 준 attention. Δ = snake(위반) − camel(준수).\n")
    for metric, label in [("attn_name", "raw attention"),
                          ("vnorm_name", "‖α·v‖ (Kobayashi)")]:
        d = paired(rows, metric)
        st = stat(d)
        if st is None:
            print(f"  {label:<20}: (데이터 없음)")
            continue
        m, se, n = st
        cam = [r[metric]["camel"] for r in rows if r[metric]["camel"] is not None]
        sna = [r[metric]["snake"] for r in rows if r[metric]["snake"] is not None]
        z = m / se if se > 0 else float("inf")
        print(f"  [{label}]  n={n} pair·seed")
        print(f"     camel(준수) 평균 {sum(cam)/len(cam):.4f} | "
              f"snake(위반) 평균 {sum(sna)/len(sna):.4f}")
        print(f"     Δ(snake−camel) = {m:+.4f} ± {se:.4f} (SE),  z≈{z:+.1f}")
        print(f"     → {'위반 이름을 더 참조(양)' if m > 0 else '위반 이름을 덜/동일 참조'} "
              f"{'*유의 가능*' if abs(z) >= 2 else '(약함)'}\n")
    print("  ※ 길이·위치 고정 → 순수 내용(snake vs camel) 효과. 관측(--observe)의 구간 교란 제거.")
    print("  ※ 여전히 상관·탐색적(H5/H6 주가설 아님). attention 크기 ≠ 인과(계획서 3.7).")


def run_content(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.summary_only:
        print_content_summary(args.out)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = args.model or OBSERVE_MODEL
    if not torch.cuda.is_available():
        sys.exit("CUDA 없음 — Colab을 GPU 런타임(T4)으로 바꿔라.")
    if "int8" in model_id.lower() or "4bit" in model_id.lower():
        sys.exit("양자화 모델 금지(CLAUDE.md).")

    register_attention()
    pairs = _load_pairs(single_token_only=not args.all_pairs)[: args.n_pairs]
    seeds = [args.base_seed + k for k in range(args.n_seeds)]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add((r["design_version"], r["model"],
                                  r["pair_idx"], r["seed"]))
                    except (json.JSONDecodeError, KeyError):
                        continue
    pending = [(pi, s) for pi in range(len(pairs)) for s in seeds
               if (DESIGN_VERSION_CONTENT, model_id, pi, s) not in done]
    print(f"pair {len(pairs)}개(single-token={not args.all_pairs}) × seed {len(seeds)} "
          f"= {len(pairs)*len(seeds)} 세션, 남은 {len(pending)}")
    if not pending:
        print_content_summary(args.out)
        return

    print(f"[{model_id}] 로드 중 (attn_implementation={ATTN_NAME}, fp16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        attn_implementation=ATTN_NAME).eval()

    n_ok, skip = 0, {}
    for pi, s in pending:
        rec, why = content_session(model, tokenizer, torch, pairs[pi], pi, s, args)
        if rec is None:
            skip[why] = skip.get(why, 0) + 1          # 정렬 실패는 조용히 버리지 않고 집계
            continue
        append_jsonl(args.out, rec)
        n_ok += 1
    print(f"기록 {n_ok}세션. 정렬로 폐기: {skip or '없음'} "
          f"(len_mismatch/align_fail = 문맥에서 토큰이 안 맞은 pair)")
    print_content_summary(args.out)


# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="2단계 attention 관측 (NIAR)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true",
                      help="512토큰 eager 대조 검증 (관측 전 필수 게이트)")
    mode.add_argument("--observe", action="store_true", help="NIAR 관측 실행")
    mode.add_argument("--content", action="store_true",
                      help="내용-분리 측정: matched pair로 위치·길이 고정, 이름만 snake↔camel")

    ap.add_argument("--model", default=None, help="기본: 관측·내용=3B, 검증=1.5B")
    # 관측
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 3, 2, 1, 0])
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=1000, help="exp1과 seed 공유")
    ap.add_argument("--gen-tokens", type=int, default=48,
                    help="관측용 생성 길이(함수 머리만 보면 됨)")
    ap.add_argument("--do-sample", action="store_true",
                    help="기본은 greedy(관측 재현성). 켜면 exp1과 동일 샘플링")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--value-norm", action="store_true", default=True,
                    help="‖α·v‖ 보완 지표도 캡처(Kobayashi 2020)")
    ap.add_argument("--out", default=None,
                    help="기본: 관측=stage2_niar.jsonl, 내용=stage2_content.jsonl")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--layer-summary", action="store_true",
                    help="층별 진단(조건 hi−lo Δ). 전체 평균이 평탄할 때 어느 층에서 갈리는지")
    ap.add_argument("--top-layers", type=int, default=8,
                    help="--layer-summary에서 |Δ| 상위 몇 개 층을 보일지")
    # 내용-분리 측정
    ap.add_argument("--n-pairs", type=int, default=80, help="matched pair 개수(앞에서부터)")
    ap.add_argument("--n-filler", type=int, default=8, help="target 주변 filler 함수 수")
    ap.add_argument("--all-pairs", action="store_true",
                    help="단일 토큰 분기 pair만이 아니라 전체 matched pair 사용")
    # 검증
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float32",
                    help="검증 정밀도. fp32면 tight tolerance")
    ap.add_argument("--tol", type=float, default=1e-3, help="검증 허용 오차")
    ap.add_argument("--max-prompt-tokens", type=int, default=512)

    args = ap.parse_args()
    if args.out is None:
        args.out = DEFAULT_CONTENT_OUT if args.content else DEFAULT_OUT
    if args.validate:
        run_validate(args)
    elif args.content:
        run_content(args)
    else:
        run_observe(args)


if __name__ == "__main__":
    main()
