"""
3단계 행동 실험 — 개입을 켠 채 **실제 생성** (준수율 · 후속 위반)

teacher-forcing 준수 선호 점수(`stage3_intervention.py`)는 '확률 선호'를 잰다.
이 스크립트는 같은 개입을 **켠 채 실제로 함수를 생성**시켜, 내부 선호가 아니라
**실제 행동**이 바뀌는지 본다. 두 실험:

  (1) `--generate`  개입 아래 새 함수 이름을 생성 → **준수율**(camel 비율).
                    비교: 손상 vs L25 이식 vs 준수 baseline vs 무작위 donor(camel/snake) vs P2(λ↑/↓).
  (2) `--gen-chain` 개입 유지하며 함수 N개 **연쇄 생성** → **후속 위반 수**(자기증폭 연쇄를 끊나).
                    비교: 손상 vs L25 이식 vs 준수 baseline.

개입·donor·정렬은 `stage3_intervention` 하네스를 그대로 재사용(같은 홀드아웃·다중 토글).
생성 함수 이름 판정은 `step1_baseline`(ast 이름 추출 + case 분류).
결과: `results/stage3_behavior.jsonl` (사전 등록 보존 규칙 5 — 새 파일). greedy 생성.

usage:
  python src/stage3_behavior.py --generate  --pair-start 10 --n-seeds 3
  python src/stage3_behavior.py --gen-chain --pair-start 10 --n-seeds 3
  python src/stage3_behavior.py --summary-only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_baseline import classify_case, extract_name          # noqa: E402
from stage3_intervention import (                                # noqa: E402
    RESULTS_DIR, RUN_MODEL, VALIDATE_MODEL, MAIN_LAYER, AUX_LAYERS, PILOT_PAIRS,
    SINK, _char_span_to_tokens, _load_model, _load_pairs, choose_decision_pair,
    split_context_pairs, companion_pairs, prepare_session, _apply_config, iv_off,
    append_jsonl, _twoway_boot, _ci,
)

DESIGN_VERSION = "stage3_behavior_v1"
DEFAULT_OUT = os.path.join(RESULTS_DIR, "stage3_behavior.jsonl")
GEN_MAX_NEW = 48
CHAIN_N = 3
# 연쇄 각 단계의 작업 지시(1단계와 같은 계열 — 이름 스타일이 자유 선택인 다-단어 작업).
CHAIN_TASKS = [
    "formats a value for display",
    "validates an email address",
    "computes a total from a list of items",
]

_RAND_SLOT = {"rand": "kv_rand", "rand_snake": "kv_rand_snake"}


# ─────────────────────────────────────────────────────────────
# 생성 + 판정
# ─────────────────────────────────────────────────────────────

def generate_once(model, tok, torch, ids, max_new):
    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(ids):].tolist(), skip_special_tokens=True)


def judge(gen_text):
    """생성 이어붙인 함수 이름을 뽑아 case 분류. 프롬프트가 '```python\\ndef'로 끝나므로
    그 뒤 생성분 앞에 다시 붙여 파싱한다. 반환: (name, case, parse_ok)."""
    full = "```python\ndef" + gen_text
    name, ok = extract_name("python", full)
    return name, classify_case(name), ok


# ─────────────────────────────────────────────────────────────
# 조건 → 개입 설정
# ─────────────────────────────────────────────────────────────

def _gen_conditions(main):
    """실험 (1) 조건. (라벨, kind, params). kind 'base'는 개입 없음."""
    return [
        ("damaged", "base", None),                                    # 위반 context, 개입 X (하한)
        ("compliant", "base", None),                                  # 준수 context, 개입 X (상한)
        (f"p1a_L{main}", "p1a", dict(pos="code", donor="comp", groups=None, layers=[main])),
        ("p1a_full", "p1a", dict(pos="code", donor="comp", groups=None, layers=None)),
        (f"noop_L{main}", "p1a", dict(pos="code", donor="self", groups=None, layers=[main])),
        (f"randcamel_L{main}", "p1a", dict(pos="code", donor="rand", groups=None, layers=[main])),
        (f"randsnake_L{main}", "p1a", dict(pos="code", donor="rand_snake", groups=None, layers=[main])),
        ("p2_lam2", "p2", dict(lam=2.0, heads=None, layers=None, conserve=True)),
        ("p2_lam0.5", "p2", dict(lam=0.5, heads=None, layers=None, conserve=True)),
    ]


def _set_iv(torch, sess, kind, params):
    """개입 세팅. 무작위 donor 없으면 False(생략)."""
    if kind == "base":
        iv_off()
        return True
    dsrc = _RAND_SLOT.get(params.get("donor"))
    if dsrc is not None and sess.get(dsrc) is None:
        return False
    _apply_config(torch, sess, kind, params)
    return True


# ─────────────────────────────────────────────────────────────
# 재개 / 로드
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
            done.add((r["mode"], r["config"], r["pair_idx"], r["seed"], r["model"]))
    return done


def _prep(args):
    model, tok, torch = _load_model(args.model, args.dtype)
    pairs = _load_pairs()
    decision = choose_decision_pair(pairs)
    _, b_pairs = split_context_pairs(pairs, decision)
    companions = companion_pairs(pairs, decision, args.n_ctx)
    print(f"[behavior] n_ctx={args.n_ctx}, pair_start={args.pair_start} "
          f"(v3 홀드아웃; 파일럿 {PILOT_PAIRS}쌍 제외는 --pair-start {PILOT_PAIRS})")
    b_pairs = b_pairs[args.pair_start:]
    if args.max_pairs:
        b_pairs = b_pairs[:args.max_pairs]
    return model, tok, torch, pairs, decision, b_pairs, companions


# ─────────────────────────────────────────────────────────────
# 실험 (1) — 개입 아래 생성 준수율
# ─────────────────────────────────────────────────────────────

def run_generate(args):
    model, tok, torch, pairs, decision, b_pairs, companions = _prep(args)
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    conds = _gen_conditions(args.main_layer)

    for li, ctx in enumerate(b_pairs):
        pi = args.pair_start + li
        for seed in seeds:
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        companions, pairs=pairs)
            if sess is None:
                append_jsonl(args.out, _skip("gen", "prep", pi, seed, ctx, args.model, why))
                continue
            for label, kind, params in conds:
                key = ("gen", label, pi, seed, args.model)
                if key in done:
                    continue
                if not _set_iv(torch, sess, kind, params):
                    append_jsonl(args.out, _skip("gen", label, pi, seed, ctx,
                                                 args.model, "no_rand_donor"))
                    continue
                ids = sess["ids_comp"] if label == "compliant" else sess["ids_viol"]
                gen = generate_once(model, tok, torch, ids, args.gen_max_new)
                iv_off()
                name, case, ok = judge(gen)
                append_jsonl(args.out, {
                    "design_version": DESIGN_VERSION, "model": args.model,
                    "mode": "gen", "config": label, "pair_idx": pi, "seed": seed,
                    "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                    "gen_name": name, "case": case,
                    "compliant": (case == "camel"), "violation": (case == "snake"),
                    "parse_ok": ok,
                })
            print(f"  gen pair{pi} seed{seed}: {len(conds)} conds")
    print_summary(args.out)


# ─────────────────────────────────────────────────────────────
# 실험 (3) — 개입 유지 연쇄 생성, 후속 위반 수
# ─────────────────────────────────────────────────────────────

def _code_pos_now(tok, prompt_text, prefix_str, prefix_len):
    enc = tok(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    cp = prompt_text.find(prefix_str)
    if cp == -1:
        return None
    pos = _char_span_to_tokens(enc["offset_mapping"], cp, cp + len(prefix_str))
    return [p for p in pos if SINK <= p < prefix_len]


def run_chain_one(model, tok, torch, sess, kind, params, n, max_new):
    """한 조건에서 함수 n개 연쇄 생성. iv는 원래 코드 구간에 유지.
    개입 세션에서 연쇄 중 코드 구간 정렬이 깨지면 None(폐기)."""
    is_comp = (kind == "base_comp")
    msgs = [dict(m) for m in (sess["messages_comp"] if is_comp else sess["messages_viol"])]
    prefix_str = sess["prefix_comp"] if is_comp else sess["prefix_viol"]
    orig_code = sess["pos"]["code"].tolist()
    steps = []
    for step in range(n):
        if step > 0:
            msgs.append({"role": "user",
                         "content": f"Add a function that {CHAIN_TASKS[step % len(CHAIN_TASKS)]}."})
        prompt_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True) + "```python\ndef"
        ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        # 개입 조건: 코드 구간이 원래와 같은 위치여야 donor가 유효.
        if kind == "p1a":
            now = _code_pos_now(tok, prompt_text, prefix_str, len(ids) - 1)
            if now != orig_code:
                return None
            _apply_config(torch, sess, "p1a", params)
        else:
            iv_off()
        gen = generate_once(model, tok, torch, ids, max_new)
        iv_off()
        name, case, ok = judge(gen)
        steps.append({"step": step + 1, "name": name, "case": case,
                      "violation": (case == "snake"), "parse_ok": ok})
        msgs.append({"role": "assistant", "content": "```python\ndef" + gen})
    return steps


def run_gen_chain(args):
    model, tok, torch, pairs, decision, b_pairs, companions = _prep(args)
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    main = args.main_layer
    conds = [
        ("damaged", "base", None),
        ("compliant", "base_comp", None),
        (f"p1a_L{main}", "p1a", dict(pos="code", donor="comp", groups=None, layers=[main])),
    ]

    for li, ctx in enumerate(b_pairs):
        pi = args.pair_start + li
        for seed in seeds:
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        companions, pairs=pairs)
            if sess is None:
                append_jsonl(args.out, _skip("chain", "prep", pi, seed, ctx, args.model, why))
                continue
            for label, kind, params in conds:
                key = ("chain", label, pi, seed, args.model)
                if key in done:
                    continue
                steps = run_chain_one(model, tok, torch, sess, kind, params,
                                      args.chain_n, args.gen_max_new)
                iv_off()
                if steps is None:
                    append_jsonl(args.out, _skip("chain", label, pi, seed, ctx,
                                                 args.model, "chain_align"))
                    continue
                nviol = sum(1 for s in steps if s["violation"])
                append_jsonl(args.out, {
                    "design_version": DESIGN_VERSION, "model": args.model,
                    "mode": "chain", "config": label, "pair_idx": pi, "seed": seed,
                    "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                    "n_steps": len(steps), "violation_count": nviol, "steps": steps,
                })
            print(f"  chain pair{pi} seed{seed}: {len(conds)} conds")
    print_summary(args.out)


def _skip(mode, config, pi, seed, ctx, model, why):
    return {"design_version": DESIGN_VERSION, "model": model, "mode": mode,
            "config": config, "pair_idx": pi, "seed": seed, "skipped": why,
            "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"]}


# ─────────────────────────────────────────────────────────────
# 집계 — 준수율 / 후속 위반 수 (two-way bootstrap CI)
# ─────────────────────────────────────────────────────────────

def _rows(out_path, mode):
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (r.get("design_version") == DESIGN_VERSION
                        and r.get("mode") == mode and "skipped" not in r):
                    rows.append(r)
    return rows


def _by_config(rows, value):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r["config"]].append({"ctx_camel": r["ctx_camel"], "seed": r["seed"],
                               "v": (1.0 if r[value] else 0.0) if isinstance(r[value], bool)
                               else float(r[value])})
    return g


def _cells_paired(rows, value, a_cfg, b_cfg):
    """같은 (이름쌍, seed) 셀에서 a−b 대응 대비 rows."""
    from collections import defaultdict
    cell = defaultdict(dict)
    for r in rows:
        v = r[value]
        v = (1.0 if v else 0.0) if isinstance(v, bool) else float(v)
        cell[(r["ctx_camel"], r["seed"])][r["config"]] = v
    out = []
    for (p, s), d in cell.items():
        if a_cfg in d and b_cfg in d:
            out.append({"ctx_camel": p, "seed": s, "c": d[a_cfg] - d[b_cfg]})
    return out


def print_summary(out_path, n_boot=2000):
    main = MAIN_LAYER
    # ── 실험 (1) 준수율 ──
    grows = _rows(out_path, "gen")
    if grows:
        print(f"\n=== (1) 개입 아래 생성 준수율 [{DESIGN_VERSION}] === n={len(grows)}")
        g = _by_config(grows, "compliant")
        print(f"{'조건':22s} {'준수율':>8s} {'95%CI':>20s}  n")
        for cfg in sorted(g):
            m, boot = _twoway_boot(g[cfg], "v", n_boot)
            lo, hi = _ci(boot)
            ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else ""
            print(f"{cfg:22s} {m:8.3f} {ci:>20s}  {len(g[cfg])}")
        # 대응 대비: L25 − 손상 (행동이 바뀌나), rand camel − rand snake (스타일)
        for lab, a_c, b_c in ((f"L{main} − 손상", f"p1a_L{main}", "damaged"),
                              ("무작위 camel − 무작위 snake",
                               f"randcamel_L{main}", f"randsnake_L{main}"),
                              ("P2 λ2 − 손상", "p2_lam2", "damaged"),
                              ("P2 λ0.5 − 손상", "p2_lam0.5", "damaged")):
            rr = _cells_paired(grows, "compliant", a_c, b_c)
            if rr:
                m, boot = _twoway_boot(rr, "c", n_boot)
                lo, hi = _ci(boot)
                sig = "0 배제 ✅" if (lo is not None and (lo > 0 or hi < 0)) else "0 포함 ❌"
                print(f"  Δ {lab:28s} = {m:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")

    # ── 실험 (3) 후속 위반 수 ──
    crows = _rows(out_path, "chain")
    if crows:
        n_steps = crows[0].get("n_steps", CHAIN_N)
        print(f"\n=== (3) 개입 유지 연쇄({n_steps}함수) 후속 위반 수 [{DESIGN_VERSION}] === n={len(crows)}")
        g = _by_config(crows, "violation_count")
        print(f"{'조건':22s} {'평균위반수':>10s} {'95%CI':>20s}  n")
        for cfg in sorted(g):
            m, boot = _twoway_boot(g[cfg], "v", n_boot)
            lo, hi = _ci(boot)
            ci = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else ""
            print(f"{cfg:22s} {m:10.3f} {ci:>20s}  {len(g[cfg])}")
        rr = _cells_paired(crows, "violation_count", f"p1a_L{main}", "damaged")
        if rr:
            m, boot = _twoway_boot(rr, "c", n_boot)
            lo, hi = _ci(boot)
            sig = "0 배제(감소) ✅" if (hi is not None and hi < 0) else "0 포함 ❌"
            print(f"  Δ L{main} − 손상 위반수 = {m:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")

    if not grows and not crows:
        print(f"(레코드 없음: {out_path})")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="3단계 행동 실험 (생성 준수율·후속 위반)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model", default=RUN_MODEL)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=8)
    ap.add_argument("--pair-start", type=int, default=0,
                    help=f"확증 홀드아웃 → --pair-start {PILOT_PAIRS}")
    ap.add_argument("--max-pairs", type=int, default=0, help="파일럿용 제한(0=전체)")
    ap.add_argument("--main-layer", type=int, default=MAIN_LAYER)
    ap.add_argument("--aux-layers", type=str, default=",".join(map(str, AUX_LAYERS)))
    ap.add_argument("--gen-max-new", type=int, default=GEN_MAX_NEW)
    ap.add_argument("--chain-n", type=int, default=CHAIN_N)
    ap.add_argument("--generate", action="store_true", help="(1) 생성 준수율")
    ap.add_argument("--gen-chain", action="store_true", help="(3) 연쇄 후속 위반")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    args.aux_layers = [int(x) for x in str(args.aux_layers).split(",") if x.strip()]

    if args.generate:
        run_generate(args)
    elif args.gen_chain:
        run_gen_chain(args)
    elif args.summary_only:
        print_summary(args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
