"""
3단계 행동 실험 — 개입을 켠 채 **실제 생성** (준수율 · 후속 위반)

teacher-forcing 준수 선호 점수(`stage3_intervention.py`)는 '확률 선호'를 잰다.
이 스크립트는 같은 개입을 **켠 채 실제로 함수를 생성**시켜, 내부 선호가 아니라
**실제 행동**이 바뀌는지 본다.

  (1) `--generate`  개입 아래 새 함수 이름을 생성 → **준수율**(camel 비율).
                    비교: 손상 vs L25 이식 vs 준수 baseline vs 무작위 donor(camel/snake) vs P2.
  (2) `--gen-chain` 개입 유지하며 함수 N개 **연쇄 생성** → **후속 위반 수**.
                    비교: 손상 vs L25 이식 vs L25 self(no-op) vs 준수 baseline.
  (3) `--calibrate` **개입 없이** base 조건(damaged/compliant)만 생성 → **행동 작동점** 탐색.
                    손상 준수율이 20~80%면 비포화. 순환 논증 방지: 이 모드는 p1a를 절대 안 본다.

v2 변경(2026-08-04, greedy 천장 대응):
  · **표집 생성**(`--sample` temp 0.7 top-p 0.95, 이름쌍당 `--n-sample-seeds`개 draw)으로
    준수율을 rate화. greedy(argmax)는 선호 마진이 커도 출력이 안 뒤집혀 천장에 걸렸다.
  · **엄격 비준수 지표** = 1(parse 실패 OR camel 아님). parse 실패가 '위반 0'으로 계산돼
    개입 효과처럼 보이던 문제를 막는다. camel/snake/기타/parse실패 분포를 함께 보고.
  · v1(greedy) 레코드와 안 섞이게 design_version=stage3_behavior_v2.

개입·donor·정렬은 `stage3_intervention` 하네스를 그대로 재사용(같은 홀드아웃·다중 토글).
생성 함수 이름 판정은 `step1_baseline`(ast 이름 추출 + case 분류).
결과: `results/stage3_behavior_v2.jsonl` (사전 등록 보존 규칙 5 — 새 파일).

usage:
  # (0) 가장 싼 작동점 탐색: dev 10쌍, 표집 20 draw, 개입 없이 base만
  python src/stage3_behavior.py --calibrate --sample --max-pairs 10 --n-sample-seeds 20
  # (1) 작동점 확정 후 홀드아웃 개입 (표집)
  python src/stage3_behavior.py --generate  --sample --pair-start 10 --n-seeds 3 --n-sample-seeds 20
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

DESIGN_VERSION = "stage3_behavior_v2"
DEFAULT_OUT = os.path.join(RESULTS_DIR, "stage3_behavior_v2.jsonl")
GEN_MAX_NEW = 48
CHAIN_N = 3
# 표집 기본값(연구계획서 5.3 디코딩과 일치).
SAMPLE_TEMP = 0.7
SAMPLE_TOPP = 0.95
N_SAMPLE_SEEDS = 20
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

def generate_draws(model, tok, torch, ids, max_new, sample, temperature, top_p,
                   n_draws, seed):
    """ids에서 이어서 n_draws개 생성. sample=False면 greedy 1개(n_draws 무시).
    표집은 seed로 재현 가능하게 고정. 반환: 생성 텍스트 리스트."""
    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if sample:
        torch.manual_seed(seed)
        kw.update(do_sample=True, temperature=temperature, top_p=top_p,
                  num_return_sequences=n_draws)
    else:
        kw.update(do_sample=False, num_return_sequences=1)   # greedy: 1개만 의미 있음
    with torch.no_grad():
        out = model.generate(inp, **kw)
    cut = len(ids)
    return [tok.decode(o[cut:].tolist(), skip_special_tokens=True) for o in out]


def judge(gen_text):
    """생성 이어붙인 함수 이름을 뽑아 case 분류. 프롬프트가 '```python\\ndef'로 끝나므로
    그 뒤 생성분 앞에 다시 붙여 파싱한다. 반환: (name, case, parse_ok)."""
    full = "```python\ndef" + gen_text
    name, ok = extract_name("python", full)
    return name, classify_case(name), ok


def _case_counts(gen_texts):
    """생성 텍스트 리스트 → case 분포 + 준수/엄격 비준수 rate.
    엄격 비준수 = parse 실패 OR camel 아님(= 1 - camel rate). parse 실패가 '위반 0'으로
    새는 걸 막는다."""
    c = {"camel": 0, "snake": 0, "pascal": 0, "other": 0, "parse_fail": 0}
    names = []
    for g in gen_texts:
        name, case, ok = judge(g)
        names.append(name)
        if name is None or case is None:
            c["parse_fail"] += 1
        elif case in c:
            c[case] += 1
        else:
            c["other"] += 1
    n = len(gen_texts)
    c["n_draws"] = n
    c["compliant_rate"] = (c["camel"] / n) if n else 0.0
    c["strict_noncompliant_rate"] = (1.0 - c["compliant_rate"]) if n else 0.0
    c["names"] = names[:8]
    return c


# ─────────────────────────────────────────────────────────────
# 조건 → 개입 설정
# ─────────────────────────────────────────────────────────────

def _base_conditions():
    """개입 없는 base 조건. 작동점 보정(--calibrate)은 이것만 쓴다(순환 논증 차단)."""
    return [
        ("damaged", "base", None),      # 위반 context, 개입 X (하한)
        ("compliant", "base", None),    # 준수 context, 개입 X (상한)
    ]


def _gen_conditions(main):
    """실험 (1) 조건. (라벨, kind, params). kind 'base'는 개입 없음."""
    return _base_conditions() + [
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
    print(f"[behavior v2] n_ctx={args.n_ctx}, pair_start={args.pair_start}, "
          f"sample={args.sample}(temp={args.temperature},top_p={args.top_p},"
          f"draws={args.n_sample_seeds}) (파일럿 {PILOT_PAIRS}쌍 제외 홀드아웃은 --pair-start {PILOT_PAIRS})")
    b_pairs = b_pairs[args.pair_start:]
    if args.max_pairs:
        b_pairs = b_pairs[:args.max_pairs]
    return model, tok, torch, pairs, decision, b_pairs, companions


def _draw_seed(pi, seed):
    """표집 재현용 seed(이름쌍·context seed로 결정)."""
    return 100003 * (pi + 1) + 97 * (seed + 1)


# ─────────────────────────────────────────────────────────────
# 실험 (0) 작동점 보정 · (1) 생성 준수율  — 공통 러너
# ─────────────────────────────────────────────────────────────

def _run_gen_like(args, mode, conds):
    """base/개입 조건들을 생성시켜 case 분포를 기록. mode='calib'(base만)/'gen'(전체)."""
    model, tok, torch, pairs, decision, b_pairs, companions = _prep(args)
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    n_draws = args.n_sample_seeds if args.sample else 1

    for li, ctx in enumerate(b_pairs):
        pi = args.pair_start + li
        for seed in seeds:
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        companions, pairs=pairs)
            if sess is None:
                append_jsonl(args.out, _skip(mode, "prep", pi, seed, ctx, args.model, why))
                continue
            for label, kind, params in conds:
                key = (mode, label, pi, seed, args.model)
                if key in done:
                    continue
                if not _set_iv(torch, sess, kind, params):
                    append_jsonl(args.out, _skip(mode, label, pi, seed, ctx,
                                                 args.model, "no_rand_donor"))
                    continue
                ids = sess["ids_comp"] if label == "compliant" else sess["ids_viol"]
                draws = generate_draws(model, tok, torch, ids, args.gen_max_new,
                                       args.sample, args.temperature, args.top_p,
                                       n_draws, _draw_seed(pi, seed))
                iv_off()
                cc = _case_counts(draws)
                append_jsonl(args.out, {
                    "design_version": DESIGN_VERSION, "model": args.model,
                    "mode": mode, "config": label, "pair_idx": pi, "seed": seed,
                    "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                    "sample": args.sample, "temperature": args.temperature, "top_p": args.top_p,
                    "n_draws": cc["n_draws"], "n_camel": cc["camel"], "n_snake": cc["snake"],
                    "n_pascal": cc["pascal"], "n_other": cc["other"], "n_parse_fail": cc["parse_fail"],
                    "compliant_rate": cc["compliant_rate"],
                    "strict_noncompliant_rate": cc["strict_noncompliant_rate"],
                    "gen_names": cc["names"],
                })
            print(f"  {mode} pair{pi} seed{seed}: {len(conds)} conds ({n_draws} draws)")
    print_summary(args.out)


def run_calibrate(args):
    """작동점 보정 — 개입 없이 base(damaged/compliant)만. p1a를 절대 안 본다(순환 논증 차단)."""
    if not args.sample:
        print("[경고] --calibrate는 표집(--sample)과 함께 쓰는 게 목적(greedy는 천장). 그래도 진행.")
    print("[calibrate] 개입 없음. 손상 준수율 20~80% + parse 성공률 높음인 작동점을 찾는다.")
    _run_gen_like(args, "calib", _base_conditions())


def run_generate(args):
    _run_gen_like(args, "gen", _gen_conditions(args.main_layer))


# ─────────────────────────────────────────────────────────────
# 실험 (2) — 개입 유지 연쇄 생성, 후속 위반 수
# ─────────────────────────────────────────────────────────────

def _code_pos_now(tok, prompt_text, prefix_str, prefix_len):
    enc = tok(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    cp = prompt_text.find(prefix_str)
    if cp == -1:
        return None
    pos = _char_span_to_tokens(enc["offset_mapping"], cp, cp + len(prefix_str))
    return [p for p in pos if SINK <= p < prefix_len]


def run_chain_one(model, tok, torch, sess, kind, params, n, max_new):
    """한 조건에서 함수 n개 연쇄 생성(greedy). iv는 원래 코드 구간에 유지.
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
        if kind == "p1a":
            now = _code_pos_now(tok, prompt_text, prefix_str, len(ids) - 1)
            if now != orig_code:
                return None
            _apply_config(torch, sess, "p1a", params)
        else:
            iv_off()
        gen = generate_draws(model, tok, torch, ids, max_new, False, 1.0, 1.0, 1, 0)[0]
        iv_off()
        name, case, ok = judge(gen)
        steps.append({"step": step + 1, "name": name, "case": case,
                      "violation": (case == "snake"),
                      "strict_noncompliant": (case != "camel"),   # parse 실패 포함
                      "parse_ok": ok})
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
        (f"noop_L{main}", "p1a", dict(pos="code", donor="self", groups=None, layers=[main])),
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
                nstrict = sum(1 for s in steps if s["strict_noncompliant"])
                nfail = sum(1 for s in steps if not s["parse_ok"])
                append_jsonl(args.out, {
                    "design_version": DESIGN_VERSION, "model": args.model,
                    "mode": "chain", "config": label, "pair_idx": pi, "seed": seed,
                    "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                    "n_steps": len(steps), "violation_count": nviol,
                    "strict_noncompliant_count": nstrict, "parse_fail_count": nfail,
                    "steps": steps,
                })
            print(f"  chain pair{pi} seed{seed}: {len(conds)} conds")
    print_summary(args.out)


def _skip(mode, config, pi, seed, ctx, model, why):
    return {"design_version": DESIGN_VERSION, "model": model, "mode": mode,
            "config": config, "pair_idx": pi, "seed": seed, "skipped": why,
            "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"]}


# ─────────────────────────────────────────────────────────────
# 집계 — 준수율 / 엄격 비준수 / 후속 위반 수 (two-way bootstrap CI)
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
        v = r[value]
        v = (1.0 if v else 0.0) if isinstance(v, bool) else float(v)
        g[r["config"]].append({"ctx_camel": r["ctx_camel"], "seed": r["seed"], "v": v})
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


def _dist_line(rows_for_cfg):
    """조건별 case 분포 합(camel/snake/pascal/other/parse실패)."""
    agg = {"n_camel": 0, "n_snake": 0, "n_pascal": 0, "n_other": 0, "n_parse_fail": 0, "n_draws": 0}
    for r in rows_for_cfg:
        for k in agg:
            agg[k] += int(r.get(k, 0))
    return agg


def _gen_table(rows, title):
    print(f"\n=== {title} [{DESIGN_VERSION}] === n={len(rows)} 세션")
    from collections import defaultdict
    byc = defaultdict(list)
    for r in rows:
        byc[r["config"]].append(r)
    g_comp = _by_config(rows, "compliant_rate")
    g_strict = _by_config(rows, "strict_noncompliant_rate")
    print(f"{'조건':20s} {'준수율':>7s} {'95%CI':>18s} {'엄격비준수':>9s}  "
          f"{'camel/snake/pas/oth/fail':>26s}")
    for cfg in sorted(g_comp):
        m, boot = _twoway_boot(g_comp[cfg], "v")
        lo, hi = _ci(boot)
        ci = f"[{lo:.3f},{hi:.3f}]" if lo is not None else ""
        ms, _ = _twoway_boot(g_strict[cfg], "v")
        d = _dist_line(byc[cfg])
        dist = f"{d['n_camel']}/{d['n_snake']}/{d['n_pascal']}/{d['n_other']}/{d['n_parse_fail']}"
        print(f"{cfg:20s} {m:7.3f} {ci:>18s} {ms:9.3f}  {dist:>26s}")
    return g_comp


def print_summary(out_path, n_boot=2000):
    main = MAIN_LAYER
    any_out = False

    # ── 작동점 보정 ──
    crows = _rows(out_path, "calib")
    if crows:
        any_out = True
        g = _gen_table(crows, "(0) 작동점 보정 — 개입 없음")
        dmg = g.get("damaged")
        if dmg:
            m, _ = _twoway_boot(dmg, "v")
            verdict = ("비포화 ✅ (20~80%)" if 0.20 <= m <= 0.80
                       else "천장/바닥 ❌ — 표집·prefix12·1단계작업·지침거리로 조정 필요")
            print(f"  → 손상 준수율 {m:.3f} : {verdict}")

    # ── (1) 개입 아래 생성 준수율 ──
    grows = _rows(out_path, "gen")
    if grows:
        any_out = True
        _gen_table(grows, "(1) 개입 아래 생성 준수율")
        for lab, a_c, b_c in ((f"L{main} − 손상 (주 검정)", f"p1a_L{main}", "damaged"),
                              ("무작위 camel − 무작위 snake (스타일)",
                               f"randcamel_L{main}", f"randsnake_L{main}"),
                              (f"L{main} − L{main} self(no-op)", f"p1a_L{main}", f"noop_L{main}"),
                              ("P2 λ2 − 손상", "p2_lam2", "damaged"),
                              ("P2 λ0.5 − 손상", "p2_lam0.5", "damaged")):
            rr = _cells_paired(grows, "compliant_rate", a_c, b_c)
            if rr:
                m, boot = _twoway_boot(rr, "c", n_boot)
                lo, hi = _ci(boot)
                sig = "0 배제 ✅" if (lo is not None and (lo > 0 or hi < 0)) else "0 포함 ❌"
                print(f"  Δ {lab:32s} = {m:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")

    # ── (2) 후속 위반 수 ──
    chrows = _rows(out_path, "chain")
    if chrows:
        any_out = True
        n_steps = chrows[0].get("n_steps", CHAIN_N)
        print(f"\n=== (2) 개입 유지 연쇄({n_steps}함수) 후속 위반 [{DESIGN_VERSION}] === n={len(chrows)}")
        from collections import defaultdict
        byc = defaultdict(list)
        for r in chrows:
            byc[r["config"]].append(r)
        gv = _by_config(chrows, "violation_count")
        gs = _by_config(chrows, "strict_noncompliant_count")
        print(f"{'조건':20s} {'snake위반':>8s} {'엄격비준수':>9s} {'parse실패':>8s}  n")
        for cfg in sorted(gv):
            mv, _ = _twoway_boot(gv[cfg], "v")
            ms, _ = _twoway_boot(gs[cfg], "v")
            nf = sum(int(r.get("parse_fail_count", 0)) for r in byc[cfg])
            print(f"{cfg:20s} {mv:8.3f} {ms:9.3f} {nf:8d}  {len(gv[cfg])}")
        for lab, val in (("snake위반", "violation_count"), ("엄격비준수", "strict_noncompliant_count")):
            rr = _cells_paired(chrows, val, f"p1a_L{main}", "damaged")
            if rr:
                m, boot = _twoway_boot(rr, "c", n_boot)
                lo, hi = _ci(boot)
                sig = "0 배제(감소) ✅" if (hi is not None and hi < 0) else "0 포함 ❌"
                print(f"  Δ L{main} − 손상 [{lab}] = {m:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")

    if not any_out:
        print(f"(레코드 없음: {out_path})")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="3단계 행동 실험 v2 (표집 생성·작동점 보정·후속 위반)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model", default=RUN_MODEL)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--n-seeds", type=int, default=3, help="context seed 수(대응 설계)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=8, help="prefix 함수 수(1단계 재현은 12)")
    ap.add_argument("--pair-start", type=int, default=0,
                    help=f"확증 홀드아웃 → --pair-start {PILOT_PAIRS}; 작동점 보정은 dev(0)")
    ap.add_argument("--max-pairs", type=int, default=0, help="파일럿용 제한(0=전체)")
    ap.add_argument("--main-layer", type=int, default=MAIN_LAYER)
    ap.add_argument("--aux-layers", type=str, default=",".join(map(str, AUX_LAYERS)))
    ap.add_argument("--gen-max-new", type=int, default=GEN_MAX_NEW)
    ap.add_argument("--chain-n", type=int, default=CHAIN_N)
    # 표집
    ap.add_argument("--sample", action="store_true", help="표집 생성(그리디 천장 대응)")
    ap.add_argument("--temperature", type=float, default=SAMPLE_TEMP)
    ap.add_argument("--top-p", dest="top_p", type=float, default=SAMPLE_TOPP)
    ap.add_argument("--n-sample-seeds", type=int, default=N_SAMPLE_SEEDS,
                    help="이름쌍·seed당 표집 draw 수(준수율 rate화)")
    # 모드
    ap.add_argument("--calibrate", action="store_true", help="(0) 개입 없이 base만 — 작동점 탐색")
    ap.add_argument("--generate", action="store_true", help="(1) 개입 아래 생성 준수율")
    ap.add_argument("--gen-chain", action="store_true", help="(2) 연쇄 후속 위반")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    args.aux_layers = [int(x) for x in str(args.aux_layers).split(",") if x.strip()]

    if args.calibrate:
        run_calibrate(args)
    elif args.generate:
        run_generate(args)
    elif args.gen_chain:
        run_gen_chain(args)
    elif args.summary_only:
        print_summary(args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
