"""
3단계 행동 실험 — 개입을 켠 채 **실제 생성** (준수율 · 후속 위반)

teacher-forcing 준수 선호 점수(`stage3_intervention.py`)는 '확률 선호'를 잰다.
이 스크립트는 같은 개입을 **켠 채 실제로 함수를 생성**시켜, 내부 선호가 아니라
**실제 행동**이 바뀌는지 본다.

  (0) `--calibrate` **개입 없이** base 조건(damaged/compliant)만 생성 → **행동 작동점** 탐색.
                    게이트: 손상 20~80% · (준수−손상)≥+20%p · 이름추출 성공률≥95%.
                    순환 논증 방지: 이 모드는 p1a를 절대 안 본다.
  (1) `--generate`  개입 아래 새 함수 이름을 생성 → **준수율**(camel 비율).
                    **calibration 게이트 PASS가 기록돼 있어야 실행**(아니면 중단; --force-ungated로 강제).
  (2) `--gen-chain` 개입 유지하며 함수 N개 **연쇄 생성** → **후속 위반 수**. (아직 greedy — 자기 출력
                    인과를 직접 보려면 별도 2×2 필요. 문서 '자기증폭 직접 연결' 참조.)

v2 설계 노트(2026-08-04):
  · **표집 생성**(temp 0.7 top-p 0.95)으로 준수율 rate화. **draw는 1개씩 반복 생성**(배치 생성 금지 —
    donor K/V가 batch 1이라 num_return_sequences>1이면 P1a에서 배치 크기 오류). 같은 draw_idx seed를
    전 조건이 공유 → 대응 설계 유지.
  · **엄격 비준수** = 1(이름 추출 실패 OR camel 아님). 이름 추출 실패가 '위반 0'으로 새는 걸 막는다.
    (주의: 이 실패는 AST parse 실패가 아니라 **identifier 추출 실패** — 이 연구의 결정 지점은 함수명.)
  · **재개 키·레코드에 실행 설정 포함**(n_ctx/표집/temp/top_p/draw수/프롬프트변형). 설정이 다른 실행을
    완료로 오인·혼합 집계하는 걸 막는다.
  · v1(greedy) 레코드와 안 섞이게 design_version=stage3_behavior_v2, 파일도 분리.

usage:
  # (0) 작동점 탐색 — dev 10쌍, 표집, 개입 없이 base만
  python src/stage3_behavior.py --calibrate --sample --max-pairs 10 --n-sample-seeds 20 --n-ctx 8
  # (1) 게이트 PASS 확인 후에만 — 홀드아웃 개입(표집)
  python src/stage3_behavior.py --generate  --sample --pair-start 10 --n-seeds 3 --n-sample-seeds 20
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
SAMPLE_TEMP = 0.7
SAMPLE_TOPP = 0.95
N_SAMPLE_SEEDS = 20
# 작동점 게이트 임계.
GATE_DMG_LO, GATE_DMG_HI = 0.20, 0.80
GATE_MIN_GAP = 0.20            # 준수 − 손상 ≥ +20%p
GATE_MIN_NAME_OK = 0.95       # 이름 추출 성공률
CHAIN_TASKS = [
    "formats a value for display",
    "validates an email address",
    "computes a total from a list of items",
]

_RAND_SLOT = {"rand": "kv_rand", "rand_snake": "kv_rand_snake"}


# ─────────────────────────────────────────────────────────────
# 실행 설정 태그 (재개 키·게이트 매칭)
# ─────────────────────────────────────────────────────────────

def _n_draws(args):
    return args.n_sample_seeds if args.sample else 1


def _run_tag(args):
    """재개 키·레코드용. draw 수까지 포함(다른 draw 수를 완료로 오인 안 하게)."""
    dec = "samp" if args.sample else "greedy"
    return f"nctx{args.n_ctx}|{dec}|t{args.temperature}|p{args.top_p}|d{_n_draws(args)}|{args.prompt_variant}"


def _gate_tag(args):
    """작동점 게이트 매칭용. draw 수는 제외(작동점은 프롬프트 설정으로 정의)."""
    dec = "samp" if args.sample else "greedy"
    return f"nctx{args.n_ctx}|{dec}|t{args.temperature}|p{args.top_p}|{args.prompt_variant}"


# ─────────────────────────────────────────────────────────────
# 생성 + 판정
# ─────────────────────────────────────────────────────────────

def generate_draws(model, tok, torch, ids, max_new, sample, temperature, top_p,
                   n_draws, base_seed):
    """ids에서 이어서 생성. **draw를 1개씩 반복**(배치 생성 금지 — donor K/V가 batch 1이라
    num_return_sequences>1이면 P1a index_copy_에서 배치 크기 오류). 표집은 base_seed+draw_idx로
    재현·전 조건 공유. sample=False면 1개(greedy). 반환: 생성 텍스트 리스트."""
    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
    cut = len(ids)
    n = n_draws if sample else 1
    outs = []
    for d in range(n):
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id,
                  num_return_sequences=1)
        if sample:
            torch.manual_seed(base_seed + d)
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(inp, **kw)
        outs.append(tok.decode(out[0][cut:].tolist(), skip_special_tokens=True))
    return outs


def judge(gen_text):
    """생성 이어붙인 함수 이름을 뽑아 case 분류. 프롬프트가 '```python\\ndef'로 끝나므로
    그 뒤 생성분 앞에 다시 붙여 파싱한다. 반환: (name, case, name_ok).
    name_ok = identifier 추출 성공 여부(AST 유효성 아님 — 결정 지점은 함수명)."""
    full = "```python\ndef" + gen_text
    name, _parse_ok = extract_name("python", full)
    return name, classify_case(name), (name is not None)


def _case_counts(gen_texts):
    """생성 텍스트 리스트 → case 분포 + 준수/엄격 비준수 rate.
    엄격 비준수 = 이름 추출 실패 OR camel 아님(= 1 - camel rate)."""
    c = {"camel": 0, "snake": 0, "pascal": 0, "other": 0, "name_fail": 0}
    names = []
    for g in gen_texts:
        name, case, name_ok = judge(g)
        names.append(name)
        if not name_ok or case is None:
            c["name_fail"] += 1
        elif case in c:
            c[case] += 1
        else:
            c["other"] += 1
    n = len(gen_texts)
    c["n_draws"] = n
    c["compliant_rate"] = (c["camel"] / n) if n else 0.0
    c["strict_noncompliant_rate"] = (1.0 - c["compliant_rate"]) if n else 0.0
    c["name_ok_rate"] = (1.0 - c["name_fail"] / n) if n else 0.0
    c["names"] = names[:8]
    return c


# ─────────────────────────────────────────────────────────────
# 조건 → 개입 설정
# ─────────────────────────────────────────────────────────────

def _base_conditions():
    return [
        ("damaged", "base", None),      # 위반 context, 개입 X (하한)
        ("compliant", "base", None),    # 준수 context, 개입 X (상한)
    ]


def _gen_conditions(main):
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
    """재개 키에 run_tag 포함 — 설정이 다른 실행을 완료로 오인 안 하게."""
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
            done.add((r["mode"], r["config"], r["pair_idx"], r["seed"], r["model"],
                      r.get("run_tag", "")))
    return done


def _prep(args):
    model, tok, torch = _load_model(args.model, args.dtype)
    pairs = _load_pairs()
    decision = choose_decision_pair(pairs)
    _, b_pairs = split_context_pairs(pairs, decision)
    companions = companion_pairs(pairs, decision, args.n_ctx)
    print(f"[behavior v2] run_tag={_run_tag(args)}  pair_start={args.pair_start}  "
          f"(홀드아웃은 --pair-start {PILOT_PAIRS})")
    if args.prompt_variant != "decision":
        print(f"[경고] prompt_variant={args.prompt_variant} 아직 미구현 — 현재는 'decision' 하네스만 실행됨.")
    b_pairs = b_pairs[args.pair_start:]
    if args.max_pairs:
        b_pairs = b_pairs[:args.max_pairs]
    return model, tok, torch, pairs, decision, b_pairs, companions


def _draw_seed(pi, seed):
    return 100003 * (pi + 1) + 97 * (seed + 1)


# ─────────────────────────────────────────────────────────────
# (0)/(1) 생성 러너 공통
# ─────────────────────────────────────────────────────────────

def _run_gen_like(args, mode, conds):
    model, tok, torch, pairs, decision, b_pairs, companions = _prep(args)
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    run_tag, gate_tag = _run_tag(args), _gate_tag(args)
    nd = _n_draws(args)

    for li, ctx in enumerate(b_pairs):
        pi = args.pair_start + li
        for seed in seeds:
            sess, why = prepare_session(model, tok, torch, ctx, decision, seed,
                                        companions, pairs=pairs)
            if sess is None:
                append_jsonl(args.out, _skip(mode, "prep", pi, seed, ctx, args.model, why, run_tag))
                continue
            for label, kind, params in conds:
                key = (mode, label, pi, seed, args.model, run_tag)
                if key in done:
                    continue
                if not _set_iv(torch, sess, kind, params):
                    append_jsonl(args.out, _skip(mode, label, pi, seed, ctx,
                                                 args.model, "no_rand_donor", run_tag))
                    continue
                ids = sess["ids_comp"] if label == "compliant" else sess["ids_viol"]
                draws = generate_draws(model, tok, torch, ids, args.gen_max_new,
                                       args.sample, args.temperature, args.top_p,
                                       nd, _draw_seed(pi, seed))
                iv_off()
                cc = _case_counts(draws)
                append_jsonl(args.out, {
                    "design_version": DESIGN_VERSION, "model": args.model,
                    "mode": mode, "config": label, "pair_idx": pi, "seed": seed,
                    "run_tag": run_tag, "gate_tag": gate_tag,
                    "n_ctx": args.n_ctx, "sample": args.sample,
                    "temperature": args.temperature, "top_p": args.top_p,
                    "prompt_variant": args.prompt_variant,
                    "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                    "n_draws": cc["n_draws"], "n_camel": cc["camel"], "n_snake": cc["snake"],
                    "n_pascal": cc["pascal"], "n_other": cc["other"], "n_name_fail": cc["name_fail"],
                    "compliant_rate": cc["compliant_rate"],
                    "strict_noncompliant_rate": cc["strict_noncompliant_rate"],
                    "name_ok_rate": cc["name_ok_rate"], "gen_names": cc["names"],
                })
            print(f"  {mode} pair{pi} seed{seed}: {len(conds)} conds ({nd} draws)")
    print_summary(args.out)


def run_calibrate(args):
    if not args.sample:
        print("[경고] --calibrate는 표집(--sample)과 함께 쓰는 게 목적(greedy는 천장). 그래도 진행.")
    print("[calibrate] 개입 없음(p1a 미실행). 게이트: 손상 "
          f"{GATE_DMG_LO:.0%}~{GATE_DMG_HI:.0%} · (준수−손상)≥{GATE_MIN_GAP:.0%}p · 이름추출≥{GATE_MIN_NAME_OK:.0%}.")
    _run_gen_like(args, "calib", _base_conditions())


def run_generate(args):
    """calibration 게이트 PASS가 기록돼 있어야 실행(--force-ungated로 강제)."""
    passed, msg = _check_gate(args.out, _gate_tag(args))
    if not passed and not args.force_ungated:
        print(f"[중단] 작동점 게이트 미통과/미기록 — 본 실험 안 돌림.\n  {msg}\n"
              f"  먼저 같은 설정으로 --calibrate 실행 후 PASS 확인. 강제하려면 --force-ungated.")
        return
    if not passed:
        print(f"[강제 실행] 게이트 미통과지만 --force-ungated: {msg}")
    else:
        print(f"[게이트 PASS] {msg}")
    _run_gen_like(args, "gen", _gen_conditions(args.main_layer))


# ─────────────────────────────────────────────────────────────
# 작동점 게이트 판정
# ─────────────────────────────────────────────────────────────

def _gate_stats(out_path, gate_tag):
    """같은 gate_tag의 calib 레코드에서 손상/준수 준수율·이름추출 성공률."""
    rows = [r for r in _rows(out_path, "calib") if r.get("gate_tag") == gate_tag]
    if not rows:
        return None
    from collections import defaultdict
    byc = defaultdict(list)
    for r in rows:
        byc[r["config"]].append(r)

    def _mean(cfg, key):
        xs = byc.get(cfg, [])
        return (sum(float(r[key]) for r in xs) / len(xs)) if xs else None

    dmg = _mean("damaged", "compliant_rate")
    comp = _mean("compliant", "compliant_rate")
    name_ok = None
    allrows = rows
    if allrows:
        name_ok = sum(float(r.get("name_ok_rate", 0.0)) for r in allrows) / len(allrows)
    return {"damaged": dmg, "compliant": comp, "name_ok": name_ok, "n": len(rows)}


def _check_gate(out_path, gate_tag):
    s = _gate_stats(out_path, gate_tag)
    if s is None or s["damaged"] is None or s["compliant"] is None:
        return False, f"calib 기록 없음(gate_tag={gate_tag})"
    dmg, comp, name_ok = s["damaged"], s["compliant"], s["name_ok"] or 0.0
    gap = comp - dmg
    ok = (GATE_DMG_LO <= dmg <= GATE_DMG_HI and gap >= GATE_MIN_GAP and name_ok >= GATE_MIN_NAME_OK)
    msg = (f"손상={dmg:.3f}(요구 {GATE_DMG_LO:.2f}~{GATE_DMG_HI:.2f}) · "
           f"준수−손상={gap:+.3f}(≥{GATE_MIN_GAP:.2f}) · 이름추출={name_ok:.3f}(≥{GATE_MIN_NAME_OK:.2f})")
    return ok, msg


# ─────────────────────────────────────────────────────────────
# (2) 연쇄 생성 (아직 greedy; 자기증폭 직접 검정은 별도 2×2)
# ─────────────────────────────────────────────────────────────

def _code_pos_now(tok, prompt_text, prefix_str, prefix_len):
    enc = tok(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    cp = prompt_text.find(prefix_str)
    if cp == -1:
        return None
    pos = _char_span_to_tokens(enc["offset_mapping"], cp, cp + len(prefix_str))
    return [p for p in pos if SINK <= p < prefix_len]


def run_chain_one(model, tok, torch, sess, kind, params, n, max_new):
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
        name, case, name_ok = judge(gen)
        steps.append({"step": step + 1, "name": name, "case": case,
                      "violation": (case == "snake"),
                      "strict_noncompliant": (case != "camel"),
                      "name_ok": name_ok})
        msgs.append({"role": "assistant", "content": "```python\ndef" + gen})
    return steps


def run_gen_chain(args):
    model, tok, torch, pairs, decision, b_pairs, companions = _prep(args)
    done = load_done(args.out)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    main = args.main_layer
    run_tag = _run_tag(args)
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
                append_jsonl(args.out, _skip("chain", "prep", pi, seed, ctx, args.model, why, run_tag))
                continue
            for label, kind, params in conds:
                key = ("chain", label, pi, seed, args.model, run_tag)
                if key in done:
                    continue
                steps = run_chain_one(model, tok, torch, sess, kind, params,
                                      args.chain_n, args.gen_max_new)
                iv_off()
                if steps is None:
                    append_jsonl(args.out, _skip("chain", label, pi, seed, ctx,
                                                 args.model, "chain_align", run_tag))
                    continue
                nviol = sum(1 for s in steps if s["violation"])
                nstrict = sum(1 for s in steps if s["strict_noncompliant"])
                nfail = sum(1 for s in steps if not s["name_ok"])
                append_jsonl(args.out, {
                    "design_version": DESIGN_VERSION, "model": args.model,
                    "mode": "chain", "config": label, "pair_idx": pi, "seed": seed,
                    "run_tag": run_tag, "n_ctx": args.n_ctx, "prompt_variant": args.prompt_variant,
                    "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"],
                    "n_steps": len(steps), "violation_count": nviol,
                    "strict_noncompliant_count": nstrict, "name_fail_count": nfail,
                    "steps": steps,
                })
            print(f"  chain pair{pi} seed{seed}: {len(conds)} conds")
    print_summary(args.out)


def _skip(mode, config, pi, seed, ctx, model, why, run_tag):
    return {"design_version": DESIGN_VERSION, "model": model, "mode": mode,
            "config": config, "pair_idx": pi, "seed": seed, "run_tag": run_tag,
            "skipped": why, "ctx_camel": ctx["camel"], "ctx_snake": ctx["snake"]}


# ─────────────────────────────────────────────────────────────
# 집계
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
    agg = {"n_camel": 0, "n_snake": 0, "n_pascal": 0, "n_other": 0, "n_name_fail": 0}
    for r in rows_for_cfg:
        for k in agg:
            agg[k] += int(r.get(k, 0))
    return agg


def _split_by_runtag(rows):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r.get("run_tag", "")].append(r)
    return g


def _gen_table(rows, title):
    print(f"\n=== {title} [{DESIGN_VERSION}] === n={len(rows)} 세션")
    from collections import defaultdict
    byc = defaultdict(list)
    for r in rows:
        byc[r["config"]].append(r)
    g_comp = _by_config(rows, "compliant_rate")
    g_strict = _by_config(rows, "strict_noncompliant_rate")
    print(f"{'조건':20s} {'준수율':>7s} {'95%CI':>18s} {'엄격비준수':>9s}  "
          f"{'camel/snake/pas/oth/이름실패':>28s}")
    for cfg in sorted(g_comp):
        m, boot = _twoway_boot(g_comp[cfg], "v")
        lo, hi = _ci(boot)
        ci = f"[{lo:.3f},{hi:.3f}]" if lo is not None else ""
        ms, _ = _twoway_boot(g_strict[cfg], "v")
        d = _dist_line(byc[cfg])
        dist = f"{d['n_camel']}/{d['n_snake']}/{d['n_pascal']}/{d['n_other']}/{d['n_name_fail']}"
        print(f"{cfg:20s} {m:7.3f} {ci:>18s} {ms:9.3f}  {dist:>28s}")


def print_summary(out_path, n_boot=2000):
    main = MAIN_LAYER
    any_out = False

    # ── 작동점 보정 (run_tag별로 분리 — 설정 혼합 집계 방지) ──
    calib = _rows(out_path, "calib")
    if calib:
        any_out = True
        for tag, rows in sorted(_split_by_runtag(calib).items()):
            _gen_table(rows, f"(0) 작동점 보정 — 개입 없음 · {tag}")
            gt = rows[0].get("gate_tag", "")
            ok, msg = _check_gate(out_path, gt)
            print(f"  → 게이트 {'PASS ✅' if ok else 'FAIL ❌'} : {msg}")

    # ── (1) 개입 아래 생성 준수율 ──
    grows = _rows(out_path, "gen")
    if grows:
        any_out = True
        for tag, rows in sorted(_split_by_runtag(grows).items()):
            _gen_table(rows, f"(1) 개입 아래 생성 준수율 · {tag}")
            for lab, a_c, b_c in ((f"L{main} − 손상 (주 검정)", f"p1a_L{main}", "damaged"),
                                  ("무작위 camel − 무작위 snake (스타일)",
                                   f"randcamel_L{main}", f"randsnake_L{main}"),
                                  (f"L{main} − L{main} self(no-op)", f"p1a_L{main}", f"noop_L{main}"),
                                  ("P2 λ2 − 손상", "p2_lam2", "damaged"),
                                  ("P2 λ0.5 − 손상", "p2_lam0.5", "damaged")):
                rr = _cells_paired(rows, "compliant_rate", a_c, b_c)
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
        print(f"{'조건':20s} {'snake위반':>8s} {'엄격비준수':>9s} {'이름실패':>8s}  n")
        for cfg in sorted(gv):
            mv, _ = _twoway_boot(gv[cfg], "v")
            ms, _ = _twoway_boot(gs[cfg], "v")
            nf = sum(int(r.get("name_fail_count", 0)) for r in byc[cfg])
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
    ap = argparse.ArgumentParser(description="3단계 행동 실험 v2 (표집·작동점 게이트·후속 위반)")
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
    ap.add_argument("--prompt-variant", default="decision",
                    help="'decision'(현재 하네스)만 구현. 'stage1'(12-prefix·교차작업)은 예정.")
    # 표집
    ap.add_argument("--sample", action="store_true", help="표집 생성(그리디 천장 대응)")
    ap.add_argument("--temperature", type=float, default=SAMPLE_TEMP)
    ap.add_argument("--top-p", dest="top_p", type=float, default=SAMPLE_TOPP)
    ap.add_argument("--n-sample-seeds", type=int, default=N_SAMPLE_SEEDS,
                    help="이름쌍·seed당 표집 draw 수(1개씩 반복 생성)")
    # 모드
    ap.add_argument("--calibrate", action="store_true", help="(0) 개입 없이 base만 — 작동점 탐색")
    ap.add_argument("--generate", action="store_true", help="(1) 개입 아래 생성 (게이트 PASS 필요)")
    ap.add_argument("--gen-chain", action="store_true", help="(2) 연쇄 후속 위반")
    ap.add_argument("--force-ungated", action="store_true", help="(1) 게이트 미통과에도 강제 실행")
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
