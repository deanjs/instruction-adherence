"""
실험 1 (확증) — context 내 위반 코드가 후속 함수의 규칙 준수를 낮추는가 (H1a)

파일럿(`exp1_pilot.py`)은 "효과가 있나·어디 있나"를 찾는 탐색이었다.
이 스크립트는 그 위에서 **논문용 확증 실험**을 돌린다. 조작은 같고 통제·표본을 갖춘다.

주 가설
  H1a: context의 위반 코드가 많을수록(= compliant 예시가 적을수록) 첫 함수의 준수율이 낮다.
       (파일럿에서 3B·weak 지침 기준 100→75→62→38→0으로 확인됨)

조작 (계획서 6.1 재설계 — docs/experiments/01_실험1.md)
  - 고정 크기 prefix(12함수) 안의 **남은 compliant(camel) 예시 수**를 조건으로 둔다.
    compliant_remaining ∈ {4,3,2,1,0}  (= snake 8/9/10/11/12개)
  - **핵심 통제:** 모든 조건에서 prefix 함수 수(12)가 같다 → "코드 양"이 아니라
    "compliant/위반 혼합비"만 다르다. 효과가 나오면 그건 스타일 혼합 탓이다.

통제 항목 (파일럿 대비 추가)
  1. 위반 위치 무작위화(seed) + 조건 간 중첩(nested)으로 대응.
  2. **작업 지시 교차배치** — target 함수의 작업 지시를 seed마다 회전시켜
     위치(1/2/3)와 작업 지시를 분리(교란 제거).
  3. **prefix 토큰 수를 공변량으로 기록** — camel/snake 토큰화 차이로 조건 간
     prefix 길이가 달라지므로, 분석에서 공변량으로 넣는다(계획서 6.1/6.4).
  4. 판정불가(이름 추출 실패)를 위반과 분리해 기록.

측정 (계획서 6.1)
  - **주 결과 = prefix 직후 첫 함수**(순수 직접효과).
  - 보조 = 2·3번째(연쇄효과, 조건 × 위치). 파일럿에서 path dependence 확인됨.

출력
  - 세션마다 함수 단위 레코드를 JSONL에 남긴다. 함수 1행 = GLMM 1관측:
    {compliant_remaining, position, spec_id, compliant, parse_ok, prefix_tokens, seed, ...}
  - 통계 모형(GLMM, 계획서 6.4)은 이 JSONL로 별도 적합(R/statsmodels). 여기선 기술통계 + H1a 대비까지.

미포함(의도적)
  - 위약 지침 조건 + TOST → attention 대조용이라 2단계에서. 여기선 행동 H1a에 집중.
  - Neutral prefix(코드 있으나 표기 신호 없음) → 실험 2(보류) 영역.

모델: Qwen2.5-Coder-3B-Instruct (fp16, 양자화 없음), 지침 = weak. temp 0.7 / top_p 0.95.

usage:
  python src/exp1_main.py                       # 기본 확증 실행
  python src/exp1_main.py --n-seeds 20
  python src/exp1_main.py --summary-only --chain
  python src/exp1_main.py --out /content/drive/MyDrive/.../exp1_main.jsonl   # Drive 보존
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_baseline import (  # noqa: E402
    FUNCTION_SPECS,
    classify_case,
    extract_name,
    is_compliant,
)
from exp1_pilot import (  # noqa: E402  (prefix 12함수·지침·기반 시스템 재사용)
    BASE_SYSTEM,
    INSTRUCTION_RULE,
    PREFIX_FUNCS,
)

# 설계 버전 — 파일럿(v2_prefix12) 및 옛 실행과 절대 안 섞이게 완료 판정·기록에 박는다.
DESIGN_VERSION = "exp1_main_v1"

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
DEFAULT_OUT = os.path.join(RESULTS_DIR, "exp1_main.jsonl")
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
INSTRUCTION = "weak"  # 파일럿에서 55~85% 경쟁 영역을 매끄럽게 관통한 지침 강도


def build_prefix(compliant_remaining, seed):
    """prefix 12함수를 조립. compliant_remaining 개를 camel(준수), 나머지를 snake(위반)로.

    - seed로 함수 순서를 섞고(위반 위치 무작위화),
    - compliant는 그 순열의 앞쪽 k개 → 조건 간 중첩(nested)이라 같은 seed에서 대응된다.
    반환: (prefix 문자열, 위반 함수명 리스트 — 기록용)
    """
    n = len(PREFIX_FUNCS)
    perm = random.Random(seed).sample(range(n), n)
    compliant_idx = set(perm[:compliant_remaining])
    parts, violated = [], []
    for i in perm:  # 섞인 순서대로 배치
        if i in compliant_idx:
            parts.append(PREFIX_FUNCS[i]["camel"])
        else:
            parts.append(PREFIX_FUNCS[i]["snake"])
            violated.append(PREFIX_FUNCS[i]["snake"].split("(")[0].replace("def ", ""))
    return "\n\n\n".join(parts), violated


def target_specs(seed, n_targets):
    """target 함수의 작업 지시를 seed마다 회전. 위치와 작업 지시를 분리(교차배치).

    같은 위치(예: 위치1)가 seed마다 다른 작업 지시를 받게 되어, '위치 효과'와
    '작업 지시 효과'가 교란되지 않는다.
    """
    pool = FUNCTION_SPECS
    start = random.Random(seed).randrange(len(pool))
    return [pool[(start + j) % len(pool)] for j in range(n_targets)]


def system_prompt():
    lang = "Python"
    return (
        f"You are a senior {lang} engineer working in this project. "
        f"Respond with exactly one {lang} function inside a single ```python code block, "
        f"and nothing else." + INSTRUCTION_RULE[INSTRUCTION]
    )


def user_prompt_first(prefix, spec):
    return (
        "Here is the existing code in this project:\n\n"
        f"```python\n{prefix}\n```\n\n"
        f"Now add a function that {spec['desc']}."
    )


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


# ─────────────────────────────────────────────────────────────
# 한 세션
# ─────────────────────────────────────────────────────────────

def run_session(model, tokenizer, compliant_remaining, seed, args, torch):
    prefix, violated = build_prefix(compliant_remaining, seed)
    specs = target_specs(seed, args.n_targets)

    # prefix 토큰 수(공변량). 조건 간 camel/snake 혼합비가 달라 값이 변한다.
    prefix_tokens = len(tokenizer(prefix, add_special_tokens=False).input_ids)

    messages = [{"role": "system", "content": system_prompt()}]

    # 대응 설계: 세션 seed를 한 번 고정(조건 간 공유).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    functions = []
    for pos, spec in enumerate(specs, start=1):
        content = (user_prompt_first(prefix, spec) if pos == 1
                   else f"Add a function that {spec['desc']}.")
        messages.append({"role": "user", "content": content})

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": gen_text})

        name, parse_ok = extract_name("python", gen_text)
        functions.append({
            "position": pos,
            "spec_id": spec["id"],
            "generated_name": name,
            "case": classify_case(name),
            "compliant": is_compliant(name, "camel"),
            "parse_ok": parse_ok,
            "raw": gen_text,
        })

    return {
        "design_version": DESIGN_VERSION,
        "model": model.name_or_path,
        "instruction": INSTRUCTION,
        "prefix_size": len(PREFIX_FUNCS),
        "compliant_remaining": compliant_remaining,
        "violation_count": len(PREFIX_FUNCS) - compliant_remaining,
        "prefix_tokens": prefix_tokens,          # 공변량
        "violated_names": violated,
        "seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_targets": args.n_targets,
        "target_spec_order": [s["id"] for s in specs],
        "position1_name": functions[0]["generated_name"],
        "position1_compliant": functions[0]["compliant"],
        "functions": functions,
    }


# ─────────────────────────────────────────────────────────────
# 집계 (기술통계 — 확증 통계 GLMM은 JSONL로 별도)
# ─────────────────────────────────────────────────────────────

def _load_rows(out_path):
    rows = []
    if not os.path.exists(out_path):
        return rows
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("design_version") == DESIGN_VERSION:
                rows.append(r)
    return rows


def _f_at(session, position):
    return next((x for x in session.get("functions", []) if x["position"] == position), None)


def _pos_stats(sessions, position):
    """(판정가능 수, 준수 수, 전체 수). 이름 추출 실패는 분모에서 제외(위반과 분리)."""
    judge = []
    for s in sessions:
        f = _f_at(s, position)
        if f is not None and f.get("generated_name") is not None:
            judge.append(bool(f["compliant"]))
    return len(judge), sum(judge), len(sessions)


def print_summary(out_path, levels):
    rows = _load_rows(out_path)
    if not rows:
        print(f"(design_version={DESIGN_VERSION} 레코드 없음: {out_path})")
        return
    models = sorted({r["model"].split("/")[-1] for r in rows})

    cell = {}
    for r in rows:
        cell.setdefault(r["compliant_remaining"], []).append(r)

    print(f"\n=== 실험1 확증 요약 [{DESIGN_VERSION}] {'/'.join(models)} "
          f"/ 위치1 camelCase 준수율 ===")
    print("열 = 남은 compliant 예시 수(4=대부분 준수 … 0=전부 위반).")
    print("값 = 위치1 준수율%(판정가능 세션수). ?k = 판정불가 k개\n")

    header = "  compliant 남음 :"
    p1line = "  위치1 준수율   :"
    tokline = "  prefix 토큰(평균):"
    for cr in levels:
        ss = cell.get(cr, [])
        nj, nc, nt = _pos_stats(ss, 1)
        header += "  {:^9s}".format(f"{cr}")
        if nj:
            s = "{:>3.0f}%(n{:d})".format(nc / nj * 100, nj)
            if nj < nt:
                s += "?{:d}".format(nt - nj)
        else:
            s = "--" if not ss else "판정불가"
        p1line += "  {:^9s}".format(s)
        toks = [x["prefix_tokens"] for x in ss if "prefix_tokens" in x]
        tokline += "  {:^9s}".format(f"{sum(toks)/len(toks):.0f}" if toks else "--")
    print(header)
    print(p1line)
    print(tokline)

    # H1a 대비: 최대 compliant vs 0 (all-violating)
    hi, lo = max(levels), min(levels)
    nj_h, nc_h, _ = _pos_stats(cell.get(hi, []), 1)
    nj_l, nc_l, _ = _pos_stats(cell.get(lo, []), 1)
    if nj_h and nj_l:
        rh, rl = nc_h / nj_h * 100, nc_l / nj_l * 100
        print(f"\n  [H1a 대비] compliant {hi} → {rh:.0f}%  vs  compliant {lo} → {rl:.0f}%  "
              f"(차이 {rh - rl:.0f}%p)")
        print("  ※ 확증 유의성은 JSONL을 GLMM(계획서 6.4)으로 적합해 판정한다.")


def print_chain(out_path, levels):
    """연쇄효과: 위치1 O/X에 따른 위치2·3 준수율 (파일럿에서 발견한 path dependence 재확인)."""
    rows = _load_rows(out_path)
    if not rows:
        return
    cell = {}
    for r in rows:
        cell.setdefault(r["compliant_remaining"], []).append(r)

    def rate(grp, pos):
        vals = [_f_at(s, pos)["compliant"] for s in grp
                if _f_at(s, pos) and _f_at(s, pos).get("generated_name") is not None]
        return (sum(vals) / len(vals) * 100, len(vals)) if vals else (None, 0)

    print("\n=== 연쇄효과: 위치1 결과 → 위치2·3 준수율 (판정가능만) ===")
    for cr in levels:
        sessions = cell.get(cr, [])
        o_grp, x_grp = [], []
        for s in sessions:
            f1 = _f_at(s, 1)
            if not f1 or f1.get("generated_name") is None:
                continue
            (o_grp if f1["compliant"] else x_grp).append(s)
        if not o_grp and not x_grp:
            continue
        print(f"  [compliant 남음 {cr}]")
        for label, grp in [("위치1=O", o_grp), ("위치1=X", x_grp)]:
            if not grp:
                continue
            (r2, n2), (r3, n3) = rate(grp, 2), rate(grp, 3)
            r2s = f"{r2:.0f}%(n{n2})" if r2 is not None else "--"
            r3s = f"{r3:.0f}%(n{n3})" if r3 is not None else "--"
            print(f"    {label} (n{len(grp)}) → 위치2 {r2s}, 위치3 {r3s}")
        print()


# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="실험 1 확증 (3B, weak, compliant 예시 수 조작)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 3, 2, 1, 0],
                    help="조건 = 남은 compliant 예시 수")
    ap.add_argument("--n-seeds", type=int, default=20, help="조건당 세션 수")
    ap.add_argument("--base-seed", type=int, default=1000,
                    help="파일럿과 seed 공간을 분리(기본 1000~)")
    ap.add_argument("--n-targets", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--chain", action="store_true", help="연쇄효과도 출력")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    max_cr = len(PREFIX_FUNCS)
    if any(cr < 0 or cr > max_cr for cr in args.levels):
        ap.error(f"--levels 는 0~{max_cr}(prefix 함수 수) 범위: {args.levels}")

    if args.summary_only:
        print_summary(args.out, args.levels)
        if args.chain:
            print_chain(args.out, args.levels)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("경고: CUDA 없음. fp16 추론은 GPU 런타임이 필요하다.", file=sys.stderr)

    seeds = [args.base_seed + k for k in range(args.n_seeds)]
    done = load_done(args.out)
    pending = [
        (cr, s)
        for cr in args.levels
        for s in seeds
        if (DESIGN_VERSION, args.model, cr, s) not in done
    ]
    print(f"완료 {len(done)}개, 남은 {len(pending)}개 세션 "
          f"(조건 {len(args.levels)} × seed {args.n_seeds})")
    if not pending:
        print_summary(args.out, args.levels)
        if args.chain:
            print_chain(args.out, args.levels)
        return

    print(f"[{args.model}] 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    for cr, s in pending:
        print(f"  compliant={cr} seed={s} ...", end=" ", flush=True)
        with torch.no_grad():
            rec = run_session(model, tokenizer, cr, s, args, torch)
        append_jsonl(args.out, rec)
        print(f"위치1 {'O' if rec['position1_compliant'] else 'X'} ({rec['position1_name']})")

    print_summary(args.out, args.levels)
    if args.chain:
        print_chain(args.out, args.levels)


if __name__ == "__main__":
    main()
