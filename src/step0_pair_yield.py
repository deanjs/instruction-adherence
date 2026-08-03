"""
Step 0 — 개입 실험용 함수명 pair 토큰 수 일치 수율 측정

3단계 개입 실험은 준수/위반 두 조건의 토큰 위치가 정확히 일치해야만 가능하다.
`getUserData` 와 `get_user_data` 가 같은 토큰 수로 쪼개지는 비율을 먼저 잰다.
이 수율이 너무 낮으면 개입 실험 설계 자체를 바꿔야 한다.

GPU 불필요. tokenizer만 받으면 CPU에서 수 초.

usage:
  python src/step0_pair_yield.py
  python src/step0_pair_yield.py --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

import argparse
import itertools
import json
import os
from collections import Counter

from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# 이름 풀의 대용. 최종적으로는 대상 저장소의 기존 모듈에서 추출한
# 명세 목록으로 교체한다 (계획서 5.3 "생성물 고정").
VERBS = [
    "get", "fetch", "create", "update", "delete", "list", "validate",
    "parse", "build", "send", "load", "save", "find", "check", "render",
    "resolve", "apply", "merge", "filter", "count", "sync", "export",
]

NOUN_GROUPS = [
    ["user"], ["order"], ["session"], ["token"], ["payment"], ["invoice"],
    ["user", "order"], ["order", "item"], ["payment", "method"],
    ["session", "token"], ["api", "key"], ["cache", "entry"],
    ["user", "profile"], ["shipping", "address"], ["access", "token"],
    ["order", "history"], ["billing", "record"], ["auth", "header"],
]

# 결정 지점 직전 context.
# BPE는 경계를 넘어 병합되므로 이름만 따로 tokenize한 결과와 다를 수 있다.
CONTEXTS = {
    "bare": "",
    "def": "\ndef ",
    "def_in_class": "\nclass OrderService:\n    def ",
}


def camel(verb, nouns):
    return verb + "".join(n.capitalize() for n in nouns)


def snake(verb, nouns):
    return "_".join([verb] + nouns)


def tokenize_span(tok, prefix, name):
    """prefix + name 에서 name이 차지하는 토큰 구간.

    prefix 토큰열이 그대로 보존되지 않으면(경계 병합) None을 반환한다.
    결정 지점 인덱스가 정의되지 않는다는 뜻이므로 개입에 쓸 수 없다.
    """
    base = tok(prefix, add_special_tokens=False).input_ids
    full = tok(prefix + name, add_special_tokens=False).input_ids
    if full[: len(base)] != base:
        return None
    return full[len(base):]


def first_divergence(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    print(f"tokenizer: {args.model}\n")
    tok = AutoTokenizer.from_pretrained(args.model)

    names = list(itertools.product(VERBS, NOUN_GROUPS))
    print(f"후보 이름 수: {len(names)}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}

    for ctx_label, prefix in CONTEXTS.items():
        matched, merged, mismatched = [], 0, Counter()
        single_token_divergence = 0

        for verb, nouns in names:
            c_name, v_name = camel(verb, nouns), snake(verb, nouns)
            c_ids = tokenize_span(tok, prefix, c_name)
            v_ids = tokenize_span(tok, prefix, v_name)

            if c_ids is None or v_ids is None:
                merged += 1
                continue

            if len(c_ids) != len(v_ids):
                mismatched[(len(c_ids), len(v_ids))] += 1
                continue

            matched.append({
                "camel": c_name,
                "snake": v_name,
                "n_tokens": len(c_ids),
                "camel_ids": c_ids,
                "snake_ids": v_ids,
            })

            # 보조 지표(계획서 3.5): 정확히 한 토큰에서만 갈리는 비율
            d = first_divergence(c_ids, v_ids)
            if c_ids[d + 1:] == v_ids[d + 1:]:
                single_token_divergence += 1

        total = len(names)
        rate = len(matched) / total * 100
        print(f"[context = {ctx_label!r}]")
        print(f"  토큰 수 일치     : {len(matched)}/{total}  ({rate:.1f}%)")
        print(f"  prefix 경계 병합 : {merged}")
        print(f"  단일 토큰 분기   : {single_token_divergence}"
              f" (매칭분의 {single_token_divergence / max(len(matched), 1) * 100:.1f}%)")
        if mismatched:
            print(f"  불일치 패턴 (camel, snake): {mismatched.most_common(5)}")
        print(f"  → 500쌍 확보에 필요한 후보 풀: 약 {int(500 / max(rate / 100, 1e-9))}개\n")

        summary[ctx_label] = {
            "n_candidates": total,
            "n_matched": len(matched),
            "match_rate_pct": round(rate, 2),
            "n_boundary_merged": merged,
            "n_single_token_divergence": single_token_divergence,
        }

        if ctx_label == "def":
            out = os.path.join(RESULTS_DIR, "matched_pairs.json")
            with open(out, "w") as f:
                json.dump(matched, f, indent=2, ensure_ascii=False)
            print(f"  저장: {out} ({len(matched)}쌍)\n")

    # 선별 표본 편향 확인 (계획서 3.6)
    print("=== 선별 표본 편향 확인 ===")
    all_names = [camel(v, n) for v, n in names]
    matched_names = [m["camel"] for m in matched]
    for label, pool in [("전체 후보", all_names), ("선별 후보", matched_names)]:
        if pool:
            avg = sum(len(x) for x in pool) / len(pool)
            print(f"  {label}: n={len(pool)}, 평균 문자 길이 {avg:.1f}")

    with open(os.path.join(RESULTS_DIR, "step0_summary.json"), "w") as f:
        json.dump({"model": args.model, "contexts": summary}, f, indent=2)


if __name__ == "__main__":
    main()
