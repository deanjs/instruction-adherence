"""
Step 0 — 개입 실험용 함수명 pair 토큰 수 일치 수율 측정 (v2)

3단계 개입 실험은 준수/위반 두 조건의 토큰 위치가 정확히 일치해야만 성립한다.
`getUserData` / `get_user_data` 가 같은 토큰 수로 쪼개지는 비율을 측정한다.

v1에서 prefix를 따로 tokenize해 앞부분 보존을 요구했는데, 이는 잘못이다.
Qwen BPE는 공백을 뒤 단어에 붙이므로 `"\\ndef "` 를 단독 tokenize한 결과가
`"\\ndef getUserData"` 의 앞부분과 일치하지 않는다. 실제 실험에서는 전체
문자열을 한 번에 tokenize하므로, 두 전체 시퀀스를 직접 비교하는 것이 맞다.

개입에 필요한 조건:
  1. 두 시퀀스의 총 토큰 수가 같다
  2. 갈리기 전까지의 토큰이 완전히 동일하다 (결정 지점 인덱스가 같다)

GPU 불필요.

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
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

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

CONTEXTS = {
    "bare": "",
    "def": "\ndef ",
    "def_in_class": "\nclass OrderService:\n    def ",
    "with_code": (
        "\nfrom typing import Any\n\n\n"
        "def loadConfig(path: str) -> dict[str, Any]:\n"
        "    with open(path) as f:\n"
        "        return json.load(f)\n\n\n"
        "def "
    ),
}


def camel(verb, nouns):
    return verb + "".join(n.capitalize() for n in nouns)


def snake(verb, nouns):
    return "_".join([verb] + nouns)


def first_divergence(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def compare_pair(tok, prefix, name_c, name_v):
    """두 전체 문자열을 tokenize해 개입 가능 여부를 판정.

    반환 None이면 토큰 수 불일치로 개입에 쓸 수 없다.
    """
    ids_c = tok(prefix + name_c, add_special_tokens=False).input_ids
    ids_v = tok(prefix + name_v, add_special_tokens=False).input_ids

    if len(ids_c) != len(ids_v):
        return None

    d = first_divergence(ids_c, ids_v)
    if d == len(ids_c):  # 완전히 동일 — 있을 수 없지만 방어
        return None

    # 갈린 뒤 정확히 한 토큰만 다르면 보조 지표(계획서 3.5) 대상
    single = ids_c[d + 1:] == ids_v[d + 1:]

    return {
        "camel": name_c,
        "snake": name_v,
        "total_tokens": len(ids_c),
        "decision_index": d,       # 결정 지점 = 처음 갈리는 위치
        "name_span": len(ids_c) - d,
        "single_token_divergence": single,
        "camel_ids": ids_c[d:],
        "snake_ids": ids_v[d:],
    }


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
    matched_for_export = []

    for ctx_label, prefix in CONTEXTS.items():
        matched, mismatched = [], Counter()

        for verb, nouns in names:
            r = compare_pair(tok, prefix, camel(verb, nouns), snake(verb, nouns))
            if r is None:
                c = len(tok(prefix + camel(verb, nouns), add_special_tokens=False).input_ids)
                v = len(tok(prefix + snake(verb, nouns), add_special_tokens=False).input_ids)
                mismatched[(c - v)] += 1
                continue
            matched.append(r)

        total = len(names)
        rate = len(matched) / total * 100
        n_single = sum(m["single_token_divergence"] for m in matched)

        print(f"[context = {ctx_label!r}]")
        print(f"  토큰 수 일치   : {len(matched)}/{total}  ({rate:.1f}%)")
        print(f"  단일 토큰 분기 : {n_single}"
              f" (매칭분의 {n_single / max(len(matched), 1) * 100:.1f}%)")
        if mismatched:
            print(f"  길이 차이 분포 (camel − snake): {dict(mismatched.most_common(5))}")
        print(f"  → 500쌍 확보에 필요한 후보 풀: 약 {int(500 / max(rate / 100, 1e-9))}개\n")

        summary[ctx_label] = {
            "n_candidates": total,
            "n_matched": len(matched),
            "match_rate_pct": round(rate, 2),
            "n_single_token_divergence": n_single,
        }

        if ctx_label == "def":
            matched_for_export = matched

    # 실험 환경에 가장 가까운 'def' 기준으로 저장
    out = os.path.join(RESULTS_DIR, "matched_pairs.json")
    with open(out, "w") as f:
        json.dump(matched_for_export, f, indent=2, ensure_ascii=False)
    print(f"저장: {out} ({len(matched_for_export)}쌍)\n")

    # 선별 표본 편향 확인 (계획서 3.6)
    print("=== 선별 표본 편향 확인 ===")
    all_names = [camel(v, n) for v, n in names]
    sel_names = [m["camel"] for m in matched_for_export]
    for label, pool in [("전체 후보", all_names), ("선별 후보", sel_names)]:
        if pool:
            avg = sum(len(x) for x in pool) / len(pool)
            print(f"  {label}: n={len(pool)}, 평균 문자 길이 {avg:.1f}")

    with open(os.path.join(RESULTS_DIR, "step0_summary.json"), "w") as f:
        json.dump({"model": args.model, "contexts": summary}, f, indent=2)


if __name__ == "__main__":
    main()