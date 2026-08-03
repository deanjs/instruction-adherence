"""
실험 1 축소 파일럿 — context 내 위반 코드가 후속 준수율을 낮추는가 (계획서 6.1)

배경 (Step 1 결과)
  표기 규칙(camelCase)에서 8셀 전부 게이트 밖이었다.
  1.5B는 지침을 무시하고 언어 관습만 따르고(python_camel 0%),
  3B는 관습을 무시하고 지침을 거의 완벽히 따른다(python_camel 99%).
  → 표기 규칙만으론 두 모델 다 '경쟁 영역(55~85%)'이 안 만들어진다.

이 파일럿이 검증하는 것
  3B는 camelCase를 '할 줄은 아는' 유일한 모델이다(개입 실험 후보).
  99%인 건 지침을 강하게 박아 천장에 붙인 것일 뿐일 수 있다. 그래서 두 축을 동시에 움직인다:
    (1) 지침 강도  {none, weak, strong} — baseline을 천장에서 떼어낼 수 있는가
    (2) 위반 개수  V0..V3 (prefix 4함수 중 snake_case 개수) — 위반 코드가 준수율을 낮추는가

  핵심 질문: 지침을 약화한 상태에서, context에 snake_case 함수를 더 깔수록
  모델이 새로 만드는 함수의 camelCase 준수율이 떨어지는가(단조 감소)?
  떨어지면 → 행동 수준 인과 효과가 존재. baseline이 높든 낮든 실험이 성립.

설계 원칙 (계획서 6.1 준수)
  - prefix 4함수는 네 조건에서 기능이 완전히 동일하고 함수명 표기만 다르다.
  - 위반 위치는 seed로 무작위화하되, V0⊂V1⊂V2⊂V3 로 중첩(nested)해 깨끗한 용량 배열.
  - 주 결과 = prefix 직후 '첫 번째 함수'(순수 직접효과). 2·3번째는 연쇄효과 보조.
  - 위반 코드는 연구자 작성 고정본(모델 생성분 아님) — 조건 간 동일성이 인과 식별의 전제.

한계 (파일럿이라 단순화)
  - 토큰 길이 정렬(±3, 공변량)은 본 실험용이라 여기선 생략. 방향(효과 유무)만 본다.
  - camel/snake 이름은 토큰 수가 다를 수 있어 잔여 교란이 남는다. 결과는 예비적.

seed는 GPU 절약을 위해 적게(기본 5). fp16, 양자화 없음.

usage:
  python src/exp1_pilot.py
  python src/exp1_pilot.py --instructions weak none --violations 0 3 --n-seeds 5
"""

import argparse
import json
import os
import random
import sys

# Step 1 하네스의 판정 헬퍼와 명세를 재사용 (동일 스크립트가 채점 기준)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_baseline import (  # noqa: E402
    FUNCTION_SPECS,
    classify_case,
    extract_name,
    is_compliant,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
DEFAULT_OUT = os.path.join(RESULTS_DIR, "exp1_pilot.jsonl")
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

# prefix 고정 4함수. camel/snake 는 함수명만 다르고 본문은 완전히 동일하다.
# (규칙은 '함수명 표기'이므로 파라미터/변수는 두 버전 모두 snake로 둔다)
PREFIX_FUNCS = [
    {
        "camel": (
            "def getUserById(user_id):\n"
            '    """Return the user record for the given id."""\n'
            "    return DATABASE.get(user_id)"
        ),
        "snake": (
            "def get_user_by_id(user_id):\n"
            '    """Return the user record for the given id."""\n'
            "    return DATABASE.get(user_id)"
        ),
    },
    {
        "camel": (
            "def calculateOrderTotal(items):\n"
            '    """Sum the price of every item in the order."""\n'
            "    return sum(item.price for item in items)"
        ),
        "snake": (
            "def calculate_order_total(items):\n"
            '    """Sum the price of every item in the order."""\n'
            "    return sum(item.price for item in items)"
        ),
    },
    {
        "camel": (
            "def formatPhoneNumber(raw):\n"
            '    """Strip separators from a phone number string."""\n'
            "    return raw.replace('-', '').replace(' ', '')"
        ),
        "snake": (
            "def format_phone_number(raw):\n"
            '    """Strip separators from a phone number string."""\n'
            "    return raw.replace('-', '').replace(' ', '')"
        ),
    },
    {
        "camel": (
            "def isValidToken(token):\n"
            '    """Check whether a token has not yet expired."""\n'
            "    return token.expires_at > now()"
        ),
        "snake": (
            "def is_valid_token(token):\n"
            '    """Check whether a token has not yet expired."""\n'
            "    return token.expires_at > now()"
        ),
    },
]

BASE_SYSTEM = (
    "You are a senior Python engineer working in this project. "
    "Respond with exactly one Python function inside a single ```python code block, "
    "and nothing else."
)

# 지침 강도 3단계. baseline(V0)을 천장에서 떼어낼 수 있는지 보려는 축.
INSTRUCTION_RULE = {
    # 규칙 없음 — 순수 관습 baseline (Python이면 snake → camel 준수율 바닥 예상)
    "none": "",
    # 약한 지침 — 평범하게 한 문장
    "weak": " Use camelCase for function names.",
    # 강한 지침 — Step 1과 동일하게 세게 박음 (천장 예상, 대조용)
    "strong": (
        " PROJECT RULE: every function name you write MUST be in camelCase. "
        "This rule applies to every single function you create, without exception."
    ),
}


def build_prefix(violation_count, seed):
    """prefix 4함수를 조립. violation_count 개를 snake로, 나머지를 camel로.

    위반 위치는 seed로 정한 순열의 앞쪽 k개 → V0⊂V1⊂V2⊂V3 중첩 배열.
    """
    perm = random.Random(seed).sample(range(len(PREFIX_FUNCS)), len(PREFIX_FUNCS))
    violation_idx = set(perm[:violation_count])
    parts = [
        f["snake"] if i in violation_idx else f["camel"]
        for i, f in enumerate(PREFIX_FUNCS)
    ]
    return "\n\n\n".join(parts)


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
                done.add((r["instruction"], r["violation_count"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(out_path, record):
    with open(out_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_session(model, tokenizer, instruction, violation_count, seed, args, torch):
    """한 세션 = prefix 제시 후 target 함수 n_targets개 순차 생성. 주 지표는 함수 1."""
    system = BASE_SYSTEM + INSTRUCTION_RULE[instruction]
    prefix = build_prefix(violation_count, seed)

    messages = [{"role": "system", "content": system}]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    functions = []
    for pos in range(1, args.n_targets + 1):
        spec = FUNCTION_SPECS[pos - 1]
        if pos == 1:
            # 첫 요청에만 기존 코드(prefix)를 함께 제시
            content = (
                "Here is the existing code in this project:\n\n"
                f"```python\n{prefix}\n```\n\n"
                f"Now add a function that {spec['desc']}."
            )
        else:
            content = f"Add a function that {spec['desc']}."
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
        "model": model.name_or_path,
        "instruction": instruction,
        "violation_count": violation_count,
        "seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_targets": args.n_targets,
        "position1_name": functions[0]["generated_name"],
        "position1_compliant": functions[0]["compliant"],
        "functions": functions,
    }


def print_summary(out_path, instructions, violations):
    if not os.path.exists(out_path):
        return
    rows = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    # (instruction, V) → 위치1 준수 리스트
    cell = {}
    for r in rows:
        k = (r["instruction"], r["violation_count"])
        cell.setdefault(k, []).append(r["position1_compliant"])

    print("\n=== 실험1 파일럿 요약 (3B, python camelCase / 위치1 준수율) ===")
    print("행=지침강도, 열=위반개수(V0..V3). 값 = 첫 함수 camelCase 준수율%(세션수)\n")
    header = "  {:8s}".format("지침")
    for v in violations:
        header += "  {:^9s}".format(f"V{v}")
    print(header + "  | 해석")

    for instr in instructions:
        line = "  {:8s}".format(instr)
        rates = []
        for v in violations:
            vals = cell.get((instr, v))
            if vals:
                rate = sum(vals) / len(vals) * 100
                rates.append(rate)
                line += "  {:>4.0f}%(n{:d})".format(rate, len(vals))
            else:
                rates.append(None)
                line += "  {:^9s}".format("--")
        # 해석: V가 커질수록 단조 감소하면 효과 O
        print(line + f"  | {_interpret(rates)}")


def _interpret(rates):
    xs = [r for r in rates if r is not None]
    if len(xs) < 2:
        return "데이터 부족"
    drop = xs[0] - xs[-1]
    mono = all(xs[i] >= xs[i + 1] - 1e-9 for i in range(len(xs) - 1))
    if xs[0] > 90 and drop < 5:
        return "천장 고정 (위반에도 안 흔들림)"
    if drop >= 10 and mono:
        return f"효과 O — 단조 감소 {drop:.0f}%p"
    if drop >= 10:
        return f"감소 {drop:.0f}%p (비단조)"
    if xs[0] < 10:
        return "바닥 고정 (신호 없음)"
    return "효과 미미"


def main():
    ap = argparse.ArgumentParser(description="실험1 축소 파일럿 (3B, 지침강도×위반개수)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--instructions", nargs="+", default=["none", "weak", "strong"],
                    choices=["none", "weak", "strong"])
    ap.add_argument("--violations", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--n-seeds", type=int, default=5, help="조합당 세션(seed) 수")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--n-targets", type=int, default=3, help="prefix 뒤 생성 함수 수(주지표=1)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.summary_only:
        print_summary(args.out, args.instructions, args.violations)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("경고: CUDA 없음. fp16 추론은 GPU 런타임이 필요하다.", file=sys.stderr)

    seeds = [args.base_seed + k for k in range(args.n_seeds)]
    done = load_done(args.out)

    pending = [
        (instr, v, s)
        for instr in args.instructions
        for v in args.violations
        for s in seeds
        if (instr, v, s) not in done
    ]
    print(f"완료 {len(done)}개, 남은 {len(pending)}개 세션")
    if not pending:
        print_summary(args.out, args.instructions, args.violations)
        return

    print(f"[{args.model}] 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    for instr, v, s in pending:
        print(f"  {instr:6s} V{v} seed={s} ...", end=" ", flush=True)
        with torch.no_grad():
            rec = run_session(model, tokenizer, instr, v, s, args, torch)
        append_jsonl(args.out, rec)
        print(f"위치1 {'O' if rec['position1_compliant'] else 'X'} ({rec['position1_name']})")

    print_summary(args.out, args.instructions, args.violations)


if __name__ == "__main__":
    main()
