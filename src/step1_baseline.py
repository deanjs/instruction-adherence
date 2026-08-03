"""
Step 1 — 기저 준수율 smoke test (계획서 5.5 게이트)

목적
  1. H1 하네스의 기저 준수율이 계획서 5.5 게이트(55~85%)에 드는지 확인.
  2. CLAUDE.md 미결정 사항 두 개를 결정한다.
     - 개입용 모델 크기: 1.5B / 3B 중 게이트를 통과하는 가장 작은 모델.
     - 언어별 지침 방향: Python/TS 각각 camelCase·snake_case 중 어느 쪽을
       요구해야 천장 효과(>85%)를 피하는지.

설계 (2 언어 × 2 지침 방향 = 4 셀)
  cell           언어         요구 표기      관용 여부
  python_camel   Python       camelCase      비관용 (anti-idiom)
  python_snake   Python       snake_case     관용   (천장 효과 우려)
  ts_snake       TypeScript   snake_case     비관용 (anti-idiom)
  ts_camel       TypeScript   camelCase      관용   (천장 효과 우려)

  네 셀 모두 동일한 함수 명세 10개를 쓴다. 셀당 10세션, 세션당 함수 10개를
  순차 생성하며 앞서 생성한 함수가 다음 생성의 context에 누적된다(H1 루프, 계획서 3.3).

측정 (계획서 6.1)
  - 함수 위치(1~10)별 준수 여부.
  - 셀별 전체 준수율.
  - 위치 1의 준수율 — prefix 직후 첫 생성이라 순수 직접효과에 해당하는 주 지표.

준수 판정
  - Python: `ast` 모듈로 첫 최상위 FunctionDef 이름을 추출 (파싱 실패 시 정규식 폴백).
  - TypeScript: 정규식으로 추출 (한계는 아래 extract_name_ts 주석 참조).

무작위성 (계획서 5.3 / CLAUDE.md)
  - temperature 0.7, top_p 0.95.
  - seed는 세션마다 기록하며 대응 설계상 조건(모델·셀) 간에 공유된다.
    즉 session_idx=k 는 모델·셀과 무관하게 항상 seed=base_seed+k 를 쓴다.

재개 (CLAUDE.md Colab 제약)
  - 결과는 세션 단위로 results/step1_baseline.jsonl 에 append.
  - 러너 시작 시 이미 완료된 (model, cell, seed) 조합은 건너뛴다.

양자화하지 않는다(fp16). 개입 실험과의 연속성을 위해 동일 정밀도를 쓴다.

usage:
  python src/step1_baseline.py
  python src/step1_baseline.py --models Qwen/Qwen2.5-Coder-1.5B-Instruct
  python src/step1_baseline.py --n-sessions 10 --base-seed 0
"""

import argparse
import ast
import json
import os
import re
import sys

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
DEFAULT_OUT = os.path.join(RESULTS_DIR, "step1_baseline.jsonl")

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
]

# 4개 셀: (언어, 요구 표기). idiomatic 은 해당 언어의 관용 표기 여부 — 보고용 메모.
CELLS = [
    {"cell": "python_camel", "language": "python",     "required_style": "camel", "idiomatic": False},
    {"cell": "python_snake", "language": "python",     "required_style": "snake", "idiomatic": True},
    {"cell": "ts_snake",     "language": "typescript", "required_style": "snake", "idiomatic": False},
    {"cell": "ts_camel",     "language": "typescript", "required_style": "camel", "idiomatic": True},
]

# 고정 함수 명세 10개. 모든 셀에서 동일. 표기 규칙을 판정 가능하게 하려고
# 전부 '여러 단어'로 이름이 자연스럽게 나오는 작업으로 골랐다.
# (단일 단어 이름은 camel/snake 구분이 불가능하므로 배제)
FUNCTION_SPECS = [
    {"id": "user_orders",     "desc": "retrieve the list of orders for a given user ID"},
    {"id": "validate_email",  "desc": "validate an email address string and return a boolean"},
    {"id": "cart_total",      "desc": "calculate the total price of the items in a shopping cart"},
    {"id": "parse_iso_date",  "desc": "parse an ISO 8601 date string into a date object"},
    {"id": "send_confirm",    "desc": "send a confirmation email to a customer"},
    {"id": "merge_sorted",    "desc": "merge two already-sorted lists of integers into one sorted list"},
    {"id": "active_sessions", "desc": "count the number of currently active sessions for a user"},
    {"id": "celsius_to_f",    "desc": "convert a temperature from Celsius to Fahrenheit"},
    {"id": "max_in_nested",   "desc": "find the maximum numeric value inside a nested structure"},
    {"id": "filter_expired",  "desc": "filter out the expired tokens from a list of tokens"},
]

STYLE_LABEL = {"camel": "camelCase", "snake": "snake_case"}


# ─────────────────────────────────────────────────────────────
# 프롬프트 구성
# ─────────────────────────────────────────────────────────────

def system_prompt(language, required_style):
    """CLAUDE.md/AGENTS.md 규칙을 흉내 낸 시스템 지침.

    지침은 system 메시지에 한 번만 두고, 이후 생성된 함수가 context에
    누적된다. '규칙은 앞에 한 번, 코드는 뒤에 쌓인다'는 본 연구의 조작 구조다.
    """
    lang_name = "Python" if language == "python" else "TypeScript"
    style = STYLE_LABEL[required_style]
    return (
        f"You are a senior {lang_name} engineer working in this project.\n"
        f"PROJECT RULE: every function name you write MUST be in {style}. "
        f"This rule applies to every single function you create in this project, "
        f"without exception.\n"
        f"Respond with exactly one {lang_name} function and nothing else. "
        f"Put it inside a single ```{'python' if language == 'python' else 'typescript'} code block."
    )


def user_prompt(language, spec):
    lang_name = "Python" if language == "python" else "TypeScript"
    return f"Write a {lang_name} function that {spec['desc']}."


# ─────────────────────────────────────────────────────────────
# 생성물에서 함수 이름 추출
# ─────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """마크다운 코드펜스가 있으면 첫 블록 내용을, 없으면 원문 전체를 반환."""
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def extract_name_python(text):
    """Python: ast로 첫 최상위 함수명을 뽑는다. 파싱 실패 시 정규식 폴백.

    반환: (name 또는 None, parse_ok)
    """
    code = extract_code(text)
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name, True
        # 최상위 함수가 없으면 (예: 클래스 안에만 있음) 정규식으로 폴백
    except SyntaxError:
        pass
    m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
    return (m.group(1) if m else None), False


# TypeScript 한계 명시:
#   TS는 표준 라이브러리 파서가 없어 정규식으로 근사한다. 다음 형태를 인식한다.
#     function foo(...)            — 함수 선언
#     const foo = (...) =>         / const foo = function(...)  — 변수 대입 화살표/함수식
#     export [default] 접두        — export 접두 허용
#   인식 못 하는 경우: 클래스 메서드, 오버로드 시그니처, 제네릭 앞 <T> 위치,
#   객체 리터럴 메서드 축약. smoke test 규모(단일 함수 요청)에서는 위 세 형태로
#   대부분 잡히며, 첫 매치를 주 함수명으로 본다. 실패분은 name=None 으로 기록되어
#   준수율 분모에서 제외되지 않고 '판정 불가'로 남으니 실패율을 함께 본다.
_TS_PATTERNS = [
    re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*[<(]"),
    re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>)"),
]


def extract_name_ts(text):
    """TypeScript: 정규식으로 첫 함수명 추출. (name 또는 None, parse_ok=False 고정)

    TS는 진짜 파서가 아니므로 parse_ok 개념이 없다. 항상 False 로 두어
    '정규식 근사'임을 결과에 남긴다.
    """
    code = extract_code(text)
    earliest = None
    for pat in _TS_PATTERNS:
        m = pat.search(code)
        if m and (earliest is None or m.start(1) < earliest[1]):
            earliest = (m.group(1), m.start(1))
    return (earliest[0] if earliest else None), False


def extract_name(language, text):
    return extract_name_python(text) if language == "python" else extract_name_ts(text)


# ─────────────────────────────────────────────────────────────
# 표기 판정
# ─────────────────────────────────────────────────────────────

_SNAKE_RE = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)+$")           # 밑줄로 구분된 소문자
_CAMEL_RE = re.compile(r"[a-z][a-z0-9]*([A-Z][a-zA-Z0-9]*)+$")    # 내부 대문자 험프, 밑줄 없음
_PASCAL_RE = re.compile(r"[A-Z][a-zA-Z0-9]*$")


def classify_case(name):
    """이름의 표기 스타일을 분류. 명세가 전부 여러 단어라 단일 단어(예: 'parse')는
    camel/snake 어느 쪽도 아닌 'other'로 본다(구분 불가 신호)."""
    if name is None:
        return None
    if _SNAKE_RE.match(name):
        return "snake"
    if _CAMEL_RE.match(name):
        return "camel"
    if _PASCAL_RE.match(name):
        return "pascal"
    return "other"


def is_compliant(name, required_style):
    return classify_case(name) == required_style


# ─────────────────────────────────────────────────────────────
# 재개 지원 — 완료된 (model, cell, seed) 건너뛰기
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
                done.add((r["model"], r["cell"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(out_path, record):
    with open(out_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# 한 세션 = 함수 10개 순차 생성
# ─────────────────────────────────────────────────────────────

def run_session(model, tokenizer, cell, seed, args, torch):
    """함수 10개를 순차 생성. 앞 생성물이 다음 턴 context에 누적된다."""
    language = cell["language"]
    required_style = cell["required_style"]

    messages = [{"role": "system", "content": system_prompt(language, required_style)}]

    # 대응 설계: 세션 seed를 한 번 고정한다. 같은 seed가 조건 간 공유된다.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    functions = []
    for pos, spec in enumerate(FUNCTION_SPECS, start=1):
        messages.append({"role": "user", "content": user_prompt(language, spec)})

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
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 생성물을 context에 누적 (H1 루프의 핵심)
        messages.append({"role": "assistant", "content": gen_text})

        name, parse_ok = extract_name(language, gen_text)
        case = classify_case(name)
        compliant = is_compliant(name, required_style)

        functions.append({
            "position": pos,
            "spec_id": spec["id"],
            "generated_name": name,
            "case": case,
            "compliant": compliant,
            "parse_ok": parse_ok,
            "raw": gen_text,
        })

    judged = [f for f in functions if f["generated_name"] is not None]
    n_compliant = sum(1 for f in judged if f["compliant"])
    pos1 = functions[0]

    return {
        "model": model.name_or_path,
        "cell": cell["cell"],
        "language": language,
        "required_style": required_style,
        "idiomatic": cell["idiomatic"],
        "seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_functions": len(functions),
        "n_judged": len(judged),
        "n_compliant": n_compliant,
        # 준수율은 이름 추출에 성공한 함수만 분모로 삼는다.
        "compliance_rate": (n_compliant / len(judged)) if judged else None,
        "position1_name": pos1["generated_name"],
        "position1_compliant": pos1["compliant"],
        "functions": functions,
    }


# ─────────────────────────────────────────────────────────────
# 요약 출력 (게이트 판정 참고용)
# ─────────────────────────────────────────────────────────────

def print_summary(out_path):
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

    print("\n=== 요약 (계획서 5.5 게이트: 55~85%) ===")
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["cell"]), []).append(r)

    for (model, cell), rs in sorted(groups.items()):
        # 전체 준수율: 판정된 함수 전부에 대한 마이크로 평균
        tot_j = sum(x["n_judged"] for x in rs)
        tot_c = sum(x["n_compliant"] for x in rs)
        overall = (tot_c / tot_j * 100) if tot_j else float("nan")
        # 위치 1 준수율: 주 지표
        p1 = [x["position1_compliant"] for x in rs if x["position1_name"] is not None]
        p1_rate = (sum(p1) / len(p1) * 100) if p1 else float("nan")
        gate = "PASS" if 55 <= overall <= 85 else ("천장" if overall > 85 else "바닥")
        short = model.split("/")[-1]
        print(f"  {short:32s} {cell:14s} "
              f"전체 {overall:5.1f}%  위치1 {p1_rate:5.1f}%  "
              f"(n={len(rs)}세션) [{gate}]")


# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Step 1 기저 준수율 smoke test")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n-sessions", type=int, default=10, help="셀당 세션 수")
    ap.add_argument("--base-seed", type=int, default=0,
                    help="session_idx k → seed = base_seed + k (조건 간 공유)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-only", action="store_true",
                    help="생성 없이 기존 JSONL 요약만 출력")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.summary_only:
        print_summary(args.out)
        return

    # 무거운 의존성은 여기서 import (요약만 볼 때는 불필요)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("경고: CUDA를 찾지 못했다. fp16 추론은 GPU 런타임이 필요하다.",
              file=sys.stderr)

    seeds = [args.base_seed + k for k in range(args.n_sessions)]
    done = load_done(args.out)
    print(f"기존 완료 세션: {len(done)}개 (건너뜀 대상)")

    for model_name in args.models:
        # 이 모델에서 돌 세션이 하나라도 있는지 먼저 확인 (없으면 로드 생략)
        pending = [
            (cell, seed)
            for cell in CELLS
            for seed in seeds
            if (model_name, cell["cell"], seed) not in done
        ]
        if not pending:
            print(f"\n[{model_name}] 모든 세션 완료됨 — 건너뜀")
            continue

        print(f"\n[{model_name}] 로드 중... (남은 세션 {len(pending)}개)")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        for cell, seed in pending:
            cell_label = cell["cell"]
            print(f"  실행: {cell_label} seed={seed} ...", end=" ", flush=True)
            with torch.no_grad():
                record = run_session(model, tokenizer, cell, seed, args, torch)
            append_jsonl(args.out, record)
            rate = record["compliance_rate"]
            rate_s = f"{rate * 100:.0f}%" if rate is not None else "n/a"
            print(f"준수율 {rate_s}, 위치1 {'O' if record['position1_compliant'] else 'X'}")

        # 다음 모델 로드 전 메모리 해제
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(args.out)


if __name__ == "__main__":
    main()
