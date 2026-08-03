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
# 설계 버전. 완료 판정·기록에 박아, prefix 길이 등 설계가 다른 옛 실행과
# 절대 섞이지 않게 한다. prefix 함수 수를 바꾸면 이 값도 반드시 바꿀 것.
DESIGN_VERSION = "v2_prefix12"
DEFAULT_OUT = os.path.join(RESULTS_DIR, "exp1_pilot_v2.jsonl")
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
    {
        "camel": (
            "def parseJsonPayload(raw):\n"
            '    """Decode a JSON request body into a dict."""\n'
            "    return json.loads(raw)"
        ),
        "snake": (
            "def parse_json_payload(raw):\n"
            '    """Decode a JSON request body into a dict."""\n'
            "    return json.loads(raw)"
        ),
    },
    {
        "camel": (
            "def buildQueryString(params):\n"
            '    """Join params into a URL query string."""\n'
            "    return '&'.join(f'{k}={v}' for k, v in params.items())"
        ),
        "snake": (
            "def build_query_string(params):\n"
            '    """Join params into a URL query string."""\n'
            "    return '&'.join(f'{k}={v}' for k, v in params.items())"
        ),
    },
    {
        "camel": (
            "def hashPassword(password):\n"
            '    """Return the SHA-256 hex digest of a password."""\n'
            "    return sha256(password.encode()).hexdigest()"
        ),
        "snake": (
            "def hash_password(password):\n"
            '    """Return the SHA-256 hex digest of a password."""\n'
            "    return sha256(password.encode()).hexdigest()"
        ),
    },
    {
        "camel": (
            "def loadConfigFile(path):\n"
            '    """Read and parse a config file from disk."""\n'
            "    with open(path) as f:\n"
            "        return json.load(f)"
        ),
        "snake": (
            "def load_config_file(path):\n"
            '    """Read and parse a config file from disk."""\n'
            "    with open(path) as f:\n"
            "        return json.load(f)"
        ),
    },
    {
        "camel": (
            "def mergeDictionaries(a, b):\n"
            '    """Merge two dicts, second wins on conflicts."""\n'
            "    return {**a, **b}"
        ),
        "snake": (
            "def merge_dictionaries(a, b):\n"
            '    """Merge two dicts, second wins on conflicts."""\n'
            "    return {**a, **b}"
        ),
    },
    {
        "camel": (
            "def filterActiveUsers(users):\n"
            '    """Keep only users whose account is active."""\n'
            "    return [u for u in users if u.active]"
        ),
        "snake": (
            "def filter_active_users(users):\n"
            '    """Keep only users whose account is active."""\n'
            "    return [u for u in users if u.active]"
        ),
    },
    {
        "camel": (
            "def serializeResponse(obj):\n"
            '    """Encode a response object as a JSON string."""\n'
            "    return json.dumps(obj)"
        ),
        "snake": (
            "def serialize_response(obj):\n"
            '    """Encode a response object as a JSON string."""\n'
            "    return json.dumps(obj)"
        ),
    },
    {
        "camel": (
            "def validateInputSchema(data, schema):\n"
            '    """Check that data matches the expected schema keys."""\n'
            "    return set(schema).issubset(data)"
        ),
        "snake": (
            "def validate_input_schema(data, schema):\n"
            '    """Check that data matches the expected schema keys."""\n'
            "    return set(schema).issubset(data)"
        ),
    },
]

BASE_SYSTEM = (
    "You are a senior Python engineer working in this project. "
    "Respond with exactly one Python function inside a single ```python code block, "
    "and nothing else."
)

# 지침 강도 4단계. baseline(V0)을 천장에서 떼어낼 수 있는지 보려는 축.
INSTRUCTION_RULE = {
    # 규칙 없음 — 순수 관습 baseline (Python이면 snake → camel 준수율 바닥 예상)
    "none": "",
    # 초약(mention) — 명령이 아니라 서술. '지나가듯 언급' 수준
    "mention": " This project generally follows camelCase naming for functions.",
    # 약한 지침 — 평범한 명령 한 문장
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
                # 완료 키에 설계 버전과 모델을 포함 — 설계·모델이 다른 실행은
                # 같은 (instruction, V, seed)라도 별개로 취급한다.
                done.add((r["design_version"], r["model"],
                          r["instruction"], r["violation_count"], r["seed"]))
            except (json.JSONDecodeError, KeyError):
                # design_version 없는 옛 레코드는 KeyError → 완료로 안 침 (섞임 방지)
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
        "design_version": DESIGN_VERSION,
        "prefix_size": len(PREFIX_FUNCS),
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


def _pos_stats(sessions, position):
    """해당 위치의 (판정가능 세션수, 준수 세션수, 전체 세션수).

    generated_name이 None(이름 추출 실패)인 세션은 '판정불가'로 분모에서 제외한다.
    즉 '위반'과 '판정불가'를 섞지 않는다.
    """
    total = len(sessions)
    judge = []
    for s in sessions:
        f = next((x for x in s.get("functions", []) if x["position"] == position), None)
        if f is not None and f.get("generated_name") is not None:
            judge.append(bool(f["compliant"]))
    return len(judge), sum(judge), total


def print_summary(out_path, instructions, violations):
    if not os.path.exists(out_path):
        return
    # 현재 설계 버전(DESIGN_VERSION) 레코드만 집계 — 옛 설계·다른 파일 섞임 방지
    rows = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("design_version") == DESIGN_VERSION:
                rows.append(r)
    if not rows:
        print(f"(design_version={DESIGN_VERSION} 레코드 없음: {out_path})")
        return

    models = sorted({r["model"].split("/")[-1] for r in rows})
    if len(models) > 1:
        print(f"주의: 여러 모델이 섞여 있음 {models} — 아래 표는 합산이다.")

    # (instruction, V) → 세션 레코드 리스트
    cell = {}
    for r in rows:
        cell.setdefault((r["instruction"], r["violation_count"]), []).append(r)

    print(f"\n=== 실험1 파일럿 요약 [{DESIGN_VERSION}] "
          f"{'/'.join(models)}, python camelCase / 위치1 준수율 ===")
    print("행=지침강도, 열=Vk(prefix 내 snake 함수 수).")
    print("값 = 위치1 camelCase 준수율%(판정가능 세션수). ?k = 이름 추출 실패(판정불가) k개\n")
    header = "  {:8s}".format("지침")
    for v in violations:
        header += "  {:^11s}".format(f"V{v}")
    print(header + "  | 해석")

    for instr in instructions:
        line = "  {:8s}".format(instr)
        rates = []
        for v in violations:
            nj, nc, nt = _pos_stats(cell.get((instr, v), []), 1)
            if nj:
                rate = nc / nj * 100
                rates.append(rate)
                s = "{:>3.0f}%(n{:d})".format(rate, nj)
                if nj < nt:
                    s += "?{:d}".format(nt - nj)  # 판정불가 개수
                line += "  {:^11s}".format(s)
            else:
                rates.append(None)
                line += "  {:^11s}".format("판정불가" if cell.get((instr, v)) else "--")
        print(line + f"  | {_interpret(rates)}")

    # 보조 — 위치별 준수율 (연쇄효과): 위치1이 흔들리거나 판정불가일 때 뒤(2·3)는 어땠나
    print("\n[보조] 위치별 준수율 (판정가능 분모 / 연쇄효과):")
    for instr in instructions:
        for pos in (1, 2, 3):
            line = "  {:8s} 위치{:d}".format(instr if pos == 1 else "", pos)
            for v in violations:
                nj, nc, _ = _pos_stats(cell.get((instr, v), []), pos)
                line += "  {:^11s}".format(
                    "{:>3.0f}%(n{:d})".format(nc / nj * 100, nj) if nj else "--")
            print(line)
        print()


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


def _load_rows(out_path):
    """현재 DESIGN_VERSION 레코드만 로드."""
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


def print_chain(out_path, instructions, violations):
    """조건부 연쇄효과: 위치1이 O였던 세션 vs X였던 세션에서 위치2·3 준수율.

    '첫 함수가 규칙을 어기면(스스로 snake를 뱉으면) 그게 다시 context에 쌓여
    뒤 함수도 따라 어기는가(자기 증폭)'를 직접 본다.
    """
    rows = _load_rows(out_path)
    if not rows:
        print(f"(design_version={DESIGN_VERSION} 레코드 없음: {out_path})")
        return
    cell = {}
    for r in rows:
        cell.setdefault((r["instruction"], r["violation_count"]), []).append(r)

    def rate(grp, pos):
        vals = [_f_at(s, pos)["compliant"] for s in grp
                if _f_at(s, pos) and _f_at(s, pos).get("generated_name") is not None]
        return (sum(vals) / len(vals) * 100, len(vals)) if vals else (None, 0)

    print("\n=== 연쇄효과: 위치1 결과에 따른 위치2·3 준수율 (판정가능만) ===")
    print("위치1을 O/X로 나눠, 뒤따르는 위치2·3의 camelCase 준수율을 본다.")
    print("(위치1=X 그룹에서 위치2·3도 낮으면 → 첫 위반이 자기 증폭한다는 신호)\n")
    for instr in instructions:
        for v in violations:
            sessions = cell.get((instr, v), [])
            o_grp, x_grp = [], []
            for s in sessions:
                f1 = _f_at(s, 1)
                if not f1 or f1.get("generated_name") is None:
                    continue
                (o_grp if f1["compliant"] else x_grp).append(s)
            if not o_grp and not x_grp:
                continue
            print(f"  [{instr} V{v}]")
            for label, grp in [("위치1=O", o_grp), ("위치1=X", x_grp)]:
                if not grp:
                    continue
                (r2, n2), (r3, n3) = rate(grp, 2), rate(grp, 3)
                r2s = f"{r2:.0f}%(n{n2})" if r2 is not None else "--"
                r3s = f"{r3:.0f}%(n{n3})" if r3 is not None else "--"
                print(f"    {label} (세션 n{len(grp)}) → 위치2 {r2s}, 위치3 {r3s}")
            print()


def main():
    ap = argparse.ArgumentParser(description="실험1 축소 파일럿 (3B, 지침강도×위반개수)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--instructions", nargs="+", default=["mention", "weak", "none"],
                    choices=["none", "mention", "weak", "strong"])
    ap.add_argument("--violations", nargs="+", type=int, default=[0, 2, 4, 8, 12])
    ap.add_argument("--n-seeds", type=int, default=5, help="조합당 세션(seed) 수")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--n-targets", type=int, default=3, help="prefix 뒤 생성 함수 수(주지표=1)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--chain", action="store_true",
                    help="위치1 O/X에 따른 위치2·3 조건부 연쇄효과도 출력")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    max_v = len(PREFIX_FUNCS)
    if any(v < 0 or v > max_v for v in args.violations):
        ap.error(f"--violations 는 0~{max_v} (prefix 함수 수) 범위여야 한다: {args.violations}")

    if args.summary_only:
        print_summary(args.out, args.instructions, args.violations)
        if args.chain:
            print_chain(args.out, args.instructions, args.violations)
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
        if (DESIGN_VERSION, args.model, instr, v, s) not in done
    ]
    print(f"완료 {len(done)}개, 남은 {len(pending)}개 세션")
    if not pending:
        print_summary(args.out, args.instructions, args.violations)
        if args.chain:
            print_chain(args.out, args.instructions, args.violations)
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
    if args.chain:
        print_chain(args.out, args.instructions, args.violations)


if __name__ == "__main__":
    main()
