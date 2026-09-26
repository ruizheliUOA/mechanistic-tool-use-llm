"""
acebench_readout_audit.py — Phase 4 parser validation (Step 6) + audit-sample tooling (Step 10).
=================================================================================================
Subcommands:
  unittests   run the locked parser unit-test suite 3x, check determinism, report pass/fail
  goldblind   static checks: parse() accepts only text; no metadata/id/filename tokens in rules
  audit-build build the stratified Step-10 audit sample from the parsed baseline details
Usage: python scripts/acebench_readout_audit.py <subcommand>
"""
from __future__ import annotations
import inspect, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import acebench_parse_outputs as PP

# ---------------------------------------------------------------- unit tests
# (expected labels follow the LOCKED rules; every ACEBench mode covered, EN+ZH, plus malformed,
#  mixed-mode, missing-field, wrong-argument, clarification, cannot-proceed, direct answers)
CASES = [
    # --- canonical official outputs, one per mode ---
    ("[get_weather(city='Beijing', date='2024-07-15')]", "tool_call", "bracket.call_list"),
    ("[search_flights(origin='PEK', dest='SHA'), book_hotel(city='Shanghai')]", "tool_call",
     "bracket.call_list"),
    ('["Missing necessary parameters (event_date) for the api (create_event)"]', "ask_user",
     "bracket.official.ask_user"),
    ('["There is incorrect value (2023-13-45) for the parameters (date) in the conversation '
     'history."]', "flag_param_error", "bracket.official.flag_param_error"),
    ('["Due to the limitations of the function, I cannot solve this problem."]', "cannot_comply",
     "bracket.official.cannot_comply"),
    # --- official templates with singular/case variations ---
    ('["Missing necessary parameter (city) for the api (get_weather)"]', "ask_user", None),
    ('["due to the limitations of the function, i cannot solve this problem."]', "cannot_comply",
     None),
    # --- markdown fence + role prefix wrappers ---
    ("```\n[get_time(zone='UTC')]\n```", "tool_call", None),
    ("assistant: [get_time(zone='UTC')]", "tool_call", None),
    # --- ZH free-text fallbacks (non-official formats -> ambiguous fallback tier) ---
    ("请问您想查询哪个城市的天气？请提供城市名称。", "ask_user", "nobracket.fallback.question_missing_info_cue"),
    ("您提供的日期格式不正确，参数 date 的取值无效。", "flag_param_error",
     "nobracket.fallback.invalid_param_cue"),
    ("很抱歉，提供的函数无法完成这个请求。", "cannot_comply", "nobracket.fallback.capability_refusal_cue"),
    # --- EN free-text fallbacks ---
    ("Could you provide the missing event date?", "ask_user", None),
    ("The value given for parameter 'age' is invalid; it violates the allowed range.",
     "flag_param_error", None),
    ("I am unable to fulfill this request with the available tools.", "cannot_comply", None),
    # --- bare call without brackets (flagged ambiguous) ---
    ("get_weather(city='Paris')", "tool_call", "nobracket.bare_call"),
    # --- malformed / missing-field / junk ---
    ("", "UNKNOWN", "empty"),
    ("   \n  ", "UNKNOWN", "empty"),
    ("Sure! Here is some general advice about your day.", "UNKNOWN", "nobracket.no_rule"),
    ("[get_weather(city='Paris'", "UNKNOWN", None),                # unbalanced bracket+paren
    ("[hello world this is not a call]", "UNKNOWN", "bracket.unclassified"),
    # pre-lock validation fix: real ACEBench has digit-leading function names (5G_...)
    ("[5G_City_Application_Summary(city='Shenzhen')]", "tool_call", "bracket.call_list"),
    ("[123abc(x=1)]", "tool_call", None),                          # digit-leading ids are legal here
    # --- wrong/missing arguments still parse as calls (validity of args is NOT a mode change) ---
    ("[get_weather()]", "tool_call", None),
    ("[get_weather(city=)]", "tool_call", None),   # syntactically consumed; args not validated
    # --- mixed-mode conflicts ---
    ('["Missing necessary parameters (a) for the api (f)" and also "There is incorrect value '
     '(x) for the parameters (b) in the conversation history."]', "flag_param_error", None),
    # call + special phrase in one block -> conservative UNKNOWN(conflict) — construct one where
    # the block is BOTH a full call list and contains an official phrase is impossible; test the
    # multi-special conflict flag instead (above) and a call followed by refusal prose:
    ("[get_weather(city='Paris')] Due to the limitations of the function, I cannot solve this "
     "problem.", "tool_call", "bracket.call_list"),  # first-block rule: call block wins; prose after
    # --- direct answer that mentions nothing actionable ---
    ("The capital of France is Paris.", "UNKNOWN", "nobracket.no_rule"),
    # --- clarification without question mark but with 请提供 (needs '?' per locked rule -> not ask) ---
    ("请提供城市名称。", "UNKNOWN", "nobracket.no_rule"),
    # --- multiple blocks: first balanced block wins ---
    ('["Due to the limitations of the function, I cannot solve this problem."] '
     '[get_weather(city="X")]', "cannot_comply", None),
]


def run_unittests():
    results = []
    for text, want_label, want_rule in CASES:
        outs = [PP.parse(text) for _ in range(3)]
        det = all(json.dumps(o, sort_keys=True) == json.dumps(outs[0], sort_keys=True)
                  for o in outs)
        got = outs[0]
        ok = (got["label"] == want_label and det and
              (want_rule is None or got["rule_path"] == want_rule))
        results.append({"text": text[:70], "want": want_label, "got": got["label"],
                        "rule": got["rule_path"], "deterministic": det, "pass": ok})
    n_pass = sum(r["pass"] for r in results)
    print(f"unit tests: {n_pass}/{len(results)} passed; deterministic on 3x repeats: "
          f"{all(r['deterministic'] for r in results)}")
    for r in results:
        if not r["pass"]:
            print("FAIL:", json.dumps(r, ensure_ascii=False))
    return results, n_pass == len(results)


def goldblind():
    """Static gold-blindness audit, scoped to the RULE functions (the CLI wrapper legitimately
    carries sample_id through for bookkeeping; it never feeds it to parse())."""
    sig = list(inspect.signature(PP.parse).parameters)
    rule_src = "".join(inspect.getsource(f) for f in
                       (PP.parse, PP._fallback, PP._official_hits, PP._first_block,
                        PP._parse_call_list))
    cli_src = inspect.getsource(PP.main)
    forbidden = re.compile(r"\bgold\b|\bsample_id\b|\bsource_file\b|\bmetadata\b|\bfilename\b|"
                           r"\bsplit\b|\blang\b", re.I)
    checks = {
        "parse_accepts_only_text": sig == ["text"],
        "rule_functions_free_of_gold_and_metadata_tokens": not forbidden.search(rule_src),
        "cli_passes_only_output_text_to_parse": 'parse(r.get("output"' in cli_src,
    }
    print(json.dumps(checks, indent=2))
    return checks, all(checks.values())


def audit_build(details_path: str, out_path: str, cap_unknown: int = 25):
    """Stratified Step-10 audit sample from parsed baseline details."""
    import random
    rows = [json.loads(l) for l in open(details_path, encoding="utf-8")]
    rng = random.Random(7)
    sample, seen = [], set()

    def add(rs, tag, k):
        rs = [r for r in rs if r["sample_id"] not in seen]
        rng.shuffle(rs)
        for r in rs[:k]:
            sample.append({**r, "audit_stratum": tag})
            seen.add(r["sample_id"])

    golds = sorted({r["gold"] for r in rows})
    preds = sorted({r["pred"] for r in rows})
    for g in golds:                       # major confusion cells: 5 per (gold, pred) pair seen
        for p in preds:
            cell = [r for r in rows if r["gold"] == g and r["pred"] == p]
            if len(cell) >= 3:
                add(cell, f"cell:{g}->{p}", 5)
    add([r for r in rows if r["pred"] == "UNKNOWN"], "unknown", cap_unknown)
    add([r for r in rows if r.get("conflict")], "conflict", 10)
    add([r for r in rows if r.get("ambiguous")], "ambiguous_fallback", 10)
    add([r for r in rows if str(r.get("rule_path", "")).startswith("bracket.unclassified")],
        "bracketed_unclassified", 10)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"audit sample: {len(sample)} rows -> {out_path}")
    from collections import Counter
    print(json.dumps(Counter(r["audit_stratum"] for r in sample), indent=1, ensure_ascii=False,
                     default=str))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "unittests"
    if cmd == "unittests":
        _, ok = run_unittests()
        sys.exit(0 if ok else 2)
    elif cmd == "goldblind":
        _, ok = goldblind()
        sys.exit(0 if ok else 2)
    elif cmd == "audit-build":
        audit_build(sys.argv[2], sys.argv[3])
