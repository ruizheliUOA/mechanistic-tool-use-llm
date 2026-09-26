"""
acebench_parse_outputs.py — Phase 4 deterministic parser for ACEBench generation outputs.
==========================================================================================
PARSER_VERSION 1.0 (locked with PHASE4_GENERATION_PROTOCOL.md §14–16; sha256 recorded in
READOUT_LOCK_MANIFEST.json).

Contract:
  parse(text: str) -> dict     # the ONLY input is the raw generated text (gold-blind by design;
                               # no sample id, no file name, no metadata is even accepted)

Returned dict:
  label               tool_call | ask_user | flag_param_error | cannot_comply | UNKNOWN
  rule_path           which locked rule produced the label (audit trail)
  ambiguous           True when a conservative fallback rule fired (non-official format)
  conflict            True when contradictory modes were expressed (label forced to UNKNOWN
                      only for call+special conflicts; multi-special resolves by official priority)
  unparseable_reason  set when label == UNKNOWN
  calls               extracted call names when a call list was parsed
  n_blocks            number of bracketed [...] blocks found

Locked rule order (protocol §14):
  0. normalize (strip whitespace / markdown fences / leading role tags)
  1. take the FIRST balanced [...] block
  2. inside it, official patterns in official priority:
       "incorrect value"            -> flag_param_error
       "missing necessary param"    -> ask_user
       "due to the limitations" / "cannot solve this problem" -> cannot_comply
     (>=2 distinct official patterns: resolve by this priority, conflict=True, ambiguous=True)
  3. else full-block call-list parse Name(k=v, ...) [, Name(...)]* -> tool_call
  4. a full call parse together with an official pattern -> UNKNOWN(conflict)   (cannot co-occur
     in practice; kept for safety)
  5. bracketed but unclassified -> fallback cues inside the block, else
     UNKNOWN(bracketed_unclassified)
  6. no bracket: conservative bilingual fallbacks (all ambiguous=True), same priority:
       value-error cues -> flag_param_error; missing-info question -> ask_user;
       capability-refusal cues -> cannot_comply; bare call syntax -> tool_call
  7. else UNKNOWN(no_rule)

No rule reads dataset metadata, ids, or file names. UNKNOWN is a first-class outcome.
"""
from __future__ import annotations
import json, re, sys

PARSER_VERSION = "1.0"
LABELS = ["tool_call", "ask_user", "flag_param_error", "cannot_comply"]

# --- official special-response patterns (EN templates are used in BOTH language prompts) ---
RE_INCORRECT = re.compile(r"incorrect\s+value", re.I)
RE_MISSING = re.compile(r"missing\s+necessary\s+param", re.I)
RE_LIMIT = re.compile(
    r"due\s+to\s+the\s+limitations?\s+of\s+the\s+function|cannot\s+solve\s+this\s+problem", re.I)

# --- conservative fallback cues (non-official formats; always ambiguous=True) ---
RE_FB_PARAM_WORD = re.compile(r"param|argument|value|参数|取值", re.I)
RE_FB_INVALID = re.compile(
    r"invalid|incorrect|not\s+valid|violat|wrong\s+(value|format)|无效|不正确|不符合|格式(不|错误)", re.I)
RE_FB_ASK = re.compile(
    r"provide|missing|specify|clarify|need\s+(to\s+know|more)|could\s+you|请提供|缺少|需要您|请问|补充|告诉我", re.I)
RE_FB_CANNOT = re.compile(r"cannot|can['’]t|unable\s+to|not\s+(possible|able)|无法|不能", re.I)
RE_FB_CAPABILITY = re.compile(r"function|api|tool|capabilit|request|函数|功能|工具|请求", re.I)
RE_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.M)
RE_ROLE = re.compile(r"^\s*(assistant|Assistant)\s*[:：]\s*")
# NOTE (pre-lock validation fix): identifiers may start with a digit — the real ACEBench schema
# contains function names like 5G_City_Application_Summary.
RE_CALL_HEAD = re.compile(r"^[A-Za-z0-9_][\w\.]*\s*\(")


def _first_block(text: str):
    """First balanced [...] block; if '[' present but unbalanced, take to end (flagged).
    Returns (block_text_without_outer_brackets, n_opening_brackets_total, unbalanced)."""
    start = text.find("[")
    n_open = text.count("[")
    if start == -1:
        return None, 0, False
    depth, i, end = 0, start, None
    while i < len(text):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end is None:
        return text[start + 1:].strip(), n_open, True
    return text[start + 1:end].strip(), n_open, False


def _parse_call_list(block: str):
    """True + call names iff the block is exactly a comma-separated list of Name(...) calls."""
    s = block.strip()
    if not s:
        return False, []
    names, i, n = [], 0, len(s)
    while i < n:
        m = re.match(r"\s*([A-Za-z0-9_][\w\.]*)\s*\(", s[i:])
        if not m:
            return False, []
        names.append(m.group(1))
        j = i + m.end()            # position just after '('
        depth, quote = 1, None
        while j < n and depth > 0:
            c = s[j]
            if quote:
                if c == quote and s[j - 1] != "\\":
                    quote = None
            elif c in "'\"":
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if depth != 0:
            return False, []
        m2 = re.match(r"\s*(,)?\s*", s[j:])
        j2 = j + m2.end()
        if s[j2:].strip() and m2.group(1) is None:
            return False, []       # junk after a call without a comma separator
        i = j2
        if not s[i:].strip():
            break
    return (len(names) > 0), names


def _official_hits(block: str):
    hits = []
    if RE_INCORRECT.search(block):
        hits.append("flag_param_error")
    if RE_MISSING.search(block):
        hits.append("ask_user")
    if RE_LIMIT.search(block):
        hits.append("cannot_comply")
    return hits


def _fallback(t: str):
    """Conservative bilingual behavioral cues, locked priority. Returns (label, cue) or None."""
    if RE_FB_INVALID.search(t) and RE_FB_PARAM_WORD.search(t):
        return ("flag_param_error", "invalid_param_cue")
    if ("?" in t or "？" in t) and RE_FB_ASK.search(t):
        return ("ask_user", "question_missing_info_cue")
    if RE_FB_CANNOT.search(t) and RE_FB_CAPABILITY.search(t):
        return ("cannot_comply", "capability_refusal_cue")
    return None


def parse(text: str) -> dict:
    out = {"label": "UNKNOWN", "rule_path": None, "ambiguous": False, "conflict": False,
           "unparseable_reason": None, "calls": [], "n_blocks": 0,
           "parser_version": PARSER_VERSION}
    if text is None or not str(text).strip():
        out.update(rule_path="empty", unparseable_reason="empty_generation")
        return out
    t = RE_ROLE.sub("", RE_FENCE.sub("", str(text)).strip()).strip()

    block, n_blocks, unbalanced = _first_block(t)
    out["n_blocks"] = n_blocks
    if block is not None:
        hits = _official_hits(block)
        is_call, names = (False, []) if unbalanced else _parse_call_list(block)
        if hits and is_call:
            out.update(label="UNKNOWN", rule_path="bracket.call_and_special",
                       conflict=True, unparseable_reason="conflicting_modes", calls=names)
            return out
        if hits:
            if len(set(hits)) > 1:
                out.update(conflict=True, ambiguous=True)
            for lab in ("flag_param_error", "ask_user", "cannot_comply"):  # official priority
                if lab in hits:
                    out.update(label=lab, rule_path=f"bracket.official.{lab}")
                    return out
        if is_call:
            out.update(label="tool_call", rule_path="bracket.call_list", calls=names)
            return out
        fb = _fallback(block)
        if fb:
            out.update(label=fb[0], rule_path=f"bracket.fallback.{fb[1]}", ambiguous=True)
            return out
        out.update(rule_path="bracket.unclassified",
                   unparseable_reason="bracketed_unclassified")
        return out

    fb = _fallback(t)
    if fb:
        out.update(label=fb[0], rule_path=f"nobracket.fallback.{fb[1]}", ambiguous=True)
        return out
    if RE_CALL_HEAD.match(t):
        ok, names = _parse_call_list(t)
        if ok:
            out.update(label="tool_call", rule_path="nobracket.bare_call",
                       ambiguous=True, calls=names)
            return out
    out.update(rule_path="nobracket.no_rule", unparseable_reason="no_rule")
    return out


def main():
    """CLI: parse a raw-generations jsonl (fields: sample_id, output) -> parsed jsonl on stdout."""
    src = sys.argv[1]
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        p = parse(r.get("output", ""))
        print(json.dumps({"sample_id": r.get("sample_id"), **p}, ensure_ascii=False))


if __name__ == "__main__":
    main()
