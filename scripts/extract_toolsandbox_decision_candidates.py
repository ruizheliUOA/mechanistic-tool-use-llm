"""
extract_toolsandbox_decision_candidates.py
==========================================
Extract SAKIKO-style *decision* candidate rows from ToolSandbox scenario
definitions, for MANUAL REVIEW. This does NOT finalize labels.

ToolSandbox scenarios are Python source (no JSON data files). They are
parsed with the `ast` module (dependency-free — ToolSandbox itself is NOT
imported and NOT modified). Each `ScenarioExtension(...)` is a mostly-literal
data structure containing:
    - name
    - messages           : user/system turns  (the query lives here)
    - tool_allow_list     : tools available at the decision point
    - tool_deny_list      : tools explicitly removed
    - milestones          : what SHOULD happen   (Milestone)
    - minefields          : what must NOT happen  (Minefield)
    - categories          : ScenarioCategories tags

Derived-label heuristics (CONSERVATIVE — candidate only, reviewer decides):
    tool_call         : source file single/multiple_tool_call AND the first
                        milestone encodes a tool effect / tool_trace.
    request_for_info  : the FIRST milestone is an AGENT->USER message (the agent
                        must ask the user) occurring before any tool call
                        (chiefly multiple_user_turn).
    cannot_answer     : insufficient_information scenario with empty milestones
                        and a minefield forbidding the tool, with the user
                        instruction stating no further information is available.
    direct            : no tool effect and no ask required (rare — flagged, not forced).

Labels emitted here are `candidate_label` with `label_confidence`; `reviewer_label`
is left EMPTY. Nothing here is a finalized SAKIKO label.

Usage:
  /path/to/project/.venv/bin/python scripts/extract_toolsandbox_decision_candidates.py \\
      --toolsandbox /path/to/project/ToolSandbox \\
      --out data/annotation/toolsandbox_decision_candidates.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

SCENARIO_FILES = {
    "single_tool_call":        "single_tool_call",
    "multiple_tool_call":      "multiple_tool_call",
    "multiple_user_turn":      "multiple_user_turn",
    "insufficient_information": "insufficient_information",
}

# ── AST literal helpers ───────────────────────────────────────────────────────

def attr_name(node) -> str | None:
    """RoleType.USER -> 'USER';  ScenarioCategories.X -> 'X'."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def str_of(node) -> str | None:
    """Extract a string from a Constant, or a BinOp/concatenation of string
    constants and Names (e.g. USER_INSTRUCTION + "...")."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = str_of(node.left)
        right = str_of(node.right)
        parts = [p for p in (left, right) if p]
        return "".join(parts) if parts else None
    if isinstance(node, ast.Name):
        # symbolic constant such as USER_INSTRUCTION — mark, do not inline
        return f"<{node.id}>"
    if isinstance(node, ast.JoinedStr):  # f-string
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            else:
                out.append("{...}")
        return "".join(out)
    return None


def list_of_str(node) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    out = []
    for el in node.elts:
        s = str_of(el)
        if s is not None:
            out.append(s)
    return out


def kwargs_of(call: ast.Call) -> dict:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


# ── message parsing ───────────────────────────────────────────────────────────

def parse_messages(node) -> list[dict]:
    """Return list of {sender, recipient, content} for Dict-literal message
    entries. Skips Starred few-shot splats (recorded as context marker)."""
    msgs = []
    if not isinstance(node, ast.List):
        return msgs
    for el in node.elts:
        if isinstance(el, ast.Starred):
            msgs.append({"sender": "FEWSHOT", "recipient": "FEWSHOT",
                         "content": "<few_shot_examples_splat>"})
            continue
        if not isinstance(el, ast.Dict):
            continue
        d = {}
        for k, v in zip(el.keys, el.values):
            key = str_of(k)
            if key in ("sender", "recipient"):
                d[key] = attr_name(v)
            elif key == "content":
                d[key] = str_of(v)
        if d.get("sender") or d.get("content"):
            msgs.append(d)
    return msgs


# ── milestone / minefield evidence ────────────────────────────────────────────

def _find_dataframe_dicts(node) -> list[dict]:
    """Walk a Milestone/Minefield Call subtree, returning each pl.DataFrame({...})
    payload as a {key: str_value} dict (sender/recipient/content/tool_trace…)."""
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            is_df = (isinstance(fn, ast.Attribute) and fn.attr == "DataFrame")
            if is_df and sub.args and isinstance(sub.args[0], ast.Dict):
                payload = {}
                for k, v in zip(sub.args[0].keys, sub.args[0].values):
                    key = str_of(k)
                    if key is None:
                        continue
                    # tool_trace is often json.dumps({...}) — pull tool_name
                    if key == "tool_trace":
                        payload["tool_trace"] = _extract_tool_name(v)
                    elif key in ("sender", "recipient"):
                        # RoleType.AGENT etc. — resolve enum attribute name
                        payload[key] = attr_name(v)
                    else:
                        payload[key] = str_of(v)
                found.append(payload)
    return found


def _extract_tool_name(node) -> str:
    """From json.dumps({'tool_name': 'X', ...}) extract X; else 'tool_call'."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
            for k, v in zip(sub.keys, sub.values):
                if str_of(k) == "tool_name":
                    return str_of(v) or "tool_call"
    return "tool_call"


def summarize_constraints(milestone_call) -> list[dict]:
    """Classify each dataframe dict inside a milestone/minefield."""
    out = []
    for df in _find_dataframe_dicts(milestone_call):
        sender = df.get("sender")
        recipient = df.get("recipient")
        if "tool_trace" in df:
            out.append({"kind": "tool_call", "tool": df["tool_trace"]})
        elif sender == "AGENT" and recipient == "USER":
            out.append({"kind": "agent_to_user", "content": df.get("content")})
        elif sender == "AGENT" and recipient == "EXECUTION_ENVIRONMENT":
            out.append({"kind": "tool_invocation", "tool": df.get("content")})
        elif recipient and recipient not in ("USER", "AGENT", "SANDBOX"):
            out.append({"kind": "state_change", "db": recipient})
        else:
            # database state change (e.g. SETTING cellular=False) — a tool effect
            keys = [k for k in df.keys() if k not in ("sender", "recipient", "content")]
            if keys:
                out.append({"kind": "state_change", "fields": keys})
            elif df.get("content"):
                out.append({"kind": "message", "sender": sender,
                            "recipient": recipient, "content": df.get("content")})
    return out


def parse_milestones(node) -> list[list[dict]]:
    """Return list (per milestone) of constraint summaries."""
    out = []
    if not isinstance(node, ast.List):
        return out
    for el in node.elts:
        if isinstance(el, ast.Call):
            out.append(summarize_constraints(el))
    return out


# ── classification heuristic ──────────────────────────────────────────────────

def classify(source: str, milestones: list, minefields: list,
             tool_allow: list, tool_deny: list, instruction: str) -> tuple[str, str, str]:
    """Return (candidate_label, confidence, notes). Conservative."""
    flat_ms = [c for ms in milestones for c in ms]
    first_ms = milestones[0] if milestones else []
    first_kinds = [c["kind"] for c in first_ms]

    has_tool = any(c["kind"] in ("tool_call", "tool_invocation", "state_change")
                   for c in flat_ms)
    first_is_ask = bool(first_ms) and all(c["kind"] in ("agent_to_user", "message")
                                          for c in first_ms) \
                   and any(c["kind"] == "agent_to_user" for c in first_ms)
    no_info = "do not have" in (instruction or "").lower() \
              or "no more information" in (instruction or "").lower() \
              or "don't have" in (instruction or "").lower()

    # cannot_answer: insufficient_information, empty milestones, minefield forbids tool
    if source == "insufficient_information":
        if not milestones and minefields:
            conf = "high" if no_info else "medium"
            return ("cannot_answer", conf,
                    "insufficient_information: empty milestones + minefield forbids tool; "
                    f"user-cannot-supply={no_info}")
        return ("cannot_answer", "low",
                "insufficient_information but milestones present — review (may be partial-RFI)")

    # request_for_info: first gold action is asking the user
    if first_is_ask and source in ("multiple_user_turn", "single_tool_call",
                                    "multiple_tool_call"):
        return ("request_for_info", "medium",
                "first milestone is AGENT->USER message before any tool call")

    # tool_call: a tool effect is required
    if source in ("single_tool_call", "multiple_tool_call") and has_tool:
        conf = "high" if source == "single_tool_call" else "medium"
        note = ("single-step tool decision" if source == "single_tool_call"
                else "multi-step; first local decision is a tool call")
        return ("tool_call", conf, note)

    # multiple_user_turn that resolves via a tool (not an ask-first)
    if source == "multiple_user_turn" and has_tool:
        return ("tool_call", "low",
                "multiple_user_turn resolved by tool call (not ask-first) — review for RFI")

    return ("direct", "low", "no tool effect and no ask detected — review (possible direct)")


# ── per-file extraction ───────────────────────────────────────────────────────

def extract_file(path: Path, source: str) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rows = []
    counter = 0
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "ScenarioExtension"):
            continue
        kw = kwargs_of(call)
        name = str_of(kw.get("name")) or f"{source}_{counter}"
        messages = parse_messages(kw.get("messages"))
        tool_allow = list_of_str(kw.get("tool_allow_list")) if "tool_allow_list" in kw else []
        tool_deny = list_of_str(kw.get("tool_deny_list")) if "tool_deny_list" in kw else []
        milestones = parse_milestones(kw.get("milestones")) if "milestones" in kw else []
        minefields = parse_milestones(kw.get("minefields")) if "minefields" in kw else []

        # query = last USER->AGENT message; instruction = SYSTEM->USER content
        user_turns = [m for m in messages if m.get("sender") == "USER"
                      and m.get("recipient") == "AGENT"]
        sys_turns = [m for m in messages if m.get("sender") == "SYSTEM"
                     and m.get("recipient") == "USER"]
        query = user_turns[-1]["content"] if user_turns else ""
        instruction = sys_turns[-1]["content"] if sys_turns else ""
        # dialogue context = everything before the final user query
        ctx_parts = []
        for m in messages:
            if m is (user_turns[-1] if user_turns else None):
                continue
            c = m.get("content")
            if c and c != "<few_shot_examples_splat>":
                ctx_parts.append(f"[{m.get('sender')}→{m.get('recipient')}] {c[:120]}")
            elif c == "<few_shot_examples_splat>":
                ctx_parts.append("[few-shot examples present]")
        dialogue_context = " || ".join(ctx_parts[-4:])

        label, conf, notes = classify(source, milestones, minefields,
                                       tool_allow, tool_deny, instruction)

        # evidence strings
        ms_ev = json.dumps(milestones, ensure_ascii=False)[:600]
        mf_tools = sorted({c.get("tool") for ms in minefields for c in ms
                           if c.get("tool")})
        mf_ev = (f"minefield_forbids={mf_tools}; tool_deny={tool_deny}"
                 if (mf_tools or tool_deny) else "")
        observed_tool_call = ""
        for ms in milestones:
            for c in ms:
                if c["kind"] in ("tool_call", "tool_invocation") and c.get("tool"):
                    observed_tool_call = c["tool"]; break
            if observed_tool_call:
                break

        counter += 1
        rows.append({
            "sample_id": f"tsd_{source[:4]}_{counter:03d}",
            "source_case_id": name,
            "turn_id": "u1",
            "candidate_label": label,
            "label_confidence": conf,
            "dialogue_context": dialogue_context,
            "current_user_query": query,
            "available_tools": ", ".join(tool_allow) if tool_allow else "(inherited/base)",
            "observed_tool_call": observed_tool_call,
            "milestone_or_state_evidence": ms_ev,
            "minefield_or_insufficient_info_evidence": (mf_ev + (
                f"; instruction={instruction[:160]}" if instruction else "")),
            "derivation_notes": notes,
            "reviewer_label": "",
            "reviewer_notes": "",
            "_source_file": source,
        })
    return rows


COLUMNS = [
    "sample_id", "source_case_id", "turn_id", "candidate_label", "label_confidence",
    "dialogue_context", "current_user_query", "available_tools", "observed_tool_call",
    "milestone_or_state_evidence", "minefield_or_insufficient_info_evidence",
    "derivation_notes", "reviewer_label", "reviewer_notes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolsandbox", default="/path/to/project/ToolSandbox")
    ap.add_argument("--out", default="data/annotation/toolsandbox_decision_candidates.csv")
    args = ap.parse_args()

    scen_dir = Path(args.toolsandbox) / "tool_sandbox" / "scenarios"
    all_rows = []
    per_source_counts = {}
    for source, stem in SCENARIO_FILES.items():
        fpath = scen_dir / f"{stem}_scenarios.py"
        rows = extract_file(fpath, source)
        all_rows.extend(rows)
        per_source_counts[source] = len(rows)
        print(f"[{source}] parsed {len(rows)} ScenarioExtensions from {fpath.name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # label tally
    from collections import Counter
    label_conf = Counter((r["candidate_label"], r["label_confidence"]) for r in all_rows)
    label_total = Counter(r["candidate_label"] for r in all_rows)
    print(f"\nWrote {len(all_rows)} candidate rows -> {out}")
    print("Per-source:", per_source_counts)
    print("Candidate-label totals:", dict(label_total))
    print("By (label, confidence):")
    for (lbl, conf), n in sorted(label_conf.items()):
        print(f"   {lbl:18s} {conf:7s} {n}")

    # also dump a small JSON tally next to CSV for the feasibility report
    tally = {
        "per_source": per_source_counts,
        "label_total": dict(label_total),
        "by_label_confidence": {f"{l}|{c}": n for (l, c), n in label_conf.items()},
        "n_total": len(all_rows),
    }
    (out.parent / "toolsandbox_candidate_tally.json").write_text(
        json.dumps(tally, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Tally -> {out.parent / 'toolsandbox_candidate_tally.json'}")


if __name__ == "__main__":
    main()
