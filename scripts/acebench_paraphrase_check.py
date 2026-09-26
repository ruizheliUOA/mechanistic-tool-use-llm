"""
acebench_paraphrase_check.py — Phase 4 pre-registered robustness gate for R0 CONDITIONAL PASS.
===============================================================================================
Locked verdict rule (PHASE4_GENERATION_PROTOCOL.md §17): under CONDITIONAL PASS, intervention
eligibility requires "a second, semantically equivalent (non-official-wording) system prompt on a
100-row stratified subset with prediction agreement >= 0.80" versus the locked baseline.

This runs that check ONLY. It re-uses the exact locked model loader / generation / parser from
`acebench_generation_baseline.py`; the ONLY change is the SYSTEM prompt, reworded into different
prose while (a) preserving the task semantics and (b) keeping the three official OUTPUT-FORMAT
template strings verbatim (those strings are the answer schema the parser reads, not "wording") and
(c) keeping the user prompt identical — so system-prompt wording is the single manipulated variable.

NOT an intervention: no activations, routers, directions, steering, or SAKIKO. A different prompt is
a black-box robustness probe of the readout, exactly as pre-registered.

Output -> final/results/acebench_generation_readout/paraphrase_check/
Raw generations -> gitignored .cache/acebench_phase4/generation_details_paraphrase.jsonl
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acebench_generation_baseline as GB  # locked code path
from acebench_parse_outputs import parse

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "final/results/acebench_generation_readout"
OUT = RES / "paraphrase_check"
FULL_DET = ROOT / ".cache/acebench_phase4/generation_details_full.jsonl"
PER_MODE = 25          # balanced 100-row subset: 25 per gold mode
AGREE_GATE = 0.80

# --- Paraphrased SPECIAL-DATA system prompts: prose reworded; the three bracketed output-format
#     template strings are kept VERBATIM (answer schema); {time}/{function} placeholders preserved. ---
PARA_SYS_EN = """You are acting as the assistant. Using the API definitions given below and the \
dialogue so far (turns 1 through t), decide what the assistant should do at turn t+1. Exactly one \
of these two situations holds:

Situation 1 — the user's request is clear and can be satisfied with the listed candidate functions:
   - If a parameter's value format is not constrained by its description, copy the value straight \
from the user's own wording.
   - When more than one of the candidate functions is needed to satisfy the request, emit all of \
them.
   - Write the call(s) as [ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', \
key2='value2', ...), ...], substituting the real API name, the real parameter names, and the real \
values. Start the output with "[" and finish it with "]", and put nothing else in the output.

Situation 2 — the request is unclear, incomplete, contains a wrong value, or asks for something the \
functions cannot do. You must state the problem plainly, checking the cases in this order:
   (1) The user supplied the key detail the API needs, but a value's type or form does not match \
the API definition: ["There is incorrect value (value) for the parameters (key) in the conversation history."]
   (2) The user is missing a key detail the API needs: ["Missing necessary parameters (key1, key2, ...) for the api (ApiName)"], with the missing parameter names and the actual API name filled in.
   (3) The request is outside what your current APIs can do: ["Due to the limitations of the function, I cannot solve this problem."]
   Evaluate (1) first; only if it does not apply consider (2), then (3). Take care to distinguish \
(1) from (2).

{time}

Roles:
user: the user
assistant: the assistant that issues the API requests

API definitions:
{function}"""

PARA_SYS_ZH = """你现在担任assistant。请依据下面给出的API说明以及已有对话（第1步到第t步），判断assistant在第t+1步\
该做什么。下面两种情形必居其一：

情形一——用户需求清楚、并且可以用候选函数列表完成：
若某参数的取值格式未受描述限定，就直接沿用用户的原话作为其取值。
当候选函数里不止一个都需要用到时，请把它们统统输出。
请把调用写成[ApiName(key1='value1', key2='value2', ...), ApiName(key1='value1', key2='value2', ...)...]的形式，\
其中ApiName、key、value都替换成真实的名称和取值。输出要以"["开头、以"]"收尾，其余内容一概不要。

情形二——用户信息含糊、残缺、有误，或者请求超出了这些函数的能力，你就要把问题清楚地指出来，并按下述次序判断：
(1) 用户已经给了调用API所需的关键信息，但某个取值的类型或形式和API定义对不上：["There is incorrect value (value) for the parameters (key) in the conversation history."]
(2) 用户缺了API所需的关键信息：["Missing necessary parameters (key1，key2...) for the api (ApiName)"]，把key、ApiName换成缺失的参数名和真实API名。
(3) 请求超出你当前API的能力：["Due to the limitations of the function, I cannot solve this problem."]
请先判断(1)，不成立再依次看(2)、(3)；注意分清(1)和(2)。

{time}

角色：
user: 用户
assistant: 发出API请求的助手

API说明：
{function}"""

PARA = {"en": PARA_SYS_EN, "zh": PARA_SYS_ZH}


def build_messages_para(row):
    """Same as GB.build_messages but with the paraphrased SYSTEM prompt; user prompt unchanged."""
    _, user_t = GB.load_official_prompts()[row["lang"]]
    system = (PARA[row["lang"]].replace("{function}",
                                        json.dumps(row["functions"], ensure_ascii=False))
              .replace("{time}", row.get("time") or ""))
    q = row["question"]
    if not q.startswith("user:"):
        q = "user: " + q
    user = user_t.replace("{question}", q)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def pick_subset(rows):
    """Balanced 100-row stratified subset: 25 per gold mode, ~half EN half ZH, lowest sample_id."""
    out = []
    for mode in ["tool_call", "ask_user", "flag_param_error", "cannot_comply"]:
        per_lang = {"en": PER_MODE // 2 + PER_MODE % 2, "zh": PER_MODE // 2}  # 13 en / 12 zh
        for lang, q in per_lang.items():
            pick = sorted([r for r in rows if r["gold_mode"] == mode and r["lang"] == lang],
                          key=lambda r: r["sample_id"])[:q]
            out.extend(pick)
    return sorted(out, key=lambda r: r["sample_id"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(GB.DATA, encoding="utf-8")]
    base = {json.loads(l)["sample_id"]: json.loads(l)["pred"]
            for l in open(FULL_DET, encoding="utf-8")}
    subset = pick_subset(rows)
    model, tok = GB.load_model()

    det = open(ROOT / ".cache/acebench_phase4/generation_details_paraphrase.jsonl", "w",
               encoding="utf-8")
    recs, t0 = [], time.time()
    for k, r in enumerate(subset):
        text, n_in, n_gen, status = GB.generate(model, tok, build_messages_para(r))
        p = parse(text if text is not None else "")
        rec = {"sample_id": r["sample_id"], "lang": r["lang"], "gold": r["gold_mode"],
               "base_pred": base.get(r["sample_id"]), "para_pred": p["label"],
               "rule_path": p["rule_path"], "finish": status, "output": text}
        recs.append(rec)
        det.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (k + 1) % 20 == 0:
            det.flush()
            GB.log.info("%d/%d", k + 1, len(subset))
    det.close()

    n = len(recs)
    agree = sum(r["base_pred"] == r["para_pred"] for r in recs)
    # agreement excluding rows where either side is UNKNOWN (label-on-label stability)
    known = [r for r in recs if r["base_pred"] != "UNKNOWN" and r["para_pred"] != "UNKNOWN"]
    agree_known = sum(r["base_pred"] == r["para_pred"] for r in known)
    by_mode = {}
    for m in ["tool_call", "ask_user", "flag_param_error", "cannot_comply"]:
        rs = [r for r in recs if r["gold"] == m]
        by_mode[m] = {"n": len(rs),
                      "agreement": round(sum(r["base_pred"] == r["para_pred"] for r in rs) /
                                         len(rs), 4) if rs else None}
    disagreements = [{"sample_id": r["sample_id"], "lang": r["lang"], "gold": r["gold"],
                      "base": r["base_pred"], "para": r["para_pred"], "finish": r["finish"]}
                     for r in recs if r["base_pred"] != r["para_pred"]]
    summary = {
        "n": n, "agreement_all": round(agree / n, 4),
        "agreement_known_only": round(agree_known / len(known), 4) if known else None,
        "n_known_pairs": len(known),
        "gate_threshold": AGREE_GATE,
        "PASS": (agree / n) >= AGREE_GATE,
        "per_mode_agreement": by_mode,
        "base_pred_dist": dict(Counter(r["base_pred"] for r in recs)),
        "para_pred_dist": dict(Counter(r["para_pred"] for r in recs)),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "note": "system-prompt paraphrase only; user prompt + output-format templates + model + "
                "decoding identical to the locked baseline",
    }
    json.dump(summary, open(OUT / "paraphrase_agreement.json", "w"), indent=2, ensure_ascii=False)
    GB.log.info("PARAPHRASE agreement_all=%.4f known=%.4f PASS=%s (%d/%d)",
                summary["agreement_all"], summary["agreement_known_only"] or -1,
                summary["PASS"], agree, n)


if __name__ == "__main__":
    main()
