# Human annotator instructions — FINAL

You are labelling 144 rows. Two annotators work **independently**. Do not discuss
rows, do not compare sheets, and do not look at the other annotator's file until
both are filed.

Your file is either `ADJUDICATION_A1_BLINDED.csv` or `ADJUDICATION_A2_BLINDED.csv`.
Fill only your own.

## The task

Each row shows a workspace assistant about to respond. Decide the **single first
action** the assistant should take. Exactly one of four:

| Label | Take it when |
|---|---|
| `tool_call` | A listed tool can do what was asked, every required argument is present or derivable from the visible context, and permission and policy allow it. |
| `request_for_info` | A tool could do it, but at least one **required** argument is missing and cannot be derived from the visible context. The single blocking need is information from the user. |
| `cannot_answer` | The action cannot proceed at all: no listed tool is capable of it, or permission is denied, or policy forbids it, or required external state is inaccessible. The blocker is not missing user information. |
| `direct_answer` | No action is being requested, or the answer is already entailed by the visible context. No tool is needed. |

## Decision order

Apply in this order and stop at the first that fires:

1. Is an action being requested at all? If no, or the answer is entailed by the
   visible context → `direct_answer`.
2. Is any listed tool capable of it, and do permission, policy and external state
   all allow it? If any of these blocks → `cannot_answer`.
3. Is any **required** argument missing and underivable from visible context?
   → `request_for_info`.
4. Otherwise → `tool_call`.

The order matters. A row can be both permission-blocked and missing an argument;
the permission block wins, because asking the user for the argument would not
unblock it.

## What you may use

**Only what is printed in the `row` cell:** `[SYSTEM_INSTRUCTION]`,
`[TOOL_SCHEMAS]`, `[WORKSPACE_CONTEXT]`, `[PERMISSION_CONTEXT]`, `[USER_QUERY]`.

## What you must not use

- Any outside knowledge of this project, its hypotheses, or its expected results.
- Any guess about how many rows should get each label. **The class balance is not
  given to you and you must not assume one.** Do not try to even out your counts.
- Wording style, politeness, or tone. If your label rests on *how* something is
  phrased rather than on capability, permission, policy, or a missing required
  field, flag it in `another_mode_defensible` and say so in the reason.
- The item token. `A1-037` carries no information; it is an arbitrary label.

## Columns to fill

- `your_label` — exactly one of `tool_call`, `request_for_info`,
  `cannot_answer`, `direct_answer`. Lowercase, no spaces. Leave none blank.
- `another_mode_defensible__yes_no` — `yes` if a second reading is genuinely
  defensible from the visible row alone; otherwise `no`. Be honest here; this
  column is how defective rows get found, and a defective row is a finding, not
  a failure.
- `operational_reason` — one short clause naming the operative fact, e.g.
  *"policy_allows_action false — retention policy forbids deletion"*, or
  *"required arg start_iso missing, not derivable"*. If you cannot write an
  operational reason, that is itself a signal: mark `another_mode_defensible: yes`.

## Rate and fatigue

Take a break every ~40 rows. Rushed annotation at the tail is the classic way an
agreement study goes wrong. There is no time limit.
