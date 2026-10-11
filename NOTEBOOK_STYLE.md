# History notebook style

Use `zeta2_history.ipynb` as the reference. The user has approved applying this style throughout the notebook collection.

## Structure

1. Title and a short description of the constant and diagnostic.
2. Optional dependency install cell, then one setup cell. Import only used dependencies and define `DEPTHS = list(range(10, 501, 10))` once.
3. Numbered examples in historical year order (announcement year when explicitly identified; otherwise publication year). Put undated work last. Keep unimplemented historical entries in their chronological position and label them clearly.
4. For each executable example: one concise Markdown cell and one code cell. Markdown gives a source link, family, CMF/start/direction, readout, and any essential index or bound caveat. Avoid derivations and execution history.
5. Comparison plots with consistent labels, followed by a compact chronological overview. Include only constructions with a CMF recovery implemented in the notebook; omit historical entries without a recovered CMF from the final table. A recovered construction whose calculation timed out may remain, with its measured value marked as not computed. Generate measured values from the current results, never copied constants.

## Example cells

- Follow the same sequence: construct `cmf`; define `start` and `trajectory`; compute `walks`; extract exact `approximants`; calculate `deltas`; save results and accuracy; print the endpoint.
- Use the same variable names and spacing throughout. Aim for roughly 10–15 readable lines, allowing longer readouts to wrap.
- Keep the mathematical choices visible. Move seed generation, staircase multiplication, symbolic construction, and independent verification into `cmfs.py` or `local_checks/`.
- Prefer `ramanujantools` pFq or Meijer G constructors to custom fields when the parameter mapping and rational readout have been verified exactly. A similar integral or a sum of hypergeometric functions is not sufficient evidence of field equivalence.
- Retain a necessary custom CMF with a short explanation of the missing equivalence. Do not replace a full field by an unverified trajectory recurrence.
- Keep prose brief, repair encoding artifacts, and omit incidental calculations unrelated to the comparison.

## Computation

- Target index 500, sampling every 10. State whether the index is a source index, walk length, or staircase parameter, including offsets. Never silently substitute one for another.
- If source index 500 is outside a construction's allowed residue class, keep its exact index convention, sample allowed indices up to 500, and label the actual endpoint in the output and table. Do not report it as delta at 500.
- Source audits without a verified CMF retain a concise status section and no fabricated examples or comparison values. Limit-identification notebooks must state any target-precision limitation.
- Use exact rational walks and readouts, then sufficient Arb precision to resolve the error. Check endpoint results at higher precision.
- During cleanup, allow up to 120 seconds per example; rerun timed-out examples with a longer requested limit (240 or 480 seconds in the current cleanup). Report the actual limit and result. Do not substitute an old result or a smaller depth as a depth-500 result.
- Clear stale outputs before executing changed cells. Keep successful outputs and regenerate plots/tables from those results only. Label skipped calculations explicitly.
- Preserve independent verification outside the presentation cells. Report missing CMFs separately from runtime skips.
