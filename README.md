# Model-Based Evaluation Suite

A standard-library-only toy experiment that evaluates fixed policies in a learned linear model and a nonlinear reference environment. It reports predicted return, actual return, their gap, and transition error over multiple deterministic seeds.

## Reproduce the experiment

```bash
python3 main.py --seeds 0 1 2 3 4 --horizon 30 \
  --output artifacts/evaluation.json \
  --aigov-event artifacts/evaluation-event.json \
  --run-id wm-demo-001 --threshold 0.25
python3 -m unittest discover -s tests -v
```

The report records model/environment identifiers, seeds, horizon, per-policy/per-seed outcomes, policy aggregates, and limitations. Repeating the same seeds and horizon produces the same evaluation report. The optional evidence event has a timestamp, so its bytes vary by run.

## Send the evaluation result to AIGov Core

Start a local development AIGov Core runtime and configure its API key as described in the [AIGov Core runtime quickstart](https://github.com/AIGovDev/aigov-core/blob/main/docs/quickstart-runtime.md). Then, from this repository:

```bash
export GOVAI_AUDIT_BASE_URL="http://127.0.0.1:8088"
export GOVAI_API_KEY="<local development key>"
curl -fsS -H "Authorization: Bearer $GOVAI_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @artifacts/evaluation-event.json \
  "$GOVAI_AUDIT_BASE_URL/evidence"
curl -fsS -H "Authorization: Bearer $GOVAI_API_KEY" \
  "$GOVAI_AUDIT_BASE_URL/compliance-summary/wm-demo-001"
curl -fsS -H "Authorization: Bearer $GOVAI_API_KEY" \
  "$GOVAI_AUDIT_BASE_URL/api/export/wm-demo-001" \
  > artifacts/aigov-audit-export.json
curl -fsS -H "Authorization: Bearer $GOVAI_API_KEY" \
  "$GOVAI_AUDIT_BASE_URL/verify/wm-demo-001"
```

With only an `evaluation_reported` event, the run is expected to remain `BLOCKED`: a complete lifecycle needs additional dataset, training, risk-review, approval, and promotion evidence. The summary and export come from the ledger-backed runtime; the local report does not make a governance decision. Use a fresh run ID when repeating against the same ledger.

## Scope and limitations

- The reference environment is synthetic, one-dimensional, and deliberately nonlinear while the learned model omits the nonlinear term.
- The policies are fixed examples, not a planner or learned controller.
- Seeds vary the initial state; they do not represent independent real-world environments.
- `passed` means only that the mean transition error is at or below the supplied threshold. It is not a safety, compliance, or deployment approval.
- AIGov Core validates the event schema and applies its configured lifecycle policy. This single event is evidence input, not proof of a complete governance process.

The experiment is a reference implementation for connecting model-based evaluation evidence to a governance ledger, not a benchmark result or a validated model-safety method.
