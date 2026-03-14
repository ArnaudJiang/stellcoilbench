# StellCoilBench Knowledge Layer

Context layer for the StellCoilBench autopilot. The LLM proposer uses
`llm_context.md` (optimization guide, threshold scaling, literature) plus
`surface_catalog.json` and run context from `build_context` (submissions +
autopilot_failures).

## Directory Structure

```
knowledge/
├── llm_context.md          ← master document: optimization guide, threshold scaling, literature (~1100 lines)
├── surface_catalog.json    ← plasma surface physical parameters (a, R, a0, nfp)
├── make_run_card.py        ← run summary → text card for LLM context
├── make_postmortem.py      ← failure analysis for failed runs
├── llm_client.py           ← unified LLM client (OpenAI, Anthropic, local)
└── llm_endpoints.py        ← call_propose (used by direct LLM proposer)
```

## LLM Context Documents

| File | Purpose |
|------|---------|
| `llm_context.md` | Optimization guide, threshold scaling, comprehensive literature. Primary source for proposer. |
| `surface_catalog.json` | Physical parameters of each plasma surface (a, R, a0, nfp, typical ncoils). |

## Data Flow

1. CI runs cases → writes to `submissions/` (success) and `policy/autopilot_failures.json` (failure).
2. `tools/build_context.py` loads summaries from submissions + failures.
3. Proposer uses `llm_context.md` + `surface_catalog.json` + build context for case generation.
4. `make_run_card` and `make_postmortem` format run data for the LLM prompt.

## Using the LLM Proposer

```bash
ANTHROPIC_API_KEY=sk-... python -m tools.propose_batch --llm --dry-run
```

Requires `ANTHROPIC_API_KEY` (or `KB_LLM_*` env vars). No KB server or paper pipeline needed.
