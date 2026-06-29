# TTEA Implementation Log

## Files changed

1. `attacks/ttea.py`
2. `attacks/__init__.py`
3. `args/attack_args.py`

## What was implemented

1. Replaced the old unused `attack(prompt)` stub with the repo-compatible attack API:
   - `__init__(args)`
   - `get_query(query_id)`
   - `parse_response(response)`

2. Added `TaxonomyNode`, the state object for each public-prior taxonomy node:
   - `node_id`
   - `label`
   - `description`
   - `parent`, `children`, `depth`
   - `prototype` embedding
   - `status`: `unvisited`, `active`, `pruned`, `mature`, `saturated`
   - `posterior`
   - `visits`
   - accumulated semantic shift
   - accumulated novelty reward
   - recent rewards
   - extracted anchors

3. Added taxonomy construction:
   - TTEA can load a JSON taxonomy from `--ak_taxonomy_path`.
   - If no taxonomy path is provided, TTEA uses a built-in public-prior taxonomy.
   - Each node description is embedded as a local semantic prototype.
   - Only root children are activated initially, so the tree expands lazily.

4. Added node scheduler:
   - The scheduler scores active or mature nodes with:
     `avg_reward + UCB exploration bonus + posterior prior + sibling entropy`.
   - Active nodes use exploration/probing.
   - Mature leaves with anchors use exploitation.

5. Added query generation:
   - Exploration generates a benign probe query for the selected taxonomy node.
   - Exploitation generates a local anchor-focused query inside a mature node.

6. Added response update:
   - Parses informative chunks from the RAG response.
   - Queries a shadow LLM with a public-knowledge-only instruction.
   - Computes semantic shift as `1 - cosine(target_response, shadow_response)`.
   - Computes novelty reward against a global memory cache.
   - Extracts candidate anchor terms and named entities from new content.
   - Updates node visits, semantic shift, novelty reward, anchors, status, and posterior.
   - Backpropagates partial shift and reward to ancestor nodes.

7. Added node lifecycle transitions:
   - Expand children when semantic shift is high.
   - Prune low-shift, zero-reward nodes.
   - Mark useful terminal or stable nodes as mature.
   - Mark repeatedly zero-reward nodes as saturated.

8. Added detailed runtime logging:
   - taxonomy initialization summary
   - scheduler utility
   - selected node, mode, posterior, visits, shift, reward, and anchor
   - generated query
   - semantic shift
   - response update
   - anchors added
   - memory cache size
   - posterior updates
   - expand, prune, mature, and saturated state changes

9. Registered TTEA:
   - Added `from .ttea import *` in `attacks/__init__.py`.
   - Added `ttea_attack` to `all_attacks`.
   - Added TTEA CLI arguments under `attack == "TTEA"` in `args/attack_args.py`.

## Verification

Command run:

```powershell
python -m py_compile attacks\ttea.py args\attack_args.py attacks\__init__.py
```

Result:

```text
Passed.
```

Additional import check attempted:

```powershell
python -c "import attacks; print('TTEA' in attacks.all_attacks, attacks.ttea_attack)"
```

Result:

```text
Failed before reaching TTEA because existing attack modules import `langchain_core`,
which is missing in the current environment.
```

This is an environment dependency issue in the current workspace, not a syntax error
in the TTEA implementation.

## Notes

- `attacks/ttea.py` was shown as an untracked file by `git status`, even though it
  existed in the IDE. It should be added to version control together with this log
  if this implementation is kept.
- The implementation is intentionally lightweight and pipeline-compatible. It does
  not require a new prompt file. A custom public taxonomy can be supplied later as
  JSON through `--ak_taxonomy_path`.
## Follow-up update: LLM-generated taxonomy

Added an optional LLM taxonomy builder after review of `_default_taxonomy`.

New behavior:

1. `--ak_taxonomy_builder static`
   - Keeps the built-in deterministic public-prior taxonomy.
   - This remains the default for reproducibility.

2. `--ak_taxonomy_builder llm`
   - Uses `--ak_topic_word` as the root category.
   - For example, `--ak_topic_word 医学` asks the attack LLM to generate public child categories under medicine.
   - The LLM is instructed to use public ontology knowledge only and to output strict JSON:
     `{"children": [{"label": "...", "description": "..."}]}`

3. Lazy LLM expansion:
   - At initialization, TTEA asks the LLM only for root children.
   - Later, when an active node should expand but has no children yet, TTEA asks the LLM for that node's child categories.
   - Newly generated child nodes are embedded immediately and activated.

4. New CLI args:
   - `--ak_taxonomy_builder`, default `static`, choices `static` or `llm`
   - `--ak_llm_taxonomy_children`, default `5`

Additional verification:

```powershell
python -m py_compile attacks\ttea.py args\attack_args.py attacks\__init__.py
```

Result:

```text
Passed.
```
## Failure analysis against other attacks

I compared TTEA with `CopyBreak`, `RandomText`, `RandomToken`, `RandomEmb`, and `IKEA` in the same `attacks/` folder.

Main finding:

TTEA currently runs, but its query/parse contract does not match the benchmark's knowledge-extraction path.

1. CopyBreak and Random attacks wrap generated information in an attack template: `query = attack_template.replace("<info>", info)`. The template asks the target RAG to repeat all retrieved context, which matches `parse_repeat_response`.

2. TTEA currently sends normal natural-language questions such as: "Please provide a concise but concrete overview..." This usually makes the RAG produce a normal summary answer, not `Context N:` blocks or quoted repeated chunks. Therefore `parse_repeat_response` often extracts zero chunks.

3. TTEA then falls back to treating the whole response as one informative chunk when `detect_refusal` says the answer is not a refusal. This can make the attack look successful internally (`reward=1`, one new chunk) even when it did not extract repeated corpus context. This is a false-positive success signal.

4. IKEA also uses natural questions, but IKEA intentionally returns `[response]` and uses the response as whole-answer feedback for anchor mutation. TTEA is different: it tries to maintain extracted memory chunks and novelty rewards. So TTEA should either use a repeat-context template like CopyBreak/Random, or explicitly switch to an IKEA-style whole-response objective.

5. Static taxonomy quality matters. If `--ak_taxonomy_preset general` is used on the Pokemon dataset, the scheduler may start from irrelevant public categories. The Pokemon run should use `--ak_taxonomy_preset pokemon`, or TTEA should add an automatic preset mapping from `topic_word`.

6. Current semantic shift is computed on the whole target response against a shadow public-knowledge answer. This is useful as a routing signal, but it is not proof that private/retrieved chunks were extracted. It should not be the only success driver.

7. Cost accounting is incomplete: the recorder tracks the main attack LLM, while TTEA also calls a shadow LLM and `detect_refusal` may call another LLM. The run can therefore under-report attack-side LLM usage.

Recommended fix:

Add a TTEA attack template, or reuse `copybreak/attack_template.txt`, so `_generate_probe_query` and `_generate_exploitation_query` produce node-specific `info` but the final query still asks the RAG to repeat retrieved context. Then change novelty reward so unparsed whole-response fallback does not count as a successful extracted chunk.

## Debug update: query-template and parser fixes

I applied the failure analysis step by step and reran the Pokemon debug attack after each major change.

1. Wrapped TTEA node probes with an attack template.
   - Added `--ak_attack_template` for TTEA.
   - TTEA now generates a taxonomy-specific `info_query`, then wraps it with an attack template before sending it to the target RAG.
   - The shadow semantic-shift prompt still uses the raw `info_query`, not the prompt-injection wrapper.

2. Removed the false-positive whole-response reward fallback.
   - Before: if `parse_repeat_response` returned no chunks and the response was non-refusal, TTEA stored the whole response as one new chunk.
   - After: only parser-extracted context blocks count as extracted chunks and novelty reward.
   - This avoids reporting success when the RAG only produced a normal summary.

3. Verified the first fix.
   - With the CopyBreak template, one run extracted 3 chunks.
   - A second run still extracted 0 chunks because the model sometimes repeated context as bare lines instead of `Context N:` blocks.
   - This showed the remaining failure was output-format instability, not taxonomy scheduling.

4. Added a TTEA-specific context-block template.
   - New file: `prompts/ttea/attack_template.txt`.
   - The template explicitly requires:
     `Context 0: "..."`, `Context 1: "..."`, etc.
   - TTEA now defaults to this template instead of `copybreak/attack_template.txt`.

5. Removed parse-miss refusal detection from TTEA.
   - On parse miss, TTEA now logs the miss and returns no chunks.
   - This avoids slow and noisy `detect_refusal` calls when no OpenAI parser key is configured.

6. Added static taxonomy preset auto-selection.
   - `--ak_taxonomy_preset` now supports `auto` and defaults to it.
   - For `--ak_topic_word pokemon`, TTEA resolves the preset to `pokemon` automatically.

7. Fixed shadow LLM token accounting.
   - If `attack_llm` and `shadow_llm` are the same model, TTEA reuses one engine.
   - If they differ, TTEA syncs shadow LLM token deltas into attack metrics so the recorder does not show zero attack-side usage.

Verification command:

```powershell
conda run -n ke-rag python -m py_compile attacks\ttea.py args\attack_args.py attacks\__init__.py
```

Result: passed.

Final Pokemon debug run:

```text
TTEA initialized: nodes=13, root_children=3, topic='pokemon', taxonomy_builder=static, taxonomy_preset=pokemon
Extracted 3 chunks from response.
TTEA response update: shift=0.8994 reward=0.6064 parsed_chunks=3 new_chunks=3 anchors_added=12
Attack LLM total tokens: 593
```

Conclusion:

The main failure was the mismatch between TTEA's natural-language query style and the benchmark parser's expected repeat-context format. The stable fix is to make TTEA use a parser-aligned `Context N:` template and to count only parsed context blocks as extracted knowledge.

## Standard-style smoke run from README settings

I ran TTEA with the same core settings used by the README/scripts standard experiments:

- `TextRAG`
- `MiniLM` retriever
- `topk=3`
- `textrag/system.txt` and `textrag/template.txt`
- `temperature=0.1`
- no debug dataset truncation
- persisted standard vector DBs: `pokemon_1k`, `enron_500k`, `harrypotter_26k`, `health_100k`
- `deepseek-v4-pro` as RAG generator, TTEA attack LLM, and shadow LLM
- `--ak_max_query 5` smoke budget per dataset

Results:

```text
Pokemon          logs/06_26/TextRAG-Pokemon/TTEA-None-21_55_12/results.jsonl          records=5 empty_extracted_info=0
Enron            logs/06_26/TextRAG-Enron/TTEA-None-21_57_06/results.jsonl            records=5 empty_extracted_info=0
HarryPotter      logs/06_26/TextRAG-HarryPotter/TTEA-None-21_59_28/results.jsonl      records=5 empty_extracted_info=0
HealthCareMagic  logs/06_26/TextRAG-HealthCareMagic/TTEA-None-22_01_12/results.jsonl  records=5 empty_extracted_info=0
```

Observed runtime behavior:

- All four standard datasets completed without crashing.
- Every query produced non-empty `extracted_info`.
- Logs consistently showed `Extracted 3 chunks from response.` because `topk=3` and the TTEA template requests `Context N:` blocks.
- Token accounting for the shadow/attack LLM is no longer zero.

Quality notes for follow-up:

- Pokemon, Enron, and HealthCareMagic resolved to sensible static presets automatically: `pokemon`, `business`, and `medicine`.
- HarryPotter currently falls back to `general`, so it starts from `Business and Organizations`; this runs but is semantically weak. A literature/fantasy/fiction preset or a topic mapping for `harry potter` would make the taxonomy object cleaner.
- Repeated early queries often retrieve the same top-3 chunks, so after query 0 the novelty reward can become 0. This is expected with the current probe text being identical across visits to the same node, but it suggests the next improvement should vary the probe query using node anchors, visit count, or sibling/child expansion sooner.

## Taxonomy-preset diagnosis and fix

I checked whether zero extraction was caused by the static taxonomy tree not matching the dataset.

Finding:

- For the current TTEA runs, empty extraction was not primarily caused by taxonomy mismatch. The `Context N:` template and parser alignment are the decisive factors for whether `extracted_info` is empty.
- The interrupted 200-query Pokemon TTEA run reached 104 records with `empty_extracted_info=0`.
- The four standard-style smoke runs also had `empty_extracted_info=0`.

However, taxonomy mismatch does hurt extraction quality and diversity:

- HarryPotter previously resolved to the `general` preset.
- That caused TTEA to start from `Business and Organizations`, which still extracted chunks but was semantically weak and led to less meaningful scheduling.

Fix:

- Added `dataset_name = args.dataset` into TTEA preset resolution.
- Added a new static `literature` taxonomy preset for fiction/book datasets.
- `auto` now maps `harry`, `potter`, `fiction`, `literature`, `book`, `novel`, and `fantasy` to `literature`.
- Added `literature` to `--ak_taxonomy_preset` choices.

Verification:

```text
TTEA initialized: topic='harry potter', taxonomy_builder=static, taxonomy_preset=literature
selected node=root.characters-and-relationships label='Characters and Relationships'
Extracted 3 chunks from response.
```

## Full Pokemon comparison run

I resumed the standard Pokemon TTEA run and completed the full 200-query budget.

Run path:

```text
logs/standard/deepseek_v4_pro/Pokemon/TTEA/results.jsonl
```

Completion check:

```text
queries=200
empty_extracted_info=0
```

I exported comparison CSVs with:

```powershell
conda run -n ke-rag python scripts\export_standard_results.py --logs-root logs\standard --output-dir logs\standard_exports_ttea_compare --dataset Pokemon
```

Output summary:

```text
logs/standard_exports_ttea_compare/summary.csv
```

Pokemon comparison summary:

```text
model            attack       queries  extracted_count  retrieved_count  total_time_s  rag_tokens  attack_tokens  overall_tokens
deepseek_chat    CopyBreak    200      599              600              583.27        87908       85432          173340
deepseek_chat    DGEA         200      501              600              13366.27      80679       0              80679
deepseek_chat    IKEA         200      187              600              3077.76       107236      268239         375475
deepseek_chat    RandomEmb    200      459              600              4371.52       82121       0              82121
deepseek_chat    RandomText   200      600              600              593.48        85047       26854          111901
deepseek_chat    RandomToken  200      517              600              308.31        85089       0              85089
deepseek_v4_pro  TTEA         200      598              600              2520.11       121109      118248         239357
```

Important comparison caveat:

- Existing methods are under `deepseek_chat` logs.
- Current `keys.yaml` exposes `deepseek-v4-pro`, so TTEA was run under `deepseek_v4_pro`.
- The comparison is useful for pipeline-level behavior and extraction count, but model naming/config is not perfectly identical.

Observed TTEA behavior:

- TTEA extraction count is high: 598/600 retrieved contexts extracted.
- No query had empty extracted info.
- Runtime and token use are higher than CopyBreak/RandomText because TTEA calls a shadow LLM every query for semantic-shift scoring.
- Novelty saturates early because many taxonomy-node probe queries retrieve the same top-k contexts repeatedly. This is now the main quality issue, not parser failure.
