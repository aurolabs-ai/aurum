# Aurum Compact -- The difference between guessing and knowing

Agentic research engine. Searches reactively. Thinks before it writes. Open source.

Most AI research tools plan all searches upfront and execute blindly. Aurum searches, reads what it finds, decides what to search next, and builds understanding iteratively. Each round of research informs the next. The result is a source-graded, director-reviewed report with a bottom-line confidence assessment.

```bash
git clone https://github.com/aurolabs-ai/aurum.git
cd aurum && npm install
export OPENAI_API_KEY=sk-...
node bin/aurum.js "What happens if the EU bans diesel cars by 2030?"
```

## How it works

```
$ aurum "Impact of US AI chip export controls on the global semiconductor market"

aurum v1.0.0

  Question: Impact of US AI chip export controls on the global semiconductor market
  Language: en | Depth: standard | Output: rapport.md

  Understanding question...                          1.2s
  Research round 1: 5 searches, 22 data points       2.1min
  Research round 2: 4 searches, 18 new findings      1.8min
  Research round 3: 3 searches, filling gaps          1.4min
  ...
  Research complete: 8 rounds, 45 searches, 180 data points
  Writing report (gpt-5.4)...                         3.2min
  Self-review...                                      1.1min
  Director review: 82/100                             0.8s
  Bottom line generated                               0.3s

  Report saved: rapport.md

  Bottom line: US export controls are accelerating Chinese semiconductor
  self-sufficiency while fragmenting the global supply chain. [Probable, 78%]

  Words:      12,400
  Sources:    67
  Time:       14.2min
  Cost:       ~EUR 5.80
```

## Agentic vs pipeline

Traditional pipeline research tools decompose your question into search queries, execute them all at once, then try to write a report from whatever came back. This is fast but brittle -- if the initial queries miss something, the report has gaps.

Aurum's agentic researcher works differently:

1. **Understand** -- Classifies the question, builds a research checklist, identifies what data is needed
2. **Search** -- Executes 3-8 targeted queries per round, including site-specific searches (prediction markets, institutional sources, financial data)
3. **Analyze** -- Reads everything found, extracts specific data points, updates the checklist, identifies gaps and contradictions
4. **Decide** -- Based on what is known and what is missing, generates new queries targeting the gaps
5. **Repeat** -- Continues until the checklist is sufficiently covered or diminishing returns are detected
6. **Write** -- Produces a structured report using all accumulated evidence, with every claim sourced
7. **Self-review** -- Checks for unsourced claims, vague language, thin sections, and fixes them
8. **Director** -- Adversarial quality review scores on 5 dimensions

The model drives its own search strategy. It sees its accumulated knowledge when deciding what to search next. It reacts to what it finds rather than following a rigid plan.

## Installation

```bash
git clone https://github.com/aurolabs-ai/aurum.git
cd aurum && npm install
```

Run directly:

```bash
node bin/aurum.js "your question"
```

Or create a global `aurum` command:

```bash
npm link
aurum "your question"
```

## Setup

You need an OpenAI API key. Get one at [platform.openai.com](https://platform.openai.com).

```bash
export OPENAI_API_KEY=sk-...
```

Or create `.aurum.yaml` in your project directory:

```yaml
openai_api_key: sk-...
language: en
depth: standard
```

## Models

Aurum Compact uses two OpenAI models:

- **gpt-5.4-mini** -- research phase (searching, analyzing, deciding next queries)
- **gpt-5.4** -- writing phase (report generation, self-review, director evaluation)

These are fixed. No configuration needed.

## Commands

| Command | Description |
|---|---|
| `aurum "question"` | Run research, save report |

## Options

| Flag | Description |
|---|---|
| `-l, --language <code>` | Output language (en, sv, de, fr, ...) |
| `-d, --depth <level>` | `quick` or `standard` |
| `-o, --output <file>` | Output filename (default: rapport.md) |
| `-i, --iterations <n>` | Director loop iterations (default: 2, max: 2) |
| `-t, --template <name>` | Force template (default: auto-detect) |
| `-h, --help` | Show help |
| `-v, --version` | Show version |

## Cost

A typical Compact report costs approximately EUR 4-8, depending on question complexity and research depth.

| Mode | Time | Searches | Cost |
|---|---|---|---|
| `quick` | 5-10 min | 10-20 | ~EUR 4 |
| `standard` | 10-25 min | 20-50 | ~EUR 5-8 |

## Templates

Templates are auto-selected based on your question. Override with `--template`:

| Template | Best for |
|---|---|
| `scenario-analysis` | Future scenarios, geopolitical, policy impact |
| `market-research` | Market sizing, segmentation, TAM/SAM/SOM |
| `competitor-audit` | Competitive landscape, positioning |

## Need deeper analysis?

Aurum Deep uses frontier models with extended research cycles for EUR 499/report. Custom templates, human analyst review, and guaranteed turnaround. See [aurolabs.ai](https://aurolabs.ai).

## License

Source-available under the Business Source License 1.1. Free for personal and non-commercial use. Commercial use requires a license from Aurolabs AB. See [LICENSE](LICENSE) for details.

---

Built by [Aurolabs AB](https://aurolabs.ai), Stockholm.
