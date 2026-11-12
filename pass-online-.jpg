# AI Research Team

A multi-agent research project built with [LangGraph](https://langchain-ai.github.io/langgraph/). Give it a topic and a team of agents plans the research, searches the web, writes a cited report, and reviews it until it is approved.

This README uses **Ollama** (a free, local model runner) as the example model provider, so you can run everything on your own machine with no API costs.

## How it works

```
START -> planner -> searcher -> writer -> critic --(approved or max revisions)--> END
                                   ^                |
                                   +-- feedback ----+
```

| Agent | Job |
|---|---|
| **Planner** | Breaks the topic into 3 to 5 focused research questions |
| **Searcher** | Runs a web search for each question and collects the sources |
| **Writer** | Writes (or revises) a Markdown report that cites the sources as [1], [2] |
| **Critic** | Checks the draft against the sources, then approves it or sends feedback back to the writer |

Key ideas the project demonstrates:

- **Shared state:** all agents read and write one typed `ResearchState`.
- **Conditional edges:** the critic's verdict decides whether to loop or finish.
- **Loop limit:** `MAX_REVISIONS` (default 3) stops the writer and critic from looping forever.
- **Structured output:** the planner and critic return Pydantic models.
- **Checkpointing:** `MemorySaver` stores state for each run.

## Requirements

- macOS, Linux, or Windows
- Python 3.10 or newer
- About 5 GB of free disk space for a local model
- An internet connection (for the web search step)

## Setup with Ollama

### 1. Install Ollama

Download the app from <https://ollama.com/download>, move it to Applications, and open it. A llama icon appears in the menu bar and the server runs in the background.

Or with Homebrew:

```bash
brew install ollama
```

Check it is installed (open a new Terminal window first):

```bash
ollama --version
```

If you installed with Homebrew and the server is not running, start it in a separate Terminal window and leave it open:

```bash
ollama serve
```

### 2. Download a model

```bash
ollama pull llama3.2:3b
```

This is a small model (about 2 GB) that works on most laptops. For better report quality and more reliable structured output, use a larger model if your machine can handle it:

```bash
ollama pull llama3.1
```

Test the model:

```bash
ollama run llama3.2:3b "Say hello"
```

### 3. Create a Python environment and install packages

```bash
mkdir research-team && cd research-team
python3 -m venv .venv
source .venv/bin/activate

pip install -U langgraph langchain langchain-ollama pydantic ddgs
```

Put `research_team.py` in this folder.

### 4. Choose the model

The script reads the model name from the `RESEARCH_MODEL` environment variable. The format is `provider:model-name`.

```bash
export RESEARCH_MODEL="ollama:llama3.2:3b"
```

This variable lasts only for the current Terminal window. Set it again in each new window, or add the line to `~/.zshrc`.

## Run it

```bash
python research_team.py "Impact of Karpenter on Kubernetes cost optimization"
```

Keep the topic in quotes. The finished report is saved to `report.md`.

Use `--out` to save under a different name:

```bash
python research_team.py "LangGraph vs CrewAI for multi-agent apps" --out frameworks.md
```

### Expected output

```
PLANNER   questions:
  - What is Karpenter and how does it provision nodes?
  - How does consolidation reduce compute costs?
  - ...
SEARCHER  collected 20 sources
WRITER    draft #1 written
CRITIC    REVISE: Several claims are not supported by a cited source...
WRITER    draft #2 written
CRITIC    APPROVED

Done: approved
Report saved to report.md
```

The questions, source counts, and verdicts differ on each run. A local model can take a few minutes, and the terminal may stay quiet while it works.

Open the report:

```bash
open report.md        # macOS
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| Model | `RESEARCH_MODEL` environment variable | `anthropic:claude-sonnet-5-5` |
| Max writer/critic rounds | `MAX_REVISIONS` in `research_team.py` | 3 |
| Search results per question | `RESULTS_PER_QUESTION` in `research_team.py` | 4 |
| Output file | `--out` command line option | `report.md` |

Because the default model is Anthropic, you must set `RESEARCH_MODEL` when using Ollama.

## Using other model providers

Only the package, the key, and `RESEARCH_MODEL` change. The code stays the same.

| Provider | Install | Environment variables |
|---|---|---|
| Ollama (local) | `pip install langchain-ollama` | `RESEARCH_MODEL="ollama:llama3.2:3b"` |
| Anthropic | `pip install langchain-anthropic` | `ANTHROPIC_API_KEY`, `RESEARCH_MODEL="anthropic:claude-sonnet-5-5"` |
| Google Gemini | `pip install langchain-google-genai` | `GOOGLE_API_KEY`, `RESEARCH_MODEL="google_genai:gemini-2.5-flash"` |
| Amazon Bedrock | `pip install langchain-aws` | AWS credentials, `RESEARCH_MODEL="bedrock_converse:<model-id>"` |

Model names change over time, so check each provider's documentation for current ones. Hosted APIs are billed separately from chat subscriptions and need credits or billing set up.

## Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: ollama` | Ollama is not installed. Install it, then open a new Terminal window. |
| `connection refused` to `localhost:11434` | The Ollama server is not running. Open the Ollama app or run `ollama serve`. |
| `model not found` | Run `ollama pull <model>` and make sure the name matches `RESEARCH_MODEL`. |
| `ModuleNotFoundError` | Activate the virtual environment and re-run the `pip install` command. |
| Parsing or validation errors from the planner or critic | The model is too small for structured output. Use `llama3.1` or a hosted model. |
| `temperature is not supported` | Remove `temperature=0` from the `init_chat_model(...)` line. |
| `SEARCHER collected 0 sources` | The searches failed. Check your internet connection and re-run. |
| Very slow runs | Use a smaller model, or close other heavy applications. |

## Project ideas to extend it

1. Add a human approval step before the final write using LangGraph's `interrupt`.
2. Send unsupported claims back to the searcher for more sources.
3. Run the searches in parallel with the `Send` API.
4. Save state with a SQLite or Postgres checkpointer so runs can be resumed.
5. Add LangSmith tracing (`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`) to inspect every agent call.

## Files

```
research-team/
  research_team.py   # the graph, agents, and command line entry point
  README.md          # this file
  report.md          # generated output
```
