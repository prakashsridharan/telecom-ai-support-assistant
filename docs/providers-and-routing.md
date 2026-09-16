# Model providers and routing strategies

Two independent choices. `LLM_PROVIDER` picks *who writes the reply*.
`ROUTING_MODE` picks *who decides which tools to call*.

|  | `ROUTING_MODE=rules` | `ROUTING_MODE=agent` |
|---|---|---|
| **`LLM_PROVIDER=demo`** | Fully working, no API key. The default. | Not possible — degrades to rules, and `/health` says so. |
| **`LLM_PROVIDER=anthropic`** | Claude phrases answers from evidence the app gathered. | Claude chooses and calls the tools itself. |
| **`LLM_PROVIDER=openai`** | GPT phrases answers from evidence the app gathered. | GPT chooses and calls the tools itself. |

Every combination returns the identical response envelope, so the UI, the API
contract and the evaluation harness never branch on configuration.

## Providers

```
app/services/providers/
├── base.py                  the contract: ChatProvider, LLMResult, prompts
├── demo.py                  deterministic, keyless
├── anthropic_provider.py    Claude (Messages API)
└── openai_provider.py       GPT (Chat Completions)
```

Nothing above this directory imports a vendor SDK. Adding a third provider
means adding one file and one entry in `build_provider()`.

### Configuring Claude

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-5
```

`ANTHROPIC_MODEL` is free-form, so any current model ID works —
`claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`.

`LLM_EFFORT` (`low` … `max`) controls how much reasoning Claude spends.
Support routing is not a reasoning-heavy task, so the default is `low`, which
keeps latency and cost down. Raise it if you extend the assistant into
multi-step troubleshooting.

### Configuring OpenAI

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### Degradation

A provider that is selected but unusable — missing key, unknown name, SDK
failure — logs a warning and falls back to the demo provider. The app always
starts; `/health` reports `degraded` with the specific reason. That is
deliberate: a deployment that silently answers from the wrong backend is worse
than one that says what it is doing.

A provider call that fails *at request time* (timeout, rate limit, outage)
degrades differently: the request falls back to knowledge-base retrieval and is
labelled `mode: "fallback"`, so the customer still gets a grounded answer and
the response says it was degraded.

## Routing strategies

### `rules` — the application decides

`Orchestrator.classify()` matches the message against regexes and keyword sets,
extracts the identifier, and calls exactly one tool. The model, if any, only
writes the reply.

- Deterministic and reproducible — the same input always routes the same way.
- Inspectable — a reviewer reads 40 lines and knows why a tool fired.
- Works with no API key.
- **Misses novel phrasings.** Recall is the price.

### `agent` — the model decides

The provider receives all four tool schemas from `app/services/tools.py` and
runs a real tool-calling loop: it picks tools, supplies arguments, reads
results, and may call more tools before answering.

- Handles phrasings no regex anticipated.
- Can chain calls — look up an order, then search the policy that explains its
  status — without the application predicting the sequence.
- **Non-deterministic**, costs a call per turn, and needs a key.

Both strategies call the *same* functions in `app/services/tools.py`, so
capability never diverges between modes.

### The loop

`AnthropicProvider.run_agent` is a manual loop rather than the SDK's beta tool
runner, for two reasons: the decision trail (which tools ran, which documents
they returned) has to be captured turn by turn for the API response, and the
project's legibility constraint argues against a beta dependency in the one
file a reviewer is most likely to read.

Guardrails in the loop:

- `AGENT_MAX_ITERATIONS` (default 5) caps the turns, so a model that keeps
  calling tools cannot spend unbounded time or money on one request. Hitting
  the cap degrades to retrieval rather than returning nothing.
- Tool exceptions become tool results with `is_error`, not 500s — the model
  sees the failure and can tell the customer.
- A "record not found" result says so explicitly and instructs the model not to
  describe a status, which is the anti-fabrication rule enforced at the tool
  boundary rather than only in the prompt.
- `stop_reason: "refusal"` is handled explicitly.

### Tool descriptions are part of the safety surface

In agent mode, the tool descriptions in `app/services/tools.py` are the only
thing telling the model when a lookup is appropriate. They are written to
forbid guessing:

> Never invent or guess an order number: if the customer has not given one,
> ask for it instead of calling this tool.

Changing a tool description changes the assistant's behaviour as much as
changing the system prompt. Treat those strings as code.

## Testing tool calling without an API key

`tests/fakes.py` provides fake Anthropic and OpenAI clients that replay
scripted turns with the same shape the SDKs return. That covers the parts most
likely to break — argument parsing, dispatch, result framing, the iteration
cap, failure paths — as ordinary unit tests that are free and deterministic.

```bash
pytest tests/test_agent_tool_calling.py -v
```

What the fakes *cannot* tell you is whether a real model chooses the right tool
for a real phrasing. That needs live evaluation against the golden set with a
key configured, and a judge or recorded fixtures to handle non-determinism —
see [evaluation.md](evaluation.md).
