# Deep Research Agent

A Python-based Deep Research Agent built for the Sarvam FDSE Assignment.

The agent searches the live web, fetches source pages, builds a bounded research context, answers with citations, stores session history, streams progress updates, and includes an evaluation harness with deterministic checks, Groq-based judging, and a hallucination probe.

---

## Demo

Video Demo: 
Repository: https://github.com/RahulGupta-30/Sarvam_Assignment
APP LINK : https://rahulgupta-30-sarvam-assignment-app-ykcjoe.streamlit.app/

---

## Problem Statement

Normal chatbots often answer confidently without showing where the information came from. For research tasks, especially current or conflicting topics, users need:

- live web evidence,
- source citations,
- transparency into which sources were used,
- session continuity across follow-up questions,
- clear uncertainty when evidence is weak or conflicting.

This project solves that by building a research pipeline that searches, fetches, filters, cites, validates, and stores the full research trace.

---

## Target Users

This agent is useful for:

- students doing current-topic research,
- analysts comparing multiple sources,
- product or policy teams tracking fast-changing areas,
- users who need citation-grounded summaries,
- anyone who wants a transparent first research pass instead of a black-box answer.

---

## Definition of Deep Research

In this implementation, **deep research** means the agent does not only ask an LLM to answer.

It:

1. Plans a standalone research query.
2. Searches the web using Tavily.
3. Selects high-quality URLs for deeper reading.
4. Fetches readable page content.
5. Chunks and ranks source text.
6. Builds a limited context with source metadata.
7. Generates an answer using only fetched web context.
8. Adds citations using title, domain, and URL.
9. Validates citation URLs against fetched sources.
10. Stores the session, conversation, and turn-level research artifacts.

This is an assignment-level research agent, not a production research platform, but it demonstrates the full research-agent loop.

---

## Features

### 1. Web Research

- Uses Tavily for web search.
- Search results include title, URL, snippet, domain, and score when available.
- Uses Tavily extract to fetch readable page content.
- Stores metadata such as URL, title, domain, and retrieval timestamp.

### 2. Query Planning

- Converts the user’s current question into a standalone research question.
- Uses recent turns and rolling summary to resolve follow-ups.
- Example: “What impact did it have?” can be rewritten using the previous topic.

### 3. URL Selection

- Uses an LLM to choose the best URLs from Tavily results.
- Selection considers relevance, source quality, recency, and diversity.
- Retry/fallback logic can reduce failures caused by model access, quota, or transient API errors.

### 4. Context Construction

The context builder:

- cleans extracted text,
- splits pages into overlapping chunks,
- scores chunks against the query,
- limits the total context size,
- limits chunks per domain to encourage source diversity,
- preserves source metadata for citation mapping.

### 5. Citation-Grounded Answers

The answer generator is instructed to:

- use only the provided web context,
- cite important factual claims,
- use citation format `[Title — domain](URL)`,
- mention source conflicts,
- state uncertainty when evidence is weak or incomplete,
- avoid invented facts.

### 6. Citation Validation

The citation validator checks:

- whether the final answer contains citations,
- whether cited URLs came from retrieved sources,
- whether the model invented citation URLs.

This is a deterministic URL-integrity check. It does not fully prove semantic support, which is why the evaluation harness adds a Groq-based judge and hallucination probe.

### 7. Session Management

SQLite is used to persist:

- sessions,
- messages,
- research turns,
- selected URLs,
- fetched page previews,
- used source chunks,
- final answers,
- rolling summaries.

### 8. Streamlit UI

The app provides:

- chat interface,
- session sidebar,
- progress updates,
- selected URL trace,
- fetched page previews,
- used context sources,
- citation validation details,
- previous turn history.

### 9. Evaluation Harness

The evaluator runs the agent over a dataset and creates JSON and Markdown reports.

It includes:

- deterministic citation URL checks,
- search/fetch health checks,
- follow-up rewrite checks,
- Groq-based LLM judge scores,
- hallucination probe results,
- no single aggregate score.

---

## Architecture

```text
User Question
   ↓
Streamlit UI / CLI Evaluation
   ↓
SQLite Session Lookup
   ↓
Rolling Summary Update
   ↓
Query Planner
   ↓
Tavily Search
   ↓
LLM URL Selector
   ↓
Tavily Page Extraction
   ↓
Context Builder
   ↓
Final Answer Generator
   ↓
Citation Validator
   ↓
Optional Citation Repair
   ↓
SQLite Turn Storage
   ↓
Answer + Research Trace
```

---

## Project Structure

```text
.
├── app.py                    # Streamlit UI
├── research_pipeline.py       # Main orchestration pipeline
├── agent.py                   # URL selection, answer generation, citation repair
├── query_planner.py           # Standalone query planning
├── llm_provider.py            # Groq/OpenAI-compatible generation helper
├── search_tavily.py           # Tavily search
├── fetch_page.py              # Tavily page extraction
├── build_context.py           # Chunking, ranking, context construction
├── citation_validator.py      # Citation URL validation
├── session_store.py           # SQLite session/message/turn storage
├── rolling_summary.py         # Long-session summary
├── evaluate.py                # Evaluation harness
├── eval_dataset.json          # Evaluation dataset
├── requirements.txt
└── research_sessions.db       # Generated locally
```

---

## Setup

### 1. Clone the repository

```bash
git clone <https://github.com/RahulGupta-30/Sarvam_Assignment>
cd Sarvam_Assignment
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env`

Create a `.env` file in the project root.

Use plain `KEY=value` format. Do not wrap comma-separated model lists in quotes.

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_API_KEY=your_groq_api_key_here

GROQ_MODEL=llama-3.3-70b-versatile

GEMINI_URL_SELECTOR_MODELS=gemini-2.5-flash
GEMINI_FINAL_ANSWER_MODELS=gemini-2.5-flash
GEMINI_CITATION_REPAIR_MODELS=gemini-2.5-flash

MAX_MODEL_RETRIES=4
MODEL_RETRY_BASE_SLEEP_SECONDS=3
MODEL_RETRY_MAX_SLEEP_SECONDS=45

EVAL_JUDGE_MODELS=llama-3.3-70b-versatile
EVAL_JUDGE_MAX_TOKENS=4000

MAX_JUDGE_RETRIES=3
EVAL_RETRY_BASE_SLEEP_SECONDS=8
EVAL_RETRY_MAX_SLEEP_SECONDS=90

EVAL_SLEEP_AFTER_PIPELINE_SECONDS=3
EVAL_SLEEP_BEFORE_JUDGE_CALL_SECONDS=4
EVAL_SLEEP_BETWEEN_TURNS_SECONDS=6
EVAL_SLEEP_BETWEEN_CASES_SECONDS=10
```

Bad example:

```env
GEMINI_URL_SELECTOR_MODELS="gemini-3.5-flash","gemini-2.5-flash"
```

That can create invalid model names with stray quotes.

---

## Run the App

```bash
streamlit run app.py
```

Example query:

```text
What are the current EU AI Act requirements for high-risk AI systems?
```

Example follow-up:

```text
What penalties apply if companies do not comply?
```

The agent should use session history to resolve the follow-up, rewrite it as a standalone query, search fresh sources, and answer with citations.

---

## Run the Evaluation

```bash
python evaluate.py
```

Outputs are saved to:

```text
evaluation_runs/
├── eval_results_<timestamp>.json
└── eval_report_<timestamp>.md
```

---

## Evaluation Methodology

The evaluator separates objective checks from judgment-based quality review.

### Deterministic Checks

1. Citation URL validity
   Checks whether every cited URL exists in the fetched or used source set.

2. Search success
   Checks whether search results were retrieved.

3. Fetch success
   Checks whether page content was fetched.

4. Follow-up rewrite check  
   For follow-up turns, checks whether the standalone question changed.

### Groq-Based LLM Judge

The evaluator uses Groq instead of Gemini for judging so that the evaluator is separate from the generation path.

The judge receives:

- original user question,
- standalone question,
- dataset expectations,
- fetched web context,
- final answer.

It scores from 1 to 5 on:

- factual grounding,
- conflict handling,
- uncertainty calibration,
- answer completeness,
- follow-up resolution quality.

Every score includes a justification.

### Hallucination Probe

A separate Groq call extracts factual claims from the answer and checks whether each claim is traceable to fetched source chunks.

It reports:

- supported claim count,
- unsupported claim count,
- unclear claim count,
- hallucination risk,
- explanations for claims needing review.


## Dataset Coverage

The dataset is designed to test:

- factual questions,
- current events,
- company status,
- regulatory topics,
- multi-hop reasoning,
- comparison questions,
- insufficient-evidence questions,
- conflicting-source questions,
- multi-turn follow-ups,
- adversarial false-premise questions,
- broad questions that should be scoped carefully,
- questions where evidence may not be publicly available.

Supported fields include:

```json
{
  "reference_answer": "Expected answer summary",
  "key_facts": ["Specific fact the answer should cover"],
  "expected_behavior": "How the agent should behave",
  "adversarial_type": "false_premise | outdated_sources | broad_scope | unfindable"
}
```

---

## Example Conversation

### Turn 1

**User**

```text
Summarize the global semiconductor shortage and its impact on the automotive industry.
```

**Agent behavior**

- plans a search strategy,
- searches Tavily,
- selects sources,
- fetches page content,
- builds context,
- answers with citations,
- stores the research turn.

### Turn 2

**User**

```text
How did major automakers respond?
```

**Agent behavior**

- uses session history to resolve the follow-up,
- rewrites the query,
- searches fresh sources,
- answers from current fetched context,
- cites sources.

---

## Success Metrics

The most important quality signals are:

1. citation URL validity,
2. factual grounding,
3. answer completeness,
4. uncertainty and conflict handling,
5. session continuity for follow-up questions,
6. hallucination risk.

---

## Assumptions

- Tavily is used for search and extraction.
- Gemini is used in the research pipeline for source selection and answer generation.
- Groq is used for evaluation judging and hallucination probing.
- SQLite is sufficient for local persistence.
- Previous answers are used for conversation understanding only, not as factual evidence.
- Final factual claims should come from newly fetched web context.

---

## Limitations

- Citation URL validation does not prove semantic support.
- Context ranking is lightweight and keyword-based.
- Some sources may fail extraction.
- Conflict detection is mostly prompt-driven.
- API rate limits and model access issues can affect reliability.
- SQLite is not ideal for production multi-user use.
- The system may miss paywalled, JavaScript-heavy, or blocked pages.

---

## Future Improvements

1. Add semantic reranking or embeddings for context selection.
2. Add stricter claim-level verification before final answers.
3. Add source quality classification.
4. Add stronger conflict detection before generation.
5. Add better handling when too few sources are fetched.
6. Add PDF export for reports.
7. Move all provider/model settings into a central config file.
8. Add PostgreSQL for multi-user deployment.

---

