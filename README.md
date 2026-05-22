# Deep Research Agent

A Python-based Deep Research Agent that searches the web, fetches page content, builds a bounded research context, and generates citation-grounded answers with persistent session memory.

The agent gathers multiple web pages , extracts their content, to generate the final result for deep research. The deep research terminology means gathering data from a wide variety of sources and presenting it to the user.

---

## Demo

**Video demo:** TODO: add your Loom/Drive/YouTube link here  
**Repository:** TODO: add your GitHub repository link here  
**Live app:** Optional: add Streamlit Cloud/Hugging Face Spaces link if deployed

---

## Features

- Web research using Tavily search.
- Page extraction using Tavily extract.
- Gemini-powered query planning for follow-up questions.
- Gemini-powered URL selection from search results.
- Concurrent page fetching for faster source acquisition.
- Context builder that ranks and limits selected snippets.
- Citation-grounded final answers using source title, domain, and URL.
- Citation validation to detect invented or unsupported URLs.
- Conflict and uncertainty handling in final responses.
- Persistent SQLite sessions.
- Conversation history and research turn history.
- Rolling summary for long sessions.
- Streamlit UI with progress updates and research trace.
- Evaluation harness with factual, multi-hop, comparison, insufficient-evidence, conflicting-source, and multi-turn tests.

---

## Architecture Overview

```text
User question
   ↓
Streamlit UI
   ↓
SQLite session lookup
   ↓
Rolling summary update
   ↓
Query planner
   ↓
Tavily search
   ↓
Gemini URL selector
   ↓
Concurrent Tavily page extraction
   ↓
Context builder
   ↓
Gemini final answer generator
   ↓
Citation validator and optional repair
   ↓
SQLite turn storage
   ↓
Answer + research trace in UI
```

---

## Project Structure

```text
.
├── app.py                    # Streamlit UI
├── research_pipeline.py       # Main research orchestration pipeline
├── agent.py                   # Gemini URL selection and answer generation
├── query_planner.py           # Rewrites follow-up questions into standalone research queries
├── search_tavily.py           # Tavily search module
├── fetch_page.py              # Tavily page extraction module
├── build_context.py           # Chunking, ranking, and context construction
├── session_store.py           # SQLite session, message, and turn storage
├── rolling_summary.py         # Long-history summarization
├── citation_validator.py      # Citation integrity checks
├── evaluate.py                # Evaluation harness
├── eval_dataset.json          # Evaluation dataset
├── requirements.txt
├── .env.example
└── research_sessions.db       # Created automatically; should not be committed
```



---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```


### 4. Create `.env`

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## Run the App

```bash
streamlit run app.py
```

Open the Streamlit URL shown in the terminal. Ask a research question such as:

```text
What is the current status of the US-Iran war?
```

Then ask a follow-up:

```text
What impact did it have on oil prices and shipping?
```

The second query should use session history to rewrite the question into a standalone research query before searching.

---

## How It Works

### 1. Query Planning

The query planner receives:

- the current user question,
- recent research turns,
- the rolling session summary.

It returns:

- whether the question is a follow-up,
- a standalone research question,
- one or more Tavily search queries,
- a short explanation of which history was used.

Example:

```json
{
  "is_follow_up": true,
  "standalone_question": "What was the impact of the US-Iran war on oil prices and shipping through the Strait of Hormuz?",
  "search_queries": [
    "US Iran war oil prices shipping Strait of Hormuz impact",
    "US Iran conflict impact on Brent crude oil and tanker routes"
  ],
  "history_used": "Used the previous turn about the US-Iran war to resolve 'it'."
}
```

### 2. Search

The search module sends the planner-generated queries to Tavily. Results are normalized into:

```json
{
  "title": "...",
  "url": "...",
  "domain": "...",
  "score": 0.91,
  "snippet": "...",
  "search_query": "..."
}
```

### 3. URL Selection

Gemini reviews the search results and selects the most useful URLs based on:

- relevance,
- source quality,
- recency when available,
- source diversity.

### 4. Page Fetching

Selected URLs are fetched concurrently. Each extracted page stores:

- URL,
- title,
- domain,
- retrieved timestamp,
- readable text.

### 5. Context Building

The context builder splits pages into chunks, scores them against the standalone query, and builds a bounded prompt context. It preserves source metadata so the final answer can cite correctly.

### 6. Final Answer Generation

Gemini receives:

- the original user question,
- the standalone research question,
- session context,
- selected web context.

It must answer using the current web context for factual claims and cite using:

```text
[Title — domain](URL)
```

### 7. Citation Validation

The citation validator checks whether cited URLs in the final answer were present in the selected context sources. If citations are missing or invented, the answer can be repaired with another Gemini call.

### 8. Session Memory

SQLite stores:

- sessions,
- user and assistant messages,
- turn-level research artifacts,
- rolling summaries.

The rolling summary compresses older messages so long conversations remain usable without sending the entire chat history to the model.

---

## Evaluation Harness

Run:

```bash
python evaluate.py
```

The evaluation script runs the agent on `eval_dataset.json` and writes results to:

```text
evaluation_runs/
├── eval_results_<timestamp>.json
└── eval_report_<timestamp>.md
```

### Dataset Coverage

The dataset includes:

- factual questions,
- multi-hop questions,
- comparison questions,
- insufficient-evidence questions,
- conflicting-source questions,
- multi-turn follow-up questions.

### Metrics

The evaluator scores each turn on:

- citation presence,
- citation integrity,
- source diversity,
- selected context usage,
- answer usefulness and length,
- required term coverage,
- uncertainty handling,
- conflict handling,
- follow-up resolution,
- search/fetch success.



## Example Conversation

### Turn 1

**User:**

```text
Summarize the US-Iran war in brief.
```

**Agent behavior:**

- plans search strategy,
- searches Tavily,
- selects sources,
- fetches pages,
- builds context,
- answers with citations,
- stores turn history.

### Turn 2

**User:**

```text
What impact did it have on oil prices and shipping?
```

**Agent behavior:**

- uses session history to resolve `it` as the US-Iran war,
- creates standalone query about oil prices and shipping impact,
- searches fresh sources,
- answers using newly fetched web context.

---

## Assumptions

- Tavily is used as the web search and page extraction provider.
- Gemini is used for query planning, URL selection, summarization, citation repair, and rolling summary generation.
- SQLite is sufficient for local persistence in this assignment; a production multi-user deployment would use PostgreSQL.
- Previous assistant answers are used only for conversation understanding, not as factual evidence.
- Final factual claims should be grounded in the current fetched web context.
- Some web pages may fail extraction or return incomplete text; the agent handles this by continuing with available sources.

---

## Limitations

- Citation validation checks URL integrity, not full semantic support for every sentence.
- Source recency depends on available metadata and page content.
- The context builder uses lightweight relevance scoring; embeddings or reranking could improve snippet selection.
- Tavily and Gemini rate limits can affect reliability.
- Live news topics may contain conflicting or rapidly changing information.
- The system may miss paywalled or JavaScript-heavy pages.

---

## Future Improvements

1. Add semantic citation verification using an LLM judge or embedding-based evidence matching.
2. Add source quality classification such as official, news, think tank, encyclopedia, blog, or primary document.
3. Add PostgreSQL support for multi-user production deployment.
4. Add retry/backoff handling for API rate limits and transient fetch failures.
5. Add stronger conflict detection before final answer generation.
6. Add exportable PDF research reports.

---


