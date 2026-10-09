# ResearchMind 🔬

A multi-agent AI research pipeline built with **LangChain** and **Streamlit**. Four specialized agents collaborate to turn a single topic into a polished, critiqued research report:

1. **Search Agent** — searches the web (via Tavily) for recent, reliable information on the topic.
2. **Reader Agent** — picks the most relevant URL from the search results and scrapes it for deeper content.
3. **Writer Chain** — synthesizes the search results and scraped content into a structured, multi-section research report.
4. **Critic Chain** — rigorously reviews the report and returns a score, strengths, weaknesses, and suggested next steps.

The LLM backing all agents is served by Groq. The model is configured with `GROQ_MODEL` in the project `.env` file.

---

## Project Structure

```
.
├── app.py              # Streamlit UI
├── agents.py           # Agent + chain definitions (search, reader, writer, critic)
├── pipeline.py          # CLI entry point that runs the full pipeline
├── tools.py             # Tools used by agents (web_search, scrape_url)
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10+
- A [Tavily](https://tavily.com) API key (free tier available)
- A [Groq](https://console.groq.com/keys) API key

## Local Setup

1. **Clone the repo and install dependencies:**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the project root (this file is gitignored — never commit it):

   ```env
   TAVILY_API_KEY=your_tavily_key_here
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```

3. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

   Or run the pipeline from the command line instead:

   ```bash
   python pipeline.py
   ```

## ⚠️ Security Note

If you ever had real API keys sitting in a plain file (`.env`, `_env`, or otherwise) that was committed to git or shared elsewhere, **rotate those keys immediately**:
- Tavily: regenerate at https://app.tavily.com
- Groq: regenerate at https://console.groq.com/keys

Never commit `.env` files. The included `.gitignore` excludes them by default.

## Deployment

See the deployment guide below for shipping this to **Streamlit Community Cloud**.
