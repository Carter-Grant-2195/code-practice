# Web Interface Options for Research Bot

## Overview

There are three main approaches to creating a web interface for your research bot:

1. **Streamlit** - Simplest, data science focused
2. **FastAPI + HTML** - Modern, async-friendly, professional
3. **Flask + HTML** - Traditional, simple, widely used

## Comparison Table

| Feature | Streamlit | FastAPI | Flask |
|---------|-----------|---------|-------|
| **Learning Curve** | Easiest | Medium | Easy |
| **Lines of Code** | ~50 | ~150 | ~100 |
| **Async Support** | Limited | Native | Add-on |
| **Customization** | Limited | Full | Full |
| **Best For** | Quick prototypes, demos | Production APIs, modern apps | Traditional web apps |
| **UI Complexity** | Auto-generated | Full control | Full control |
| **Deployment** | Easy | Medium | Easy |

## Option 1: Streamlit (RECOMMENDED FOR BEGINNERS)

### Why Choose Streamlit?
- **Fastest to implement:** 30-50 lines of Python
- **No HTML/CSS/JavaScript needed:** Pure Python
- **Built-in components:** Buttons, sliders, text boxes automatically styled
- **Auto-refresh:** UI updates automatically when code changes
- **Perfect for:** Demos, prototypes, data science apps

### Example Streamlit App

```python
import streamlit as st
import asyncio
from research_bot.config import Config
from research_bot.agents.research_agent import ResearchAgent

st.set_page_config(page_title="Research Bot", page_icon="🔍")

st.title("🔍 Research Bot")
st.markdown("Ask any question and get AI-powered research!")

# Sidebar settings
st.sidebar.header("Settings")
max_iterations = st.sidebar.slider("Max Iterations", 1, 15, 5)
model = st.sidebar.selectbox(
    "Model",
    ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-3-5-20241022"]
)

# Main interface
query = st.text_input("Research Query:", placeholder="What do you want to research?")
search_button = st.button("🔍 Research", type="primary")

if search_button and query:
    with st.spinner(f"Researching: {query}..."):
        # Build config
        config = Config.from_env()
        config.max_iterations = max_iterations
        config.model = model

        # Run research
        agent = ResearchAgent(config)
        result = asyncio.run(agent.research(query))

    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["📝 Summary", "📚 Sources", "💰 Usage"])

    with tab1:
        st.markdown(result.summary)

    with tab2:
        if result.sources:
            for i, source in enumerate(result.sources, 1):
                st.markdown(f"**{i}. {source.get('title', 'Untitled')}**")
                st.markdown(f"[{source.get('url', '')}]({source.get('url', '')})")
                st.divider()
        else:
            st.info("No sources fetched")

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Iterations", result.iterations)
        with col2:
            st.metric("Total Tokens", f"{result.usage.total_tokens:,}")
        with col3:
            st.metric("Cost", f"${result.usage.cost_usd:.4f}")

        st.json({
            "model": result.usage.model,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "sources": len(result.sources)
        })
```

### Running Streamlit

```bash
# Install
pip install streamlit

# Run
streamlit run web_streamlit.py

# Automatically opens in browser at http://localhost:8501
```

### Pros of Streamlit
- ✅ Extremely fast to develop
- ✅ No frontend knowledge needed
- ✅ Professional-looking UI out of the box
- ✅ Great for sharing with non-technical users
- ✅ Built-in deployment options (Streamlit Cloud)

### Cons of Streamlit
- ❌ Less customization control
- ❌ Full page refreshes on interaction
- ❌ Limited to Streamlit's components
- ❌ Not ideal for complex multi-page apps

---

## Option 2: FastAPI (BEST FOR PRODUCTION)

### Why Choose FastAPI?
- **Modern and fast:** Built on async Python
- **API-first:** Easy to integrate with other services
- **Interactive docs:** Automatic API documentation
- **Type checking:** Uses Python type hints
- **Perfect for:** Production apps, APIs, modern web services

### Example FastAPI App

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio

from research_bot.config import Config
from research_bot.agents.research_agent import ResearchAgent

app = FastAPI(title="Research Bot API")

class ResearchRequest(BaseModel):
    query: str
    max_iterations: int = 5
    model: str = "claude-sonnet-4-20250514"

class ResearchResponse(BaseModel):
    query: str
    summary: str
    sources: list
    iterations: int
    cost: float
    completed: bool

@app.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """Perform research on a query."""
    try:
        config = Config.from_env()
        config.max_iterations = request.max_iterations
        config.model = request.model
        config.validate()

        agent = ResearchAgent(config)
        result = await agent.research(request.query)

        return ResearchResponse(
            query=result.query,
            summary=result.summary,
            sources=result.sources,
            iterations=result.iterations,
            cost=result.usage.cost_usd,
            completed=result.completed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Bot</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            input, button { font-size: 16px; padding: 10px; margin: 10px 0; }
            #query { width: 100%; }
            #result { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-top: 20px; }
            .loading { color: #666; }
        </style>
    </head>
    <body>
        <h1>🔍 Research Bot</h1>
        <input type="text" id="query" placeholder="Enter research query...">
        <button onclick="research()">Research</button>
        <div id="result"></div>

        <script>
        async function research() {
            const query = document.getElementById('query').value;
            const resultDiv = document.getElementById('result');

            resultDiv.innerHTML = '<p class="loading">Researching...</p>';

            const response = await fetch('/api/research', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });

            const data = await response.json();

            resultDiv.innerHTML = `
                <h2>Results</h2>
                <p><strong>Cost:</strong> $${data.cost.toFixed(4)}</p>
                <p><strong>Iterations:</strong> ${data.iterations}</p>
                <h3>Summary</h3>
                <p>${data.summary}</p>
                <h3>Sources</h3>
                ${data.sources.map(s => `<p>- ${s.title}<br><a href="${s.url}">${s.url}</a></p>`).join('')}
            `;
        }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running FastAPI

```bash
# Install
pip install fastapi uvicorn

# Run
uvicorn web_fastapi:app --reload

# Open http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Pros of FastAPI
- ✅ Full control over UI/UX
- ✅ Async native (perfect for our bot)
- ✅ Auto-generated API documentation
- ✅ Can build mobile apps on top of API
- ✅ Production-ready performance

### Cons of FastAPI
- ❌ Need to write HTML/CSS/JavaScript
- ❌ More code than Streamlit
- ❌ Steeper learning curve

---

## Option 3: Flask (TRADITIONAL CHOICE)

### Why Choose Flask?
- **Simple and proven:** Been around for years
- **Lots of tutorials:** Huge community
- **Flexible:** Easy to start, scales to complex apps
- **Perfect for:** Traditional web apps, learning

### Example Flask App

```python
from flask import Flask, render_template, request, jsonify
import asyncio

from research_bot.config import Config
from research_bot.agents.research_agent import ResearchAgent

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/research', methods=['POST'])
def research():
    data = request.json
    query = data.get('query')
    max_iterations = data.get('max_iterations', 5)
    model = data.get('model', 'claude-sonnet-4-20250514')

    try:
        config = Config.from_env()
        config.max_iterations = max_iterations
        config.model = model
        config.validate()

        agent = ResearchAgent(config)
        result = asyncio.run(agent.research(query))

        return jsonify({
            'query': result.query,
            'summary': result.summary,
            'sources': result.sources,
            'iterations': result.iterations,
            'cost': result.usage.cost_usd,
            'completed': result.completed
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Running Flask

```bash
# Install
pip install flask

# Run
python web_flask.py

# Open http://localhost:5000
```

### Pros of Flask
- ✅ Simple to learn
- ✅ Lots of documentation
- ✅ Full UI control
- ✅ Easy templating with Jinja2
- ✅ Good for learning web development

### Cons of Flask
- ❌ Not async by default
- ❌ Need to write more boilerplate
- ❌ Less modern than FastAPI

---

## Which Should You Choose?

### Choose **Streamlit** if:
- You want something working in 30 minutes
- You're comfortable with Python but not HTML/CSS
- You want to share a demo quickly
- You don't need heavy customization

### Choose **FastAPI** if:
- You want a production-ready solution
- You might build a mobile app later
- You want automatic API docs
- You're comfortable with HTML/CSS/JS

### Choose **Flask** if:
- You're learning web development
- You want something traditional and proven
- You prefer extensive tutorials
- You don't need async features

## My Recommendation

**Start with Streamlit!** Here's why:
1. You can have a working web app in 20 minutes
2. No HTML/CSS/JavaScript knowledge needed
3. Professional-looking out of the box
4. Easy to share with others
5. You can always build a more custom solution later

**Then, if needed, move to FastAPI** for production use with more control.

## Next Steps

I can help you implement any of these! Would you like me to:
1. Create a complete Streamlit web app?
2. Build a FastAPI application with custom HTML?
3. Develop a Flask-based web interface?

Each would integrate seamlessly with your existing research bot code!

## Deployment Options

Once you build a web interface, you can deploy it:

### Streamlit Deployment
- **Streamlit Cloud:** Free, easiest (streamlit.io)
- **Heroku:** Free tier available
- **AWS/GCP/Azure:** Professional hosting

### FastAPI/Flask Deployment
- **Railway:** Simple, free tier
- **Render:** Easy deployment
- **DigitalOcean:** Full control, $5/month
- **AWS/GCP/Azure:** Enterprise solutions

---

Ready to build a web interface? Let me know which option you'd like to pursue! 🚀
