"""
Streamlit Web Interface for Research Bot.

LEARNING NOTE: This demonstrates creating a web app with pure Python!
Streamlit handles all the HTML/CSS/JavaScript for you.

Run with: streamlit run src/research_bot/web_streamlit.py
Or: streamlit run web_streamlit.py (from src/research_bot directory)
"""

import streamlit as st
import asyncio
from datetime import datetime

# LEARNING NOTE - Streamlit Imports:
# Import our existing bot code - the web interface just wraps it!
from research_bot.config import Config
from research_bot.agents.research_agent import ResearchAgent


# LEARNING NOTE - Page Configuration:
# This sets up the browser tab title, icon, and layout
# Must be the first Streamlit command
st.set_page_config(
    page_title="Research Bot",
    page_icon="🔍",
    layout="wide",  # Use full width of browser
    initial_sidebar_state="expanded"  # Sidebar open by default
)


# LEARNING NOTE - Session State:
# Streamlit reruns the entire script on every interaction!
# Session state persists data across reruns
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'research_history' not in st.session_state:
    st.session_state.research_history = []


# === Main UI ===

# Title and description
st.title("🔍 AI Research Bot")
st.markdown(
    "Ask any question and get AI-powered research with sources and citations!"
)
st.divider()


# === Sidebar: Settings and Info ===

st.sidebar.header("⚙️ Settings")

# Max iterations slider
max_iterations = st.sidebar.slider(
    "Max Research Iterations",
    min_value=1,
    max_value=15,
    value=5,
    help="More iterations = more thorough research but higher cost"
)

# Model selection
model = st.sidebar.selectbox(
    "Claude Model",
    options=[
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-3-5-20241022"
    ],
    index=0,  # Default to Sonnet
    help="Sonnet: balanced, Opus: most powerful, Haiku: fastest/cheapest"
)

st.sidebar.divider()

# Cost tracking
st.sidebar.header("💰 Cost Tracking")
st.sidebar.metric(
    "Session Total",
    f"${st.session_state.total_cost:.4f}",
    help="Total API cost this session"
)

# Reset cost button
if st.sidebar.button("Reset Cost Counter"):
    st.session_state.total_cost = 0.0
    st.rerun()  # Refresh the page

st.sidebar.divider()

# Info section
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    This bot uses Claude AI to:
    1. Search the web for information
    2. Fetch and analyze relevant sources
    3. Synthesize a comprehensive answer

    **Tip:** More iterations = better research but higher cost!
    """
)


# === Main Content: Query Input ===

# LEARNING NOTE - Columns:
# Create multiple columns for layout
col1, col2 = st.columns([4, 1])

with col1:
    # Text input for query
    query = st.text_input(
        "Research Query:",
        placeholder="e.g., What are the latest developments in quantum computing?",
        label_visibility="collapsed"
    )

with col2:
    # Research button
    # LEARNING NOTE - type="primary" makes it blue and prominent
    search_button = st.button("🔍 Research", type="primary", use_container_width=True)


# === Research Logic ===

if search_button:
    # Validate input
    if not query or not query.strip():
        st.error("⚠️ Please enter a research query!")
    else:
        # LEARNING NOTE - st.spinner():
        # Shows a loading animation while code runs
        # Everything inside 'with' block runs while spinner is active
        with st.spinner(f"Researching: {query}..."):
            try:
                # Build configuration
                config = Config.from_env()
                config.max_iterations = max_iterations
                config.model = model
                config.validate()

                # Run research
                # LEARNING NOTE - asyncio.run():
                # Our bot is async, but Streamlit isn't
                # Use asyncio.run() to bridge the gap
                agent = ResearchAgent(config)
                result = asyncio.run(agent.research(query))

                # Update session cost
                st.session_state.total_cost += result.usage.cost_usd

                # Add to history
                st.session_state.research_history.insert(0, {
                    'query': query,
                    'time': datetime.now(),
                    'cost': result.usage.cost_usd
                })

                # LEARNING NOTE - Success message:
                # Shows a green success banner
                st.success(
                    f"✅ Research complete! "
                    f"({result.iterations} iterations, "
                    f"{len(result.sources)} sources, "
                    f"${result.usage.cost_usd:.4f})"
                )

                # === Display Results in Tabs ===

                # LEARNING NOTE - Tabs:
                # Creates a tabbed interface like in the GUI
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📝 Summary", "📚 Sources", "📊 Usage", "📜 History"]
                )

                # Summary Tab
                with tab1:
                    if not result.completed:
                        st.warning(
                            "⚠️ Maximum iterations reached. "
                            "Results may be incomplete. "
                            "Try increasing max iterations."
                        )

                    st.markdown("### Research Summary")
                    st.markdown(result.summary)

                    # Download button
                    # LEARNING NOTE - st.download_button():
                    # Creates a button that triggers file download
                    st.download_button(
                        label="📥 Download Summary",
                        data=result.summary,
                        file_name=f"research_{datetime.now():%Y%m%d_%H%M%S}.txt",
                        mime="text/plain"
                    )

                # Sources Tab
                with tab2:
                    st.markdown("### Sources Consulted")

                    if result.sources:
                        st.info(f"📚 Found {len(result.sources)} sources")

                        # Display each source
                        for i, source in enumerate(result.sources, 1):
                            # LEARNING NOTE - st.expander():
                            # Creates a collapsible section
                            with st.expander(f"Source {i}: {source.get('title', 'Untitled')}"):
                                st.markdown(f"**URL:** [{source.get('url', '')}]({source.get('url', '')})")

                                # Show excerpt if available
                                idx = i - 1
                                if idx < len(result.raw_findings):
                                    st.markdown("**Excerpt:**")
                                    st.text(result.raw_findings[idx][:500] + "...")
                    else:
                        st.warning("No sources were fetched during this research.")

                # Usage Tab
                with tab3:
                    st.markdown("### API Usage Details")

                    # LEARNING NOTE - st.columns() for metrics:
                    # Display multiple metrics side-by-side
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Iterations",
                            result.iterations,
                            help="Number of research loops performed"
                        )

                    with col2:
                        st.metric(
                            "Total Tokens",
                            f"{result.usage.total_tokens:,}",
                            help="Input + output tokens"
                        )

                    with col3:
                        st.metric(
                            "Cost (USD)",
                            f"${result.usage.cost_usd:.4f}",
                            delta=f"-${result.usage.cost_usd:.4f}",
                            delta_color="inverse"  # Show decrease as red
                        )

                    st.divider()

                    # Detailed breakdown
                    st.markdown("#### Token Breakdown")

                    # LEARNING NOTE - st.json():
                    # Display JSON data in an expandable, formatted view
                    st.json({
                        "model": result.usage.model,
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "sources_fetched": len(result.sources),
                        "completed": result.completed
                    })

                # History Tab
                with tab4:
                    st.markdown("### Research History (This Session)")

                    if st.session_state.research_history:
                        # LEARNING NOTE - st.dataframe():
                        # Display data in an interactive table
                        import pandas as pd

                        history_df = pd.DataFrame([
                            {
                                'Query': item['query'][:50] + ('...' if len(item['query']) > 50 else ''),
                                'Time': item['time'].strftime('%H:%M:%S'),
                                'Cost': f"${item['cost']:.4f}"
                            }
                            for item in st.session_state.research_history
                        ])

                        st.dataframe(
                            history_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No research history yet. Start by submitting a query!")

            except ValueError as e:
                # Config validation error (e.g., missing API key)
                st.error(f"❌ Configuration Error: {str(e)}")
                st.info(
                    "💡 Make sure ANTHROPIC_API_KEY is set in your environment:\n\n"
                    "```bash\n"
                    "export ANTHROPIC_API_KEY='your-key'\n"
                    "```"
                )

            except Exception as e:
                # Unexpected error
                st.error(f"❌ Research Failed: {str(e)}")
                st.exception(e)  # Show full traceback for debugging


# === Footer ===

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col1:
    st.caption("Made with ❤️ using Claude AI")

with footer_col2:
    st.caption("Powered by Anthropic Claude & DuckDuckGo")

with footer_col3:
    if st.button("🗑️ Clear History"):
        st.session_state.research_history = []
        st.rerun()


# LEARNING NOTE - How Streamlit Works:
# Every interaction (button click, text input, etc.) reruns this entire script!
# That's why we use session_state to persist data across reruns.
# It might seem inefficient, but Streamlit is smart about caching and only
# updates what changed.
