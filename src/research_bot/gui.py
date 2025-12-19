"""
CustomTkinter GUI for the Research Bot.

Run with: python src/research_bot/gui.py
Or:       research-bot-gui (if installed via pip)
"""

import sys
import threading
import asyncio
from datetime import datetime
from pathlib import Path

# Handle running directly vs as installed package
if __name__ == "__main__" and __package__ is None:
    # Running as script - add src to path for imports
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from research_bot.config import Config
    from research_bot.agents.research_agent import ResearchAgent, ResearchResult
else:
    # Running as module or installed package
    from .config import Config
    from .agents.research_agent import ResearchAgent, ResearchResult

import customtkinter as ctk


# Appearance settings
ctk.set_appearance_mode("dark")  # "dark", "light", or "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class ResearchBotGUI(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Research Bot")
        self.geometry("900x700")
        self.minsize(700, 500)

        # State
        self.is_researching = False
        self.current_result: ResearchResult | None = None
        self.total_cost = 0.0  # Track cumulative cost for session

        # Build UI
        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self):
        """Create all UI components."""

        # === Top Frame: Query Input ===
        self.top_frame = ctk.CTkFrame(self)

        self.query_label = ctk.CTkLabel(
            self.top_frame,
            text="Research Query:",
            font=ctk.CTkFont(size=14, weight="bold"),
        )

        self.query_entry = ctk.CTkEntry(
            self.top_frame,
            placeholder_text="Enter your research topic...",
            font=ctk.CTkFont(size=14),
            height=40,
        )

        self.search_button = ctk.CTkButton(
            self.top_frame,
            text="Research",
            command=self._on_research_click,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=120,
        )

        # === Settings Frame ===
        self.settings_frame = ctk.CTkFrame(self)

        self.iterations_label = ctk.CTkLabel(
            self.settings_frame,
            text="Max Iterations:",
        )

        self.iterations_slider = ctk.CTkSlider(
            self.settings_frame,
            from_=1,
            to=15,
            number_of_steps=14,
            command=self._on_slider_change,
        )
        self.iterations_slider.set(5)

        self.iterations_value = ctk.CTkLabel(
            self.settings_frame,
            text="5",
            width=30,
        )

        self.model_label = ctk.CTkLabel(
            self.settings_frame,
            text="Model:",
        )

        self.model_dropdown = ctk.CTkComboBox(
            self.settings_frame,
            values=[
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-3-5-20241022",
            ],
            width=220,
        )
        self.model_dropdown.set("claude-sonnet-4-20250514")

        # === Status Frame ===
        self.status_frame = ctk.CTkFrame(self)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12),
        )

        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.set(0)

        # === Cost Frame ===
        self.cost_frame = ctk.CTkFrame(self)

        self.cost_label = ctk.CTkLabel(
            self.cost_frame,
            text="API Cost:",
            font=ctk.CTkFont(size=12, weight="bold"),
        )

        self.last_cost_label = ctk.CTkLabel(
            self.cost_frame,
            text="Last: $0.0000",
            font=ctk.CTkFont(size=11),
        )

        self.session_cost_label = ctk.CTkLabel(
            self.cost_frame,
            text="Session: $0.0000",
            font=ctk.CTkFont(size=11),
        )

        self.tokens_label = ctk.CTkLabel(
            self.cost_frame,
            text="Tokens: 0 in / 0 out",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )

        # === Results Notebook (Tabbed View) ===
        self.tabview = ctk.CTkTabview(self)
        self.tabview.add("Summary")
        self.tabview.add("Sources")
        self.tabview.add("Usage")

        # Summary tab
        self.summary_text = ctk.CTkTextbox(
            self.tabview.tab("Summary"),
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )

        # Sources tab
        self.sources_text = ctk.CTkTextbox(
            self.tabview.tab("Sources"),
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )

        # Usage tab
        self.usage_text = ctk.CTkTextbox(
            self.tabview.tab("Usage"),
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )

        # === Bottom Frame: Actions ===
        self.bottom_frame = ctk.CTkFrame(self)

        self.save_button = ctk.CTkButton(
            self.bottom_frame,
            text="Save Results",
            command=self._on_save_click,
            state="disabled",
            width=120,
        )

        self.clear_button = ctk.CTkButton(
            self.bottom_frame,
            text="Clear",
            command=self._on_clear_click,
            fg_color="gray",
            hover_color="darkgray",
            width=100,
        )

        self.reset_cost_button = ctk.CTkButton(
            self.bottom_frame,
            text="Reset Cost",
            command=self._on_reset_cost_click,
            fg_color="#8B4513",
            hover_color="#A0522D",
            width=100,
        )

        self.theme_switch = ctk.CTkSwitch(
            self.bottom_frame,
            text="Dark Mode",
            command=self._toggle_theme,
        )
        self.theme_switch.select()  # Dark mode on by default

    def _layout_widgets(self):
        """Arrange widgets in the window."""

        # Top frame - query input
        self.top_frame.pack(fill="x", padx=15, pady=(15, 10))
        self.query_label.pack(anchor="w", padx=10, pady=(10, 5))
        self.query_entry.pack(side="left", fill="x", expand=True, padx=(10, 10), pady=(0, 10))
        self.search_button.pack(side="right", padx=(0, 10), pady=(0, 10))

        # Settings frame
        self.settings_frame.pack(fill="x", padx=15, pady=5)
        self.iterations_label.pack(side="left", padx=(10, 5), pady=10)
        self.iterations_slider.pack(side="left", padx=5, pady=10)
        self.iterations_value.pack(side="left", padx=(0, 20), pady=10)
        self.model_label.pack(side="left", padx=(10, 5), pady=10)
        self.model_dropdown.pack(side="left", padx=5, pady=10)

        # Status frame
        self.status_frame.pack(fill="x", padx=15, pady=5)
        self.status_label.pack(side="left", padx=10, pady=10)
        self.progress_bar.pack(side="right", fill="x", expand=True, padx=10, pady=10)

        # Cost frame
        self.cost_frame.pack(fill="x", padx=15, pady=5)
        self.cost_label.pack(side="left", padx=10, pady=10)
        self.last_cost_label.pack(side="left", padx=15, pady=10)
        self.session_cost_label.pack(side="left", padx=15, pady=10)
        self.tokens_label.pack(side="right", padx=10, pady=10)

        # Tabview - results
        self.tabview.pack(fill="both", expand=True, padx=15, pady=10)
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.sources_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.usage_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Bottom frame - actions
        self.bottom_frame.pack(fill="x", padx=15, pady=(5, 15))
        self.save_button.pack(side="left", padx=10, pady=10)
        self.clear_button.pack(side="left", padx=5, pady=10)
        self.reset_cost_button.pack(side="left", padx=5, pady=10)
        self.theme_switch.pack(side="right", padx=10, pady=10)

        # Bind Enter key to search
        self.query_entry.bind("<Return>", lambda e: self._on_research_click())

    # === Event Handlers ===

    def _on_slider_change(self, value: float):
        """Update iterations label when slider moves."""
        self.iterations_value.configure(text=str(int(value)))

    def _on_research_click(self):
        """Start research when button clicked."""
        query = self.query_entry.get().strip()
        if not query:
            self._show_error("Please enter a research query.")
            return

        if self.is_researching:
            return

        # Get settings
        max_iterations = int(self.iterations_slider.get())
        model = self.model_dropdown.get()

        # Start research in background thread
        self._start_research(query, max_iterations, model)

    def _on_save_click(self):
        """Save results to file."""
        if not self.current_result:
            return

        # Ask for save location
        filepath = ctk.filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"research_{datetime.now():%Y%m%d_%H%M%S}.txt",
        )

        if filepath:
            self._save_to_file(filepath)

    def _on_clear_click(self):
        """Clear all results."""
        self.summary_text.delete("1.0", "end")
        self.sources_text.delete("1.0", "end")
        self.usage_text.delete("1.0", "end")
        self.current_result = None
        self.save_button.configure(state="disabled")
        self.progress_bar.set(0)
        self._set_status("Ready")

    def _on_reset_cost_click(self):
        """Reset session cost counter."""
        self.total_cost = 0.0
        self.session_cost_label.configure(text="Session: $0.0000")
        self.last_cost_label.configure(text="Last: $0.0000")
        self.tokens_label.configure(text="Tokens: 0 in / 0 out")

    def _toggle_theme(self):
        """Switch between dark and light mode."""
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")

    # === Research Logic ===

    def _start_research(self, query: str, max_iterations: int, model: str):
        """Run research in a background thread."""
        self.is_researching = True
        self.search_button.configure(state="disabled", text="Researching...")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self._set_status(f"Researching: {query[:50]}...")

        # Clear previous results
        self.summary_text.delete("1.0", "end")
        self.sources_text.delete("1.0", "end")
        self.usage_text.delete("1.0", "end")

        def run():
            try:
                # Build config
                config = Config.from_env()
                config.max_iterations = max_iterations
                config.model = model
                config.validate()

                # Run research
                agent = ResearchAgent(config)
                result = asyncio.run(agent.research(query))

                # Update UI on main thread
                self.after(0, lambda: self._on_research_complete(result))

            except ValueError as e:
                self.after(0, lambda: self._show_error(str(e)))
                self.after(0, self._reset_ui)
            except Exception as e:
                self.after(0, lambda: self._show_error(f"Research failed: {e}"))
                self.after(0, self._reset_ui)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _on_research_complete(self, result: ResearchResult):
        """Handle completed research."""
        self.current_result = result
        self._reset_ui()

        # Update cost tracking
        cost = result.usage.cost_usd
        self.total_cost += cost
        self.last_cost_label.configure(text=f"Last: ${cost:.4f}")
        self.session_cost_label.configure(text=f"Session: ${self.total_cost:.4f}")
        self.tokens_label.configure(
            text=f"Tokens: {result.usage.input_tokens:,} in / {result.usage.output_tokens:,} out"
        )

        # Display summary with completion status
        if not result.completed:
            header = "[Note: Max iterations reached - showing partial results]\n\n"
            self.summary_text.insert("1.0", header + result.summary)
        else:
            self.summary_text.insert("1.0", result.summary)

        # Display sources
        if result.sources:
            sources_text = f"Found {len(result.sources)} sources:\n\n"
            for i, src in enumerate(result.sources, 1):
                sources_text += f"{i}. {src.get('title', 'Untitled')}\n"
                sources_text += f"   {src.get('url', 'No URL')}\n\n"
        else:
            sources_text = "No sources were fetched during research."

        self.sources_text.insert("1.0", sources_text)

        # Display usage details
        usage_text = f"""API Usage Details
=================

Model: {result.usage.model}

Tokens:
  - Input:  {result.usage.input_tokens:,}
  - Output: {result.usage.output_tokens:,}
  - Total:  {result.usage.total_tokens:,}

Cost Breakdown:
  - This query:    ${cost:.4f}
  - Session total: ${self.total_cost:.4f}

Research Stats:
  - Iterations: {result.iterations}
  - Sources fetched: {len(result.sources)}
"""
        self.usage_text.insert("1.0", usage_text)

        # Update status with cost and completion state
        status_prefix = "Complete" if result.completed else "Partial"
        self._set_status(
            f"{status_prefix} - {result.iterations} iterations, {len(result.sources)} sources, ${cost:.4f}"
        )
        self.save_button.configure(state="normal")

        # Switch to summary tab
        self.tabview.set("Summary")

    def _reset_ui(self):
        """Reset UI after research completes."""
        self.is_researching = False
        self.search_button.configure(state="normal", text="Research")
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(1.0)

    # === Helpers ===

    def _set_status(self, text: str):
        """Update status label."""
        self.status_label.configure(text=text)

    def _show_error(self, message: str):
        """Show error in the UI."""
        self._set_status(f"Error: {message}")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", f"ERROR: {message}\n\nMake sure ANTHROPIC_API_KEY is set.")

    def _save_to_file(self, filepath: str):
        """Save results to a text file."""
        if not self.current_result:
            return

        result = self.current_result
        content = f"""Research Results
================
Query: {result.query}
Date: {datetime.now():%Y-%m-%d %H:%M:%S}
Model: {result.usage.model}
Iterations: {result.iterations}
Sources: {len(result.sources)}

API Usage:
  Input tokens:  {result.usage.input_tokens:,}
  Output tokens: {result.usage.output_tokens:,}
  Total tokens:  {result.usage.total_tokens:,}
  Cost: ${result.usage.cost_usd:.4f}

Summary
-------
{result.summary}

Sources
-------
"""
        for i, src in enumerate(result.sources, 1):
            content += f"{i}. {src.get('title', 'Untitled')}\n"
            content += f"   {src.get('url', 'No URL')}\n\n"

        Path(filepath).write_text(content, encoding="utf-8")
        self._set_status(f"Saved to {Path(filepath).name}")


def main():
    """Launch the GUI application."""
    app = ResearchBotGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
