"""
CustomTkinter GUI for the Research Bot.

LEARNING NOTE: This file demonstrates desktop GUI development in Python:
1. CustomTkinter - Modern, customizable Tkinter (built-in Python GUI library)
2. Threading - Running long tasks without freezing the GUI
3. Event-driven programming - Responding to user actions (button clicks, etc.)
4. MVC-like pattern - Separating UI from business logic

Run with: python src/research_bot/gui.py
Or:       research-bot-gui (if installed via pip)
"""

import sys
import threading  # For running research in background without freezing GUI
import asyncio  # For running async research function
from datetime import datetime
from pathlib import Path

# Handle running directly vs as installed package
if __name__ == "__main__" and __package__ is None:
    # Running as script - add src to path for imports
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from research_bot.config import Config
    from research_bot.agents.research_agent import ResearchAgent, ResearchResult
    from research_bot.exporters import get_exporter
else:
    # Running as module or installed package
    from .config import Config
    from .agents.research_agent import ResearchAgent, ResearchResult
    from .exporters import get_exporter

import customtkinter as ctk


# LEARNING NOTE - Global GUI settings:
# These affect all windows in the application
ctk.set_appearance_mode("dark")  # "dark", "light", or "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class ResearchBotGUI(ctk.CTk):
    """
    Main application window.

    LEARNING NOTE - Object-Oriented GUI:
    We inherit from ctk.CTk (CustomTkinter's main window class).
    This gives us all the window functionality automatically.

    LEARNING NOTE - GUI Structure:
    1. __init__: Set up the window and create UI elements
    2. _create_widgets: Create all buttons, text boxes, etc.
    3. _layout_widgets: Arrange them on screen
    4. Event handlers (_on_*_click): Respond to user actions

    This separation makes the code organized and maintainable.
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()  # Call parent class constructor

        # LEARNING NOTE - Window configuration:
        # These methods set up the basic window properties
        self.title("Research Bot")  # Title bar text
        self.geometry("900x700")  # Width x Height in pixels
        self.minsize(700, 500)  # Minimum window size

        # LEARNING NOTE - Application state:
        # These variables track what's happening in the app
        # Unlike widgets (buttons, etc.), these are just data
        self.is_researching = False  # Prevent multiple simultaneous searches
        self.current_result: ResearchResult | None = None  # Last result
        self.total_cost = 0.0  # Track cumulative cost for session

        # LEARNING NOTE - Two-phase UI construction:
        # 1. Create widgets (buttons, text boxes, etc.)
        # 2. Layout widgets (position them on screen)
        # This separation makes the code cleaner
        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self):
        """
        Create all UI components.

        LEARNING NOTE - What are widgets?
        Widgets are GUI elements: buttons, text boxes, labels, etc.
        Each widget is an object with properties (text, color, size, etc.)

        LEARNING NOTE - Frames:
        Frames are containers that group related widgets together.
        Think of them like invisible boxes that organize your layout.
        """

        # === Top Frame: Query Input ===
        # LEARNING NOTE: self.top_frame means this frame belongs to this window
        # ctk.CTkFrame(self) means "create a frame inside this window"
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
        self.tabview.add("Key Findings")
        self.tabview.add("Extracted Data")
        self.tabview.add("Sources")
        self.tabview.add("Usage")

        # Summary tab
        self.summary_text = ctk.CTkTextbox(
            self.tabview.tab("Summary"),
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )

        # Key Findings tab
        self.findings_text = ctk.CTkTextbox(
            self.tabview.tab("Key Findings"),
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="word",
        )

        # Extracted Data tab
        self.extracted_text = ctk.CTkTextbox(
            self.tabview.tab("Extracted Data"),
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

        # Export format selection
        self.format_label = ctk.CTkLabel(
            self.bottom_frame,
            text="Format:",
            font=ctk.CTkFont(size=12),
        )

        # LEARNING NOTE - Export Format Dropdown:
        # This lets users choose between JSON, Markdown, HTML, and CSV
        # Each format has different use cases (see comments in exporters)
        self.format_dropdown = ctk.CTkComboBox(
            self.bottom_frame,
            values=["Text (.txt)", "JSON (.json)", "Markdown (.md)", "HTML (.html)", "CSV (.csv)"],
            width=140,
        )
        self.format_dropdown.set("Text (.txt)")

        self.save_button = ctk.CTkButton(
            self.bottom_frame,
            text="Export",
            command=self._on_save_click,
            state="disabled",
            width=100,
        )

        self.clear_button = ctk.CTkButton(
            self.bottom_frame,
            text="Clear",
            command=self._on_clear_click,
            fg_color="gray",
            hover_color="darkgray",
            width=80,
        )

        self.reset_cost_button = ctk.CTkButton(
            self.bottom_frame,
            text="Reset Cost",
            command=self._on_reset_cost_click,
            fg_color="#8B4513",
            hover_color="#A0522D",
            width=90,
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
        self.findings_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.extracted_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.sources_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.usage_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Bottom frame - actions
        self.bottom_frame.pack(fill="x", padx=15, pady=(5, 15))
        self.format_label.pack(side="left", padx=(10, 5), pady=10)
        self.format_dropdown.pack(side="left", padx=5, pady=10)
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
        """
        Save/export results to file.

        LEARNING NOTE - Export Format Selection:
        We use the format dropdown to determine which exporter to use.
        Each format has different extensions and file type filters.
        """
        if not self.current_result:
            return

        # Get selected format
        format_selection = self.format_dropdown.get()

        # Map format selection to exporter type and extension
        format_map = {
            "Text (.txt)": ("txt", ".txt", [("Text files", "*.txt")]),
            "JSON (.json)": ("json", ".json", [("JSON files", "*.json")]),
            "Markdown (.md)": ("markdown", ".md", [("Markdown files", "*.md")]),
            "HTML (.html)": ("html", ".html", [("HTML files", "*.html")]),
            "CSV (.csv)": ("csv", ".csv", [("CSV files", "*.csv")]),
        }

        format_type, extension, filetypes = format_map.get(
            format_selection, ("txt", ".txt", [("Text files", "*.txt")])
        )
        filetypes.append(("All files", "*.*"))

        # Ask for save location
        filepath = ctk.filedialog.asksaveasfilename(
            defaultextension=extension,
            filetypes=filetypes,
            initialfile=f"research_{datetime.now():%Y%m%d_%H%M%S}{extension}",
        )

        if filepath:
            self._save_to_file(filepath, format_type)

    def _on_clear_click(self):
        """Clear all results."""
        self.summary_text.delete("1.0", "end")
        self.findings_text.delete("1.0", "end")
        self.extracted_text.delete("1.0", "end")
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
        """
        Run research in a background thread.

        LEARNING NOTE - Why threading?
        If we run research directly, the GUI freezes!
        Research takes time (searching web, calling API), and during that time
        the GUI can't respond to clicks or update the screen.

        Solution: Run research in a separate thread (background worker).
        - Main thread: Keeps GUI responsive
        - Background thread: Does the research work

        LEARNING NOTE - Threading safety:
        GUIs aren't thread-safe! You can't update widgets from background threads.
        We use self.after(0, ...) to safely schedule UI updates on the main thread.
        """
        # Update UI to show we're working
        self.is_researching = True
        self.search_button.configure(state="disabled", text="Researching...")
        self.progress_bar.configure(mode="indeterminate")  # Animated progress bar
        self.progress_bar.start()
        self._set_status(f"Researching: {query[:50]}...")

        # Clear previous results
        self.summary_text.delete("1.0", "end")
        self.findings_text.delete("1.0", "end")
        self.extracted_text.delete("1.0", "end")
        self.sources_text.delete("1.0", "end")
        self.usage_text.delete("1.0", "end")

        def run():
            """
            This function runs in the background thread.

            LEARNING NOTE - Why nested function?
            We define this function inside _start_research so it has access
            to query, max_iterations, and model from the outer scope.
            This is called a "closure" in Python.
            """
            try:
                # Build config from settings
                config = Config.from_env()
                config.max_iterations = max_iterations
                config.model = model
                config.validate()

                # Run research (this takes time!)
                agent = ResearchAgent(config)
                result = asyncio.run(agent.research(query))

                # LEARNING NOTE - self.after():
                # Schedule a function to run on the main thread
                # after(0, func) means "run func as soon as possible on main thread"
                # This is how we safely update the GUI from a background thread
                self.after(0, lambda: self._on_research_complete(result))

            except ValueError as e:
                # Invalid config (missing API key, etc.)
                self.after(0, lambda: self._show_error(str(e)))
                self.after(0, self._reset_ui)
            except Exception as e:
                # Unexpected error
                self.after(0, lambda: self._show_error(f"Research failed: {e}"))
                self.after(0, self._reset_ui)

        # LEARNING NOTE - Creating and starting a thread:
        # threading.Thread creates a new thread
        # - target=run: function to run in the thread
        # - daemon=True: thread dies when main program exits
        # thread.start() begins execution
        thread = threading.Thread(target=run, daemon=True)
        thread.start()  # This returns immediately; run() executes in background

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

        # Display key findings (NEW!)
        if result.key_findings:
            findings_text = f"Found {len(result.key_findings)} key findings:\n\n"
            for i, finding in enumerate(result.key_findings, 1):
                findings_text += f"{i}. {finding}\n\n"
        else:
            findings_text = "No key findings extracted.\n\n"
            findings_text += "Key findings are automatically extracted from bullet points\n"
            findings_text += "and numbered lists in the research summary."
        self.findings_text.insert("1.0", findings_text)

        # Display extracted data (NEW!)
        extracted_text = self._format_extracted_data(result)
        self.extracted_text.insert("1.0", extracted_text)

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
  - Key findings: {len(result.key_findings)}
  - Has specs: {'Yes' if result.extracted_data.specifications else 'No'}
  - Has code: {'Yes' if result.extracted_data.code_snippets else 'No'}
  - Has prices: {'Yes' if result.extracted_data.prices else 'No'}
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

    def _format_extracted_data(self, result: ResearchResult) -> str:
        """
        Format extracted data for display in the Extracted Data tab.

        LEARNING NOTE - Structured Data Display:
        This shows all the data we extracted using regex patterns -
        no additional API calls were needed for this extraction!
        """
        data = result.extracted_data
        lines = ["Extracted Data\n==============\n"]

        if data.is_empty():
            lines.append("No structured data was extracted.\n\n")
            lines.append("The data extractor looks for:\n")
            lines.append("  - Technical specifications (RAM: 16GB, CPU: M3)\n")
            lines.append("  - Statistics and percentages (85%, 1.5M users)\n")
            lines.append("  - Prices ($19.99, $100/month)\n")
            lines.append("  - Version numbers (v2.0, Python 3.12)\n")
            lines.append("  - Dates (2024-01-15, March 2024)\n")
            lines.append("  - Code snippets (from markdown code blocks)\n")
            lines.append("  - Key entities (frequently mentioned terms)\n")
            return "\n".join(lines)

        # Technical Specifications
        if data.specifications:
            lines.append("\nTechnical Specifications\n------------------------")
            for key, value in data.specifications.items():
                lines.append(f"  {key}: {value}")

        # Statistics
        if data.statistics:
            lines.append("\n\nStatistics\n----------")
            for stat, context in list(data.statistics.items())[:10]:
                lines.append(f"  {stat}")
                # Show truncated context
                ctx = context[:80] + "..." if len(context) > 80 else context
                lines.append(f"    Context: {ctx}")

        # Prices
        if data.prices:
            lines.append("\n\nPricing\n-------")
            for item, price in data.prices.items():
                lines.append(f"  {item}: {price}")

        # Versions
        if data.versions:
            lines.append("\n\nVersion Numbers\n---------------")
            lines.append(f"  {', '.join(data.versions[:15])}")

        # Dates
        if data.dates:
            lines.append("\n\nDates Mentioned\n---------------")
            lines.append(f"  {', '.join(data.dates[:15])}")

        # Code Snippets
        if data.code_snippets:
            lines.append("\n\nCode Snippets\n-------------")
            for i, snippet in enumerate(data.code_snippets[:5], 1):
                lang = snippet.get("language", "text")
                code = snippet.get("code", "")
                # Truncate long code
                if len(code) > 200:
                    code = code[:200] + "..."
                lines.append(f"\n  [{i}] Language: {lang}")
                lines.append(f"  {code}")

        # Key Entities
        if data.entities:
            lines.append("\n\nKey Entities Mentioned\n----------------------")
            lines.append(f"  {', '.join(data.entities[:15])}")

        # URLs
        if data.urls:
            lines.append(f"\n\nURLs Found: {len(data.urls)}")
            for url in data.urls[:5]:
                lines.append(f"  - {url}")
            if len(data.urls) > 5:
                lines.append(f"  ... and {len(data.urls) - 5} more")

        return "\n".join(lines)

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

    def _save_to_file(self, filepath: str, format_type: str = "txt"):
        """
        Save results to file using the appropriate exporter.

        LEARNING NOTE - Exporter Factory:
        We use get_exporter() to get the right exporter for the format.
        This is the Factory Pattern - we create objects based on input.
        """
        if not self.current_result:
            return

        result = self.current_result

        # Use legacy text format for .txt (backwards compatible)
        if format_type == "txt":
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

Key Findings
------------
"""
            if result.key_findings:
                for i, finding in enumerate(result.key_findings, 1):
                    content += f"{i}. {finding}\n"
            else:
                content += "(No key findings extracted)\n"

            content += "\nSources\n-------\n"
            for i, src in enumerate(result.sources, 1):
                content += f"{i}. {src.get('title', 'Untitled')}\n"
                content += f"   {src.get('url', 'No URL')}\n\n"

            Path(filepath).write_text(content, encoding="utf-8")
        else:
            # Use the appropriate exporter
            try:
                exporter = get_exporter(format_type)
                exporter.export(result, filepath)
            except Exception as e:
                self._show_error(f"Export failed: {e}")
                return

        self._set_status(f"Exported to {Path(filepath).name}")


def main():
    """Launch the GUI application."""
    app = ResearchBotGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
