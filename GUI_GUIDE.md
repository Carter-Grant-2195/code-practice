# Research Bot GUI Guide

## Overview

The Research Bot comes with a modern desktop GUI built using CustomTkinter. It provides an easy-to-use interface for conducting research without using the command line.

## Running the GUI

### Option 1: Direct execution
```bash
python src/research_bot/gui.py
```

### Option 2: Using Python module syntax
```bash
python -m research_bot.gui
```

### Option 3: If installed with pip
```bash
research-bot-gui
```

## GUI Features

### 1. Research Query Input
- **Location:** Top of the window
- **Usage:** Type your research question and press Enter or click "Research"
- **Example:** "What are the latest developments in quantum computing?"

### 2. Settings Panel
- **Max Iterations:** Control how many research loops the agent can perform
  - Lower (1-3): Faster, less thorough
  - Medium (4-7): Balanced
  - Higher (8-15): Slower, more thorough

- **Model Selection:**
  - `claude-sonnet-4`: Best balance (recommended)
  - `claude-opus-4`: Most powerful, highest cost
  - `claude-haiku-3.5`: Fastest, lowest cost

### 3. Status Bar
- Shows current activity: "Ready", "Researching...", "Complete"
- Progress bar animates during research

### 4. Cost Tracking
- **Last:** Cost of the most recent query
- **Session:** Total cost for all queries since app launched
- **Tokens:** Input and output tokens for the last query
- **Reset Cost Button:** Reset the session counter

### 5. Results Tabs

#### Summary Tab
- Main research findings and answer
- Citations to sources
- Shows if research was completed or partial (max iterations reached)

#### Sources Tab
- List of all web pages fetched during research
- Titles and URLs
- Click to see what sources were consulted

#### Usage Tab
- Detailed API usage statistics
- Token breakdown (input vs output)
- Cost analysis
- Research metadata (iterations, sources count)

### 6. Action Buttons

#### Save Results
- Saves current research to a text file
- Choose location and filename
- Includes query, summary, sources, and usage stats

#### Clear
- Clears all results from display
- Resets UI to ready state

#### Dark Mode Toggle
- Switch between dark and light themes
- Changes immediately, no restart needed

## Keyboard Shortcuts

- **Enter:** Start research (when in query field)
- **Alt+F4:** Close application (Windows)
- **Cmd+Q:** Close application (Mac)

## How It Works

### Threading Explained
When you click "Research", here's what happens:

1. **Main Thread (GUI):**
   - Disables the research button
   - Starts animated progress bar
   - Updates status to "Researching..."

2. **Background Thread:**
   - Creates research agent
   - Performs web searches
   - Calls Claude API
   - Collects results

3. **Back to Main Thread:**
   - Results are safely passed back
   - UI updates with findings
   - Button re-enabled
   - Status updated

**Why this matters:** Without threading, the entire GUI would freeze during research. You couldn't even move the window! Threading keeps it responsive.

### Event-Driven Programming
The GUI uses an event loop:

```
┌──────────────────────────────┐
│  Wait for user action        │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  User clicks button/types    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Call event handler          │
│  (_on_research_click, etc.)  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Update UI                   │
└──────────┬───────────────────┘
           │
           └───► Back to waiting
```

## Common Issues and Solutions

### 1. "API Key Error"
**Problem:** ANTHROPIC_API_KEY not set

**Solution:**
```bash
# Windows
set ANTHROPIC_API_KEY=your-key-here

# Mac/Linux
export ANTHROPIC_API_KEY=your-key-here
```

### 2. GUI Won't Start
**Problem:** CustomTkinter not installed

**Solution:**
```bash
pip install customtkinter
```

### 3. GUI Freezes
**Problem:** Threading issue (shouldn't happen with current code)

**Solution:** Close and restart. If persistent, check error messages.

### 4. Results Not Appearing
**Possible causes:**
- Network issues (can't reach search engines)
- API rate limits
- Invalid search query

**Solution:** Check the error message in the Summary tab

### 5. High API Costs
**Problem:** Using too many iterations or expensive model

**Solutions:**
- Reduce max iterations to 3-5
- Use claude-haiku for cheaper queries
- Use "Reset Cost" button to track spending per session

## GUI Architecture

### Component Hierarchy
```
ResearchBotGUI (Main Window)
├── top_frame (Query Input)
│   ├── query_label
│   ├── query_entry
│   └── search_button
├── settings_frame
│   ├── iterations_slider
│   ├── iterations_value
│   ├── model_label
│   └── model_dropdown
├── status_frame
│   ├── status_label
│   └── progress_bar
├── cost_frame
│   ├── cost_label
│   ├── last_cost_label
│   ├── session_cost_label
│   └── tokens_label
├── tabview (Results)
│   ├── Summary Tab
│   │   └── summary_text
│   ├── Sources Tab
│   │   └── sources_text
│   └── Usage Tab
│       └── usage_text
└── bottom_frame (Actions)
    ├── save_button
    ├── clear_button
    ├── reset_cost_button
    └── theme_switch
```

### State Management
The GUI maintains state through instance variables:
- `self.is_researching`: Prevents multiple simultaneous searches
- `self.current_result`: Stores last research result for saving
- `self.total_cost`: Cumulative session cost

## Customization

### Changing Colors
Edit `gui.py` line 37-38:
```python
ctk.set_appearance_mode("dark")  # or "light" or "system"
ctk.set_default_color_theme("blue")  # or "green" or "dark-blue"
```

### Changing Window Size
Edit `gui.py` line 64-65:
```python
self.geometry("900x700")  # Width x Height
self.minsize(700, 500)    # Minimum size
```

### Changing Default Settings
Edit `gui.py` lines 97-119:
```python
self.iterations_slider.set(5)  # Default iterations
self.model_dropdown.set("claude-sonnet-4-20250514")  # Default model
```

## Learning Exercises

1. **Add a new button:** Try adding a "Copy to Clipboard" button
2. **Add a setting:** Add a text size selector
3. **Add a feature:** Add history of past queries
4. **Theme customization:** Create a custom color theme
5. **Add validation:** Check that API key is set before allowing research

## Comparison: CLI vs GUI

| Feature | CLI | GUI |
|---------|-----|-----|
| **Ease of use** | Requires typing commands | Click and type |
| **Visual feedback** | Text-based | Visual progress bars, tabs |
| **Multitasking** | Blocks terminal | Can minimize/background |
| **Power users** | Faster for experts | Better for beginners |
| **Automation** | Easy to script | Harder to automate |
| **Cost tracking** | Manual calculation | Automatic display |

Both interfaces use the same underlying research engine, so results are identical!

## Next Steps

- Try researching different topics
- Experiment with different iteration counts
- Compare models (cost vs quality)
- Save interesting results
- Check the source code with all the learning comments!

## Resources

- [CustomTkinter Documentation](https://customtkinter.tomschimansky.com/)
- [Python Threading Tutorial](https://docs.python.org/3/library/threading.html)
- [GUI Programming Basics](https://realpython.com/python-gui-tkinter/)

Happy researching! 🔍
