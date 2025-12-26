# ResearchBot - Distribution Guide

## For Users: How to Use the Standalone Executable

### Quick Start

1. **Download** the `ResearchBot.exe` file
2. **Set your API key** (one-time setup)
3. **Run** the application

### Setting Up Your API Key

The ResearchBot needs an Anthropic API key to work. You have two options:

#### Option 1: Using Environment Variables (Recommended)

**Windows:**
1. Open Command Prompt
2. Set the API key for the current session:
   ```cmd
   set ANTHROPIC_API_KEY=your-api-key-here
   ```
3. Run the app from the same Command Prompt:
   ```cmd
   ResearchBot.exe
   ```

**To make it permanent:**
1. Search for "Environment Variables" in Windows Start Menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables" button
4. Under "User variables", click "New"
5. Variable name: `ANTHROPIC_API_KEY`
6. Variable value: Your API key
7. Click OK and restart any open Command Prompts

#### Option 2: Create a Launcher Script

Create a file named `start-research-bot.bat` next to `ResearchBot.exe`:

```batch
@echo off
set ANTHROPIC_API_KEY=your-api-key-here
start ResearchBot.exe
```

Replace `your-api-key-here` with your actual API key, then double-click `start-research-bot.bat` to launch.

### Getting an API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy it (you won't be able to see it again!)

### Using the Application

Once launched, you'll see a GUI with:
- **Query box**: Enter your research topic
- **Max Iterations slider**: Control research depth (1-15)
- **Model selector**: Choose your Claude model
- **Research button**: Start the research
- **Results tabs**: View Summary, Sources, and API Usage
- **Save button**: Export results to a text file
- **Cost tracking**: Monitor your API spending

### System Requirements

- **OS**: Windows 10 or later
- **RAM**: At least 4GB recommended
- **Internet**: Required for API calls
- **API Key**: Anthropic API key (costs apply)

### Troubleshooting

**"API key not found" error:**
- Make sure ANTHROPIC_API_KEY is set in your environment
- If using the launcher script, check the API key is correct
- Try restarting your computer if you set a permanent environment variable

**Application won't start:**
- Make sure you're running on Windows 10 or later
- Check Windows Defender or antivirus isn't blocking it
- Try running as Administrator

**High API costs:**
- Lower the "Max Iterations" slider
- Use claude-haiku model for cheaper queries
- Monitor the cost display before running queries

### File Information

- **Executable**: `ResearchBot.exe` (~34 MB)
- **No installation required**: Just download and run
- **Portable**: Can be run from any folder or USB drive

---

## For Developers: Building the Executable

If you want to build the executable yourself:

### Prerequisites

```bash
pip install pyinstaller
pip install -e ".[dev]"
```

### Build Command

```bash
python -m PyInstaller --onefile --windowed --name "ResearchBot" --add-data "src/research_bot;research_bot" src/research_bot/gui.py
```

### Output

The executable will be created in `dist/ResearchBot.exe`

### What Gets Included

- Python runtime
- All dependencies (anthropic, beautifulsoup4, customtkinter, httpx)
- Your research bot code
- Tkinter/CustomTkinter assets

### Building for Different Platforms

PyInstaller creates platform-specific executables:
- Build on Windows → `.exe` for Windows
- Build on Mac → `.app` or binary for macOS
- Build on Linux → binary for Linux

You cannot cross-compile, so you need to build on each platform separately.

### Customizing the Build

Edit `ResearchBot.spec` (generated after first build) to:
- Add an icon: `icon='path/to/icon.ico'`
- Include additional files
- Customize compression settings
- Add version information

### Distribution Checklist

- [ ] Build the executable
- [ ] Test on a clean Windows installation
- [ ] Create user documentation
- [ ] Package with sample launcher script
- [ ] Consider code signing (prevents Windows warnings)
- [ ] Zip or create installer for distribution

### File Sharing Options

1. **GitHub Releases**: Upload to GitHub releases page
2. **Cloud Storage**: Google Drive, Dropbox, OneDrive
3. **File Hosting**: WeTransfer, Send, Mega
4. **Your Website**: Self-host the download

### Security Notes

- **Never include your API key in the executable**
- Users must provide their own API keys
- Consider adding a disclaimer about API costs
- Warn users about downloading executables from trusted sources only

---

## API Cost Information

Research queries use the Anthropic API and incur costs:

**Typical Costs (estimates):**
- Simple query (3-5 iterations): $0.01 - $0.05
- Complex query (10+ iterations): $0.10 - $0.50
- Model choice matters:
  - claude-haiku: ~5x cheaper (fastest)
  - claude-sonnet: balanced (recommended)
  - claude-opus: ~3x more expensive (most capable)

**Users are responsible for their own API costs.**

The application displays real-time cost tracking to help users monitor spending.
