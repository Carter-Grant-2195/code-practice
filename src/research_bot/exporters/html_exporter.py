"""
HTML exporter for research results.

LEARNING NOTE - HTML Export:
HTML creates shareable web pages. Great for:
- Sharing results via email
- Opening in any browser
- Professional-looking reports
- Including styling and formatting
"""

from datetime import datetime
from html import escape
from typing import TYPE_CHECKING

from .base import BaseExporter, ExportFormat

if TYPE_CHECKING:
    from ..agents.research_agent import ResearchResult


class HTMLExporter(BaseExporter):
    """
    Export research results as styled HTML.

    Produces a self-contained HTML file with:
    - Clean, modern styling
    - Responsive design
    - Interactive sections
    - Print-friendly layout
    """

    format = ExportFormat.HTML
    extension = "html"

    def __init__(self, dark_mode: bool = False):
        """
        Initialize HTML exporter.

        Args:
            dark_mode: If True, use dark color scheme
        """
        self.dark_mode = dark_mode

    def _format_content(self, result: "ResearchResult") -> str:
        """Convert research result to HTML string."""
        # Color scheme
        if self.dark_mode:
            bg_color = "#1a1a2e"
            text_color = "#eaeaea"
            card_bg = "#16213e"
            accent = "#0f4c75"
            link_color = "#3282b8"
        else:
            bg_color = "#f5f5f5"
            text_color = "#333"
            card_bg = "#fff"
            accent = "#2c3e50"
            link_color = "#3498db"

        status = "Complete" if result.completed else "Partial"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build HTML
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research: {escape(result.query)}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: {text_color};
            background: {bg_color};
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            color: {accent};
            border-bottom: 3px solid {accent};
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: {accent};
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .card {{
            background: {card_bg};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        .meta-item {{
            background: {accent};
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
        }}
        .summary {{
            font-size: 1.1em;
            white-space: pre-wrap;
        }}
        .findings li {{
            margin-bottom: 10px;
            padding-left: 10px;
        }}
        .source {{
            padding: 10px;
            border-left: 3px solid {accent};
            margin-bottom: 10px;
        }}
        .source a {{
            color: {link_color};
            text-decoration: none;
        }}
        .source a:hover {{
            text-decoration: underline;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid {accent}40;
        }}
        th {{
            background: {accent}20;
        }}
        pre {{
            background: {accent}10;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid {accent}40;
            font-size: 0.9em;
            color: {text_color}80;
        }}
        @media print {{
            body {{ background: white; color: black; }}
            .card {{ box-shadow: none; border: 1px solid #ddd; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Research: {escape(result.query)}</h1>

        <div class="meta">
            <span class="meta-item">Status: {status}</span>
            <span class="meta-item">Iterations: {result.iterations}</span>
            <span class="meta-item">Sources: {len(result.sources)}</span>
            <span class="meta-item">Cost: ${result.usage.cost_usd:.4f}</span>
            <span class="meta-item">Model: {result.usage.model}</span>
        </div>

        <div class="card">
            <h2>Summary</h2>
            <div class="summary">{escape(result.summary)}</div>
        </div>
'''

        # Key Findings
        if result.key_findings:
            html += '''
        <div class="card">
            <h2>Key Findings</h2>
            <ul class="findings">
'''
            for finding in result.key_findings:
                html += f'                <li>{escape(finding)}</li>\n'
            html += '''            </ul>
        </div>
'''

        # Sources
        html += '''
        <div class="card">
            <h2>Sources</h2>
'''
        if result.sources:
            for i, source in enumerate(result.sources, 1):
                title = escape(source.get("title", "Untitled"))
                url = escape(source.get("url", "#"))
                html += f'''            <div class="source">
                <strong>{i}.</strong> <a href="{url}" target="_blank">{title}</a>
            </div>
'''
        else:
            html += '            <p><em>No sources were fetched during research.</em></p>\n'
        html += '        </div>\n'

        # Extracted Data
        if not result.extracted_data.is_empty():
            html += self._format_extracted_data(result.extracted_data)

        # Usage Stats
        html += f'''
        <div class="card">
            <h2>API Usage</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Model</td><td>{result.usage.model}</td></tr>
                <tr><td>Input Tokens</td><td>{result.usage.input_tokens:,}</td></tr>
                <tr><td>Output Tokens</td><td>{result.usage.output_tokens:,}</td></tr>
                <tr><td>Total Tokens</td><td>{result.usage.total_tokens:,}</td></tr>
                <tr><td>Cost (USD)</td><td>${result.usage.cost_usd:.4f}</td></tr>
            </table>
        </div>

        <div class="footer">
            <p>Generated by Research Bot on {timestamp}</p>
        </div>
    </div>
</body>
</html>'''

        return html

    def _format_extracted_data(self, data) -> str:
        """Format extracted data as HTML."""
        html = '''
        <div class="card">
            <h2>Extracted Data</h2>
'''

        # Specifications
        if data.specifications:
            html += '''            <h3>Technical Specifications</h3>
            <table>
                <tr><th>Specification</th><th>Value</th></tr>
'''
            for key, value in data.specifications.items():
                html += f'                <tr><td>{escape(key)}</td><td>{escape(value)}</td></tr>\n'
            html += '            </table>\n'

        # Statistics
        if data.statistics:
            html += '            <h3>Statistics</h3>\n            <ul>\n'
            for stat, context in list(data.statistics.items())[:10]:
                html += f'                <li><strong>{escape(stat)}</strong>: {escape(context)}</li>\n'
            html += '            </ul>\n'

        # Prices
        if data.prices:
            html += '''            <h3>Pricing</h3>
            <table>
                <tr><th>Item</th><th>Price</th></tr>
'''
            for item, price in data.prices.items():
                html += f'                <tr><td>{escape(item)}</td><td>{escape(price)}</td></tr>\n'
            html += '            </table>\n'

        # Versions
        if data.versions:
            html += '            <h3>Version Numbers</h3>\n            <p>'
            html += ', '.join(escape(v) for v in data.versions[:10])
            html += '</p>\n'

        # Dates
        if data.dates:
            html += '            <h3>Dates Mentioned</h3>\n            <p>'
            html += ', '.join(escape(d) for d in data.dates[:10])
            html += '</p>\n'

        # Code Snippets
        if data.code_snippets:
            html += '            <h3>Code Snippets</h3>\n'
            for snippet in data.code_snippets[:3]:
                lang = escape(snippet.get("language", ""))
                code = escape(snippet.get("code", ""))
                html += f'            <pre><code class="language-{lang}">{code}</code></pre>\n'

        html += '        </div>\n'
        return html
