import markdown2
import pdfkit


html_template = """
    <html>
    <head>
        <style>
            body {{
                font-family: 'Helvetica', sans-serif;
                font-size: 18px;
                line-height: 1.4;
                margin: 30px;
                color: #333;
            }}
            h1 {{
                font-size: 30px;
                font-weight: bold;
                margin-bottom: 12px;
                page-break-before: always;  /* Ensure headings don’t get split */
            }}
            h2, h3, h4 {{
                page-break-inside: avoid;  /* Avoid breaking headers */
            }}
            p, ul, li {{
                font-size: 18px;
                margin-bottom: 5px;
                page-break-inside: avoid;  /* Prevent splitting lists and paragraphs */
            }}
            ul {{
                margin-left: 20px;
                padding-left: 20px;
                margin-bottom: 5px;
            }}
            li {{
                margin-bottom: 3px;
            }}
            strong {{
                font-size: 18px;
                font-weight: bold;
            }}
            pre {{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
                line-height: 1.4;
                overflow-x: auto;
                white-space: pre-wrap;
                color: #333;
                page-break-inside: avoid;  /* Prevent code blocks from splitting */
            }}
            code {{
                font-family: 'Courier New', monospace;
                font-size: 15px;
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        {}
    </body>
    </html>
"""


def write(markdown: str, output_path: str):
    html_text = markdown2.markdown(markdown, extras=["fenced-code-blocks", "wrap-code"])

    styled_html = html_template.format(html_text)

    print(f"\nWriting PDF and Markdown...")
    pdfkit.from_string(styled_html, output_path)

    output_md_path = output_path.replace(".pdf", ".md")
    with open(output_md_path, 'w') as file:
        file.write(markdown)