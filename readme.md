

# Legal Citation Validator

This tool provides a comprehensive, AI-powered analysis of legal citations within a PDF document. It pushes the frontier of legal tech by moving beyond simple verification to perform a multi-stage analysis:

1.  **Technical Validation**: Checks if a citation exists and if its metadata (case name, year) matches the official record from CourtListener.
2.  **Substantive Validation**: Uses a powerful Large Language Model (LLM) to read the full text of the cited case and determine if it substantively supports the legal proposition for which it was cited in the brief.

## Core Methodology: The "Context-Aware Skim"

To ensure the highest degree of accuracy while managing costs, this tool employs a "Context-Aware Skim" for its substantive analysis. This method is designed to solve the critical problem of context stripping (e.g., failing to distinguish a majority opinion from a dissent).

1.  **Heuristic Sectioning**: The full text of a retrieved case is first pre-processed to identify and label its major sections (Majority, Dissent, Concurrence).
2.  **Context-Aware Skim**: A fast, cost-effective LLM (`gemini-2.5-flash`) reads the entire labeled document to extract only the paragraphs relevant to the proposition, *preserving their section labels*.
3.  **Final Analysis**: The most powerful LLM (`gemini-2.5-pro`) receives only the handful of context-labeled, relevant passages and performs the final "Yes/No/Uncertain" judgment, with explicit instructions on how to weigh majority vs. dissenting opinions.

This approach allows the tool to analyze the full context of a legal opinion in a way that is both legally sound and economically viable.

## Requirements

### 1\. Python and Libraries

  - Python 3.8+
  - Required Python packages:
    ```bash
    pip install langextract google-generativeai requests PyPDF2 python-dotenv
    ```

### 2\. API Keys

  - **CourtListener API Key**: From the [Free Law Project](https://free.law/2024/04/16/citation-lookup-api/).
  - **Google Gemini API Key**: From [Google AI Studio](https://aistudio.google.com/app/apikey).

## Setup

1.  **Save the Script**: Save the `cite_checker.py` script to a new directory.
2.  **Install Dependencies**: Open your terminal, navigate to the directory, and run the `pip install` command listed above.
3.  **Create `.env` File**: In the same directory, create a file named `.env` and add your API keys:
    ```
    COURTLISTENER_API_KEY="your_courtlistener_api_key_here"
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```

## How to Run

Run the script from your terminal, providing the path to your legal brief PDF as an argument.

### Analysis Modes: Cost vs. Quality

#### Default Mode: Context-Aware Skim (Recommended)

This is the default mode, offering the best balance of cost, speed, and accuracy.

```bash
python cite_checker.py path/to/your/brief.pdf
```

#### High-Fidelity Mode: `--no-smart-skim`

For cases where the absolute highest analytical quality is required and cost is not a concern, you can disable the skim. This will send a very large portion of the case text directly to the most powerful model.

```bash
python cite_checker.py path/to/your/brief.pdf --no-smart-skim
```

## Output Format

The script outputs a JSON array, where each object represents one analyzed citation. The structure for each object is as follows:

```json
{
  "proposition_from_brief": "The legal assertion made in the brief.",
  "citation_from_brief": "The citation string as it appeared.",
  "langextract_analysis": {
    "case_name": "Case Name",
    "composite_citation": "123 U.S. 456"
  },
  "courtlistener_validation": {
    "citation_found": true,
    "retrieved_case_name": "Official Case Name",
    "case_name_match": true
  },
  "substantive_analysis": {
    "supports": "No",
    "explanation": "The proposition is directly contradicted by the holding in the Majority opinion.",
    "quote": "A direct quote from the case that supports the explanation."
  }
}
```
