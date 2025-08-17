Legal Citation Validator
This tool provides a comprehensive analysis of legal citations within a PDF document. It goes beyond simple verification by performing a two-level check:

Technical Validation: Checks if a citation exists and if its metadata (case name, year) matches the official record from CourtListener.

Substantive Validation: Uses a powerful Large Language Model (LLM) to read the full text of the cited case and determine if it substantively supports the legal proposition for which it was cited in the brief.

Features
PDF Parsing: Directly ingests and processes text from PDF files.

Advanced Extraction: Uses langextract and Google's Gemini model to accurately identify case citations and the legal propositions they support.

Official API Integration: Retrieves case data and full opinion text from the CourtListener API.

Concurrent Processing: Makes efficient, concurrent API calls to quickly process documents with many citations.

AI-Powered Analysis: Leverages Google's Gemini 1.5 Pro to perform nuanced, substantive analysis of legal arguments.

Structured JSON Output: Produces a detailed JSON report, perfect for review or integration into other applications.

Requirements
1. Python and Libraries
Python 3.8+

The following Python packages are required:

Bash

pip install langextract google-generativeai requests PyPDF2 python-dotenv
2. API Keys
You will need two API keys:

CourtListener API Key: From the Free Law Project.

Google Gemini API Key: From Google AI Studio.

Setup
Clone or Download: Save the cite_checker.py script to a new directory.

Install Dependencies: Open your terminal, navigate to the directory, and run the pip install command listed above.

Create .env File: In the same directory, create a file named .env. Add your API keys to this file as follows:

COURTLISTENER_API_KEY="your_courtlistener_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
How to Run
Run the script from your terminal, providing the path to your legal brief PDF as an argument.

Bash

python cite_checker.py path/to/your/brief.pdf
Example
Download a sample brief and save it as sample_brief.pdf.

Run the command:

Bash

python cite_checker.py sample_brief.pdf
The script will print progress to the console and output the final, detailed JSON report.

Output Format
The script outputs a JSON array, where each object represents one analyzed citation. The structure for each object is as follows:

JSON

{
  "proposition_from_brief": "The legal assertion made in the brief.",
  "citation_from_brief": "The citation string as it appeared.",
  "langextract_analysis": {
    "case_name": "Case Name",
    "volume": "123",
    "reporter": "U.S.",
    "page": "456",
    "year": "2024",
    "composite_citation": "123 U.S. 456"
  },
  "courtlistener_validation": {
    "citation_found": true,
    "api_error": null,
    "retrieved_case_name": "Official Case Name",
    "retrieved_year": "2024",
    "case_name_match": true
  },
  "substantive_analysis": {
    "supports": "Yes",
    "explanation": "The LLM's one-sentence summary of its findings.",
    "quote": "A direct quote from the case that supports the explanation."
  }
}
