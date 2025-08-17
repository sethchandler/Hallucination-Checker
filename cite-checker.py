# -*- coding: utf-8 -*-
"""
Legal Citation Validator

This script analyzes a legal brief provided as a PDF. It performs a three-stage
analysis for each case citation found in the document:

1.  EXTRACT: It uses the `langextract` library with a powerful language model
    (e.g., Gemini 1.5 Flash) to identify and parse case citations and the legal
    propositions they support.

2.  RETRIEVE: For each extracted citation, it queries the CourtListener API
    to retrieve metadata (case name, year) and the full plain text of the
    opinion. This step performs a technical validation of the citation.

3.  ANALYZE: For each technically valid citation, it uses an advanced language
    model (e.g., Gemini 1.5 Pro) to perform a substantive validation. The model
    reads the full text of the case to determine if it supports the proposition
    made in the original brief.

The final output is a structured JSON object detailing the findings for each
citation, suitable for review or as an input for a GUI application.
"""

import argparse
import json
import os
import re
import textwrap
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
import PyPDF2
import requests
from dotenv import load_dotenv

import langextract as lx

# --- Configuration ---
load_dotenv()

# API Keys and Model Configuration from .env file
COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EXTRACTION_MODEL_ID = "gemini-1.5-flash"  # Good for fast, accurate extraction
ANALYSIS_MODEL_ID = "gemini-1.5-pro"      # Better for in-depth analysis

# Configure the Gemini client for substantive analysis
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def log(message, level="INFO"):
    """Simple logging function to track progress in the terminal."""
    print(f"[{level}] {message}")

# --- Functional Core ---

def parse_pdf(file_path):
    """
    Parses a PDF file and returns its text content.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF, or None if an error occurs.
    """
    log(f"Parsing PDF: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        log(f"Failed to parse PDF: {e}", level="ERROR")
        return None

def get_langextract_examples():
    """
    Provides a diverse set of examples to guide the langextract model.
    These examples cover various jurisdictions, reporters, case name styles,
    and the use of 'id.' to improve extraction accuracy.

    Returns:
        list: A list of lx.data.ExampleData objects.
    """
    return [
        # Base example
        lx.data.ExampleData(
            text="The precedent for judicial review was established in Marbury v. Madison, 5 U.S. 137 (1803).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="The precedent for judicial review was established in Marbury v. Madison, 5 U.S. 137 (1803)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="Marbury v. Madison, 5 U.S. 137 (1803)", attributes={ "case_name": "Marbury v. Madison", "volume": "5", "reporter": "U.S.", "page": "137", "year": "1803", "composite_citation": "5 U.S. 137" })
            ]
        ),
        # Example with 'id.'
        lx.data.ExampleData(
            text="The court in United States v. Nixon, 418 U.S. 683, 703 (1974) held that executive privilege is not absolute. The ruling at id. at 704 further clarified the scope.",
            extractions=[
                 lx.data.Extraction(extraction_class="proposition", extraction_text="The court in United States v. Nixon, 418 U.S. 683, 703 (1974) held that executive privilege is not absolute."),
                 lx.data.Extraction(extraction_class="case_citation", extraction_text="United States v. Nixon, 418 U.S. 683, 703 (1974)", attributes={ "case_name": "United States v. Nixon", "volume": "418", "reporter": "U.S.", "page": "683", "pin_cite": "703", "year": "1974", "composite_citation": "418 U.S. 683, 703" }),
                 lx.data.Extraction(extraction_class="proposition", extraction_text="The ruling at id. at 704 further clarified the scope."),
                 lx.data.Extraction(extraction_class="case_citation", extraction_text="id. at 704", attributes={ "case_name": "United States v. Nixon", "volume": "418", "reporter": "U.S.", "page": "683", "pin_cite": "704", "year": "1974", "composite_citation": "418 U.S. 683, 704" })
            ]
        ),
        # Example with 'In re'
        lx.data.ExampleData(
            text="The automatic stay is a key feature of bankruptcy law, as seen in In re Johns-Manville Corp., 837 F.2d 89 (2d Cir. 1988).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="The automatic stay is a key feature of bankruptcy law, as seen in In re Johns-Manville Corp., 837 F.2d 89 (2d Cir. 1988)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="In re Johns-Manville Corp., 837 F.2d 89 (2d Cir. 1988)", attributes={ "case_name": "In re Johns-Manville Corp.", "volume": "837", "reporter": "F.2d", "page": "89", "year": "1988", "composite_citation": "837 F.2d 89" })
            ]
        ),
        # Example with 'ex rel.' and a jurisdictional issue
        lx.data.ExampleData(
            text="Jurisdictional questions were central to the proceedings in Wisconsin ex rel. Doe v. Smith, 123 N.W.2d 456 (Wis. 1963).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="Jurisdictional questions were central to the proceedings in Wisconsin ex rel. Doe v. Smith, 123 N.W.2d 456 (Wis. 1963)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="Wisconsin ex rel. Doe v. Smith, 123 N.W.2d 456 (Wis. 1963)", attributes={"case_name": "Wisconsin ex rel. Doe v. Smith", "volume": "123", "reporter": "N.W.2d", "page": "456", "year": "1963", "composite_citation": "123 N.W.2d 456"})
            ]
        ),
        # Example with an obscure reporter and rule interpretation
        lx.data.ExampleData(
            text="The court's interpretation of Rule 23 was pivotal in Johnson v. State, 45 P.3d 890 (Okla. Crim. App. 2002).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="The court's interpretation of Rule 23 was pivotal in Johnson v. State, 45 P.3d 890 (Okla. Crim. App. 2002)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="Johnson v. State, 45 P.3d 890 (Okla. Crim. App. 2002)", attributes={"case_name": "Johnson v. State", "volume": "45", "reporter": "P.3d", "page": "890", "year": "2002", "composite_citation": "45 P.3d 890"})
            ]
        ),
        # Example with parallel citations and summary judgment
        lx.data.ExampleData(
            text="Summary judgment was rejected because genuine issues of material fact remained, as held in Miller v. Jones, 142 Ill. 2d 1, 566 N.E.2d 138 (1991).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="Summary judgment was rejected because genuine issues of material fact remained, as held in Miller v. Jones, 142 Ill. 2d 1, 566 N.E.2d 138 (1991)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="Miller v. Jones, 142 Ill. 2d 1, 566 N.E.2d 138 (1991)", attributes={"case_name": "Miller v. Jones", "volume": "142", "reporter": "Ill. 2d", "page": "1", "year": "1991", "composite_citation": "142 Ill. 2d 1, 566 N.E.2d 138"})
            ]
        )
    ]

def extract_and_link_citations(text):
    """
    Uses langextract to extract citations and propositions, then links them.

    This function runs the primary extraction task. It identifies propositions
    (as the full sentence containing a citation) and the citation itself. It
    then links them together based on string containment.

    Args:
        text (str): The text of the legal brief.

    Returns:
        list: A list of dictionaries, where each dictionary contains a
              'proposition' and its associated 'citation_details'.
    """
    log("Starting extraction of citations and propositions...")
    prompt = textwrap.dedent("""
        From the legal text, extract two types of entities:
        1. 'case_citation': The full legal citation, including volume, reporter, page, and year.
        2. 'proposition': The complete sentence that contains the case citation, representing the legal argument being made.
        For each 'case_citation', extract its components (case_name, volume, reporter, page, pin_cite, year, composite_citation) as attributes.
    """)
    
    try:
        # The main call to the langextract library
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=get_langextract_examples(),
            model_id=EXTRACTION_MODEL_ID,
        )

        propositions = [e for e in result.extractions if e.extraction_class == 'proposition']
        citations = [e for e in result.extractions if e.extraction_class == 'case_citation']

        # Link propositions to the citations they contain
        linked_data = []
        for prop in propositions:
            for cit in citations:
                if cit.extraction_text in prop.extraction_text:
                    linked_data.append({
                        "proposition": prop.extraction_text,
                        "citation_details": cit.attributes
                    })
                    # For simplicity, we assume one primary citation per proposition.
                    # This could be expanded to handle multiple citations in one sentence.
                    break
        
        log(f"Successfully extracted and linked {len(linked_data)} citations.")
        return linked_data
    except Exception as e:
        log(f"Langextract failed: {e}", level="ERROR")
        return []

def fetch_case_text(citation_query):
    """
    Fetches full case text and metadata from CourtListener for a single citation.

    Args:
        citation_query (str): The citation string (e.g., "5 U.S. 137").

    Returns:
        tuple: A tuple containing (citation_query, data, error_message).
               'data' is the JSON response from the API, or None on failure.
    """
    if not citation_query:
        return citation_query, None, "No citation query provided."

    encoded_citation = urllib.parse.quote(citation_query)
    api_url = f"https://www.courtlistener.com/api/rest/v3/opinions/?q={encoded_citation}"
    headers = {"Authorization": f"Token {COURTLISTENER_API_KEY}"}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            case_data = data["results"][0]
            return citation_query, case_data, None
        else:
            return citation_query, None, "Citation not found."
    except requests.exceptions.RequestException as e:
        return citation_query, None, str(e)

def retrieve_all_case_texts(linked_citations):
    """
    Retrieves all case texts from CourtListener concurrently.

    Uses a thread pool to send multiple API requests at once, improving
    performance. Includes a respectful pause between requests.

    Args:
        linked_citations (list): The list of linked citations from langextract.

    Returns:
        dict: A dictionary mapping composite_citation strings to their
              retrieved data or an error message.
    """
    citations_to_fetch = [
        item['citation_details'].get('composite_citation') 
        for item in linked_citations
    ]
    retrieved_data = {}

    log(f"Fetching text for {len(citations_to_fetch)} citations from CourtListener...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_citation = {
            executor.submit(fetch_case_text, cit): cit for cit in citations_to_fetch if cit
        }

        total_citations = len(future_to_citation)
        for i, future in enumerate(as_completed(future_to_citation)):
            citation_query, case_data, error = future.result()
            retrieved_data[citation_query] = {"data": case_data, "error": error}
            log(f"  ({i+1}/{total_citations}) Retrieved data for: {citation_query}")
            # Respectful pause to avoid overwhelming the API
            time.sleep(0.2)

    return retrieved_data

def validate_proposition_with_llm(proposition, case_text):
    """
    Uses a powerful LLM to validate if a case text substantively supports a proposition.

    Args:
        proposition (str): The legal assertion from the brief.
        case_text (str): The full text of the cited case.

    Returns:
        dict: A dictionary containing the LLM's analysis.
    """
    if not case_text or "No text available" in case_text:
        return {"supports": "Uncertain", "explanation": "Case text was not available for analysis.", "quote": None}

    log(f"  Asking LLM for substantive analysis of proposition: \"{proposition[:50]}...\"")
    model = genai.GenerativeModel(ANALYSIS_MODEL_ID)
    
    prompt = textwrap.dedent(f"""
        Please act as a legal analyst. Analyze the provided legal opinion to determine if it supports the given proposition.

        Proposition From Brief: "{proposition}"

        Full Text of Cited Case (first 30,000 characters):
        ---
        {case_text[:30000]}
        ---

        Based on the text of the opinion, does it substantively support the proposition?
        Please respond in a valid JSON object with three keys:
        1. "supports": A string, either "Yes", "No", or "Uncertain".
        2. "explanation": A brief, one-sentence explanation for your answer.
        3. "quote": A direct, relevant quote from the opinion that best supports your explanation.
    """)

    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON before parsing
        cleaned_response = re.sub(r'```json\n|\n```', '', response.text).strip()
        return json.loads(cleaned_response)
    except Exception as e:
        log(f"LLM validation call failed: {e}", level="ERROR")
        return {"supports": "Error", "explanation": f"LLM analysis failed: {e}", "quote": None}

def build_final_report(linked_citations, retrieved_texts):
    """
    Builds the final JSON report by performing technical and substantive validation.

    Args:
        linked_citations (list): The langextract output.
        retrieved_texts (dict): The data retrieved from CourtListener.

    Returns:
        list: A list of dictionaries, one for each analyzed citation.
    """
    log("Building final report...")
    report = []

    for item in linked_citations:
        citation_details = item['citation_details']
        composite_citation = citation_details.get('composite_citation')
        retrieved = retrieved_texts.get(composite_citation, {})
        
        # Initialize the report entry for this citation
        entry = {
            "proposition_from_brief": item['proposition'],
            "citation_from_brief": composite_citation,
            "langextract_analysis": citation_details,
            "courtlistener_validation": {
                "citation_found": False,
                "api_error": retrieved.get('error'),
                "retrieved_case_name": None,
                "retrieved_year": None,
                "case_name_match": None
            },
            "substantive_analysis": {}
        }

        # If we successfully retrieved data, perform technical and substantive validation
        if retrieved.get('data'):
            case_data = retrieved['data']
            entry['courtlistener_validation']['citation_found'] = True
            entry['courtlistener_validation']['retrieved_case_name'] = case_data.get("case_name")
            entry['courtlistener_validation']['retrieved_year'] = case_data.get("date_filed", "")[:4]

            # Perform technical validation (case name match)
            if citation_details.get('case_name') and case_data.get('case_name'):
                entry['courtlistener_validation']['case_name_match'] = (
                    citation_details['case_name'].lower() in case_data['case_name'].lower()
                )

            # Perform substantive validation using the retrieved text
            entry['substantive_analysis'] = validate_proposition_with_llm(
                item['proposition'],
                case_data.get('plain_text')
            )
        
        report.append(entry)

    log("Report generation complete.")
    return report

# --- Main Application Logic ---

def main():
    """Main function to orchestrate the citation checking process."""
    if not COURTLISTENER_API_KEY or not GEMINI_API_KEY:
        log("API keys for CourtListener and Gemini must be set in a .env file.", level="ERROR")
        return

    parser = argparse.ArgumentParser(
        description="A tool to technically and substantively validate legal citations in a PDF brief.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pdf_path", type=str, help="The path to the legal brief PDF file.")
    args = parser.parse_args()

    # --- Execute the Pipeline ---
    brief_text = parse_pdf(args.pdf_path)
    if not brief_text:
        return

    linked_citations = extract_and_link_citations(brief_text)
    if not linked_citations:
        log("No citations were found to process.", level="WARN")
        return

    retrieved_texts = retrieve_all_case_texts(linked_citations)
    
    final_report = build_final_report(linked_citations, retrieved_texts)

    # --- Output Final JSON ---
    print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    main()


