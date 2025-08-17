# -*- coding: utf-8 -*-
"""
Legal Citation Validator

This script analyzes a legal brief provided as a PDF. It performs a three-stage
analysis for each case citation found in the document:

1.  EXTRACT: It uses the `langextract` library with a powerful language model
    (e.g., Gemini 1.5 Flash) to identify and parse case citations and the legal
    propositions they support.

2.  RETRIEVE: For each extracted citation, it queries the CourtListener API
    to retrieve metadata and the full plain text of the opinion. This step
    includes a local file-based cache to avoid redundant API calls.

3.  ANALYZE (Smart Skim Method): For each valid citation, it uses a two-stage
    LLM process for substantive validation:
    a. A cost-effective model (Gemini 1.5 Flash) skims the ENTIRE case text
       to find the most relevant paragraphs related to the proposition.
    b. A more powerful model (Gemini 1.5 Pro) performs the final analysis
       on ONLY the relevant text, ensuring accuracy and managing costs.
    
    This "Smart Skim" can be disabled with the --no-smart-skim flag for a
    more direct, but more expensive, analysis.

The final output is a structured JSON object detailing the findings for each
citation.
"""

import argparse
import json
import os
import re
import textwrap
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
EXTRACTION_MODEL_ID = "gemini-1.5-flash"
ANALYSIS_MODEL_ID = "gemini-1.5-pro"

# Configure the Gemini client for substantive analysis
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Caching Setup for Retrieved Texts ---
CACHE_DIR = Path(".cite_cache")

def ensure_cache_dir_exists():
    """Creates the cache directory if it doesn't already exist."""
    CACHE_DIR.mkdir(exist_ok=True)

def get_cache_filename(citation_query):
    """Creates a filesystem-safe filename from a citation query."""
    safe_name = re.sub(r'[^\w\.\-]', '_', citation_query)
    return f"{safe_name}.json"

def log(message, level="INFO"):
    """Simple logging function to track progress in the terminal."""
    print(f"[{level}] {message}")

# --- Functional Core ---

def parse_pdf(file_path):
    # ... (code is the same as before) ...
    log(f"Parsing PDF: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        log(f"Failed to parse PDF: {e}", level="ERROR")
        return None

def get_langextract_examples():
    # ... (code is the same as before) ...
    return [
        lx.data.ExampleData(
            text="The precedent for judicial review was established in Marbury v. Madison, 5 U.S. 137 (1803).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="The precedent for judicial review was established in Marbury v. Madison, 5 U.S. 137 (1803)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="Marbury v. Madison, 5 U.S. 137 (1803)", attributes={ "case_name": "Marbury v. Madison", "volume": "5", "reporter": "U.S.", "page": "137", "year": "1803", "composite_citation": "5 U.S. 137" })
            ]
        ),
        lx.data.ExampleData(
            text="Jurisdictional questions were central to the proceedings in Wisconsin ex rel. Doe v. Smith, 123 N.W.2d 456 (Wis. 1963).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="Jurisdictional questions were central to the proceedings in Wisconsin ex rel. Doe v. Smith, 123 N.W.2d 456 (Wis. 1963)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="Wisconsin ex rel. Doe v. Smith, 123 N.W.2d 456 (Wis. 1963)", attributes={"case_name": "Wisconsin ex rel. Doe v. Smith", "volume": "123", "reporter": "N.W.2d", "page": "456", "year": "1963", "composite_citation": "123 N.W.2d 456"})
            ]
        ),
        lx.data.ExampleData(
            text="Summary judgment was rejected because genuine issues of material fact remained, as held in Miller v. Jones, 142 Ill. 2d 1, 566 N.E.2d 138 (1991).",
            extractions=[
                lx.data.Extraction(extraction_class="proposition", extraction_text="Summary judgment was rejected because genuine issues of material fact remained, as held in Miller v. Jones, 142 Ill. 2d 1, 566 N.E.2d 138 (1991)."),
                lx.data.Extraction(extraction_class="case_citation", extraction_text="Miller v. Jones, 142 Ill. 2d 1, 566 N.E.2d 138 (1991)", attributes={"case_name": "Miller v. Jones", "volume": "142", "reporter": "Ill. 2d", "page": "1", "year": "1991", "composite_citation": "142 Ill. 2d 1, 566 N.E.2d 138"})
            ]
        )
    ]

def extract_and_link_citations(text):
    # ... (code is the same as before) ...
    log("Starting extraction of citations and propositions...")
    prompt = textwrap.dedent("""
        From the legal text, extract two types of entities:
        1. 'case_citation': The full legal citation.
        2. 'proposition': The full sentence containing the citation.
        For each 'case_citation', extract its components as attributes.
    """)
    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=get_langextract_examples(),
            model_id=EXTRACTION_MODEL_ID,
        )
        propositions = [e for e in result.extractions if e.extraction_class == 'proposition']
        citations = [e for e in result.extractions if e.extraction_class == 'case_citation']
        linked_data = [
            {"proposition": prop.extraction_text, "citation_details": cit.attributes}
            for prop in propositions
            for cit in citations
            if cit.extraction_text in prop.extraction_text
        ]
        log(f"Successfully extracted and linked {len(linked_data)} citations.")
        return linked_data
    except Exception as e:
        log(f"Langextract failed: {e}", level="ERROR")
        return []

def fetch_case_text(citation_query):
    # ... (code is the same as before) ...
    if not citation_query:
        return citation_query, None, "No citation query provided."
    cache_file = CACHE_DIR / get_cache_filename(citation_query)
    if cache_file.exists():
        log(f"  Cache HIT for: {citation_query}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            return citation_query, cached_data.get('data'), cached_data.get('error')
    log(f"  Cache MISS for: {citation_query}. Fetching from API...")
    encoded_citation = urllib.parse.quote(citation_query)
    api_url = f"https://www.courtlistener.com/api/rest/v3/opinions/?q={encoded_citation}"
    headers = {"Authorization": f"Token {COURTLISTENER_API_KEY}"}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        case_data = data["results"][0] if data.get("results") else None
        error_msg = None if case_data else "Citation not found."
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"data": case_data, "error": error_msg}, f)
        return citation_query, case_data, error_msg
    except requests.exceptions.RequestException as e:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"data": None, "error": str(e)}, f)
        return citation_query, None, str(e)

def retrieve_all_case_texts(linked_citations):
    # ... (code is the same as before) ...
    citations_to_fetch = {
        item['citation_details'].get('composite_citation') 
        for item in linked_citations if item.get('citation_details')
    }
    retrieved_data = {}
    log(f"Fetching text for {len(citations_to_fetch)} unique citations from CourtListener...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_citation = {executor.submit(fetch_case_text, cit): cit for cit in citations_to_fetch}
        for i, future in enumerate(as_completed(future_to_citation)):
            citation_query, case_data, error = future.result()
            retrieved_data[citation_query] = {"data": case_data, "error": error}
            log(f"  ({i+1}/{len(citations_to_fetch)}) Processed data for: {citation_query}")
            time.sleep(0.2)
    return retrieved_data

def validate_proposition_with_llm(proposition, case_text, use_smart_skim=True):
    """
    Performs validation, using the Smart Skim method by default.
    If use_smart_skim is False, it sends a large chunk of text directly to the
    powerful analysis model.
    """
    if not case_text or "No text available" in case_text:
        return {"supports": "Uncertain", "explanation": "Case text was not available for analysis.", "quote": None}

    relevant_text = case_text
    analysis_context_text = "the full text of the legal opinion"

    if use_smart_skim:
        # --- Stage 1: The Smart Skim (using a cheaper model) ---
        log("  Performing 'Smart Skim' to find relevant text...")
        skim_model = genai.GenerativeModel(EXTRACTION_MODEL_ID)
        skim_prompt = textwrap.dedent(f"""
            You are a legal assistant. Your task is to find and extract the most relevant paragraphs from the following legal opinion that relate to the proposition below. Do not summarize or analyze, just extract the verbatim text.

            Proposition: "{proposition}"

            Legal Opinion Text:
            ---
            {case_text}
            ---
            Extract the most relevant paragraphs below.
        """)
        try:
            relevant_text = skim_model.generate_content(skim_prompt).text
            analysis_context_text = "the following relevant text from a legal opinion"
        except Exception as e:
            log(f"Smart Skim failed: {e}", level="ERROR")
            relevant_text = "Error during relevance filtering."
    else:
        # --- Direct Analysis Mode ---
        log("  Smart Skim disabled. Sending large context to powerful model...")
        # A large but safe character limit for powerful models like Gemini 1.5 Pro
        relevant_text = case_text[:200000] 

    # --- Stage 2: The Final Analysis (using the powerful model) ---
    log("  Asking powerful LLM for substantive analysis...")
    analysis_model = genai.GenerativeModel(ANALYSIS_MODEL_ID)
    analysis_prompt = textwrap.dedent(f"""
        Please act as a legal analyst. Based ONLY on {analysis_context_text}, determine if it supports the given proposition.

        Proposition From Brief: "{proposition}"

        Relevant Text From Cited Case:
        ---
        {relevant_text}
        ---

        Based on the provided relevant text, does it substantively support the proposition?
        Please respond in a valid JSON object with three keys:
        1. "supports": A string, either "Yes", "No", or "Uncertain".
        2. "explanation": A brief, one-sentence explanation for your answer.
        3. "quote": A direct, relevant quote from the provided text that best supports your explanation.
    """)
    try:
        analysis_response = analysis_model.generate_content(analysis_prompt)
        cleaned_response = re.sub(r'```json\n|\n```', '', analysis_response.text).strip()
        return json.loads(cleaned_response)
    except Exception as e:
        log(f"LLM analysis call failed: {e}", level="ERROR")
        return {"supports": "Error", "explanation": f"LLM analysis failed: {e}", "quote": None}

def build_final_report(linked_citations, retrieved_texts, use_smart_skim):
    """Builds the final JSON report by performing technical and substantive validation."""
    log("Building final report...")
    report = []
    for item in linked_citations:
        citation_details = item.get('citation_details', {})
        composite_citation = citation_details.get('composite_citation')
        retrieved = retrieved_texts.get(composite_citation, {})
        entry = {
            "proposition_from_brief": item.get('proposition'),
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
        if retrieved.get('data'):
            case_data = retrieved['data']
            entry['courtlistener_validation'].update({
                "citation_found": True,
                "retrieved_case_name": case_data.get("case_name"),
                "retrieved_year": case_data.get("date_filed", "")[:4],
                "case_name_match": (
                    citation_details.get('case_name', '').lower() in case_data.get('case_name', '').lower()
                    if citation_details.get('case_name') and case_data.get('case_name') else None
                )
            })
            entry['substantive_analysis'] = validate_proposition_with_llm(
                item.get('proposition'),
                case_data.get('plain_text'),
                use_smart_skim
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
    parser.add_argument(
        "--no-smart-skim",
        action="store_true",
        help="Disable the cost-saving 'Smart Skim' and send a large chunk of text directly to the powerful model."
    )
    args = parser.parse_args()

    use_smart_skim = not args.no_smart_skim

    # --- Execute the Pipeline ---
    ensure_cache_dir_exists()
    
    brief_text = parse_pdf(args.pdf_path)
    if not brief_text:
        return

    linked_citations = extract_and_link_citations(brief_text)
    if not linked_citations:
        log("No citations found to process.", level="WARN")
        return

    retrieved_texts = retrieve_all_case_texts(linked_citations)
    
    final_report = build_final_report(linked_citations, retrieved_texts, use_smart_skim)

    # --- Output Final JSON ---
    print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    main()
