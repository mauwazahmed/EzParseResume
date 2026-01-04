import streamlit as st
import json
import base64
import zlib
import tempfile
import os
from dotenv import load_dotenv

import pikepdf
import fitz  # PyMuPDF
from openai import OpenAI

load_dotenv()
# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# Encode / Decode metadata
# -----------------------------
def encode_metadata(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, 9)
    return base64.b64encode(compressed).decode("utf-8")


def decode_metadata(blob: str) -> dict:
    decoded = base64.b64decode(blob)
    decompressed = zlib.decompress(decoded)
    return json.loads(decompressed)


# -----------------------------
# Extract raw text from PDF
# -----------------------------
def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


# -----------------------------
# Parse resume using OpenAI
# -----------------------------
def parse_resume_with_openai(raw_text: str) -> dict:
    prompt = f"""
You are a resume parser.
Extract all resume details and return ONLY valid JSON.
For dates or years, use start_date and end_date fields
Use clear key-value pairs.
Do not include explanations.

Resume Text:
\"\"\"
{raw_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You output only JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)


# -----------------------------
# Embed metadata into PDF
# -----------------------------
def embed_metadata(input_pdf, payload, output_pdf):
    with pikepdf.open(input_pdf) as pdf:
        with pdf.open_metadata(set_pikepdf_as_editor=False) as xmp:
            xmp["resume:payload"] = payload

        pdf.save(output_pdf)


# -----------------------------
# Read metadata from PDF
# -----------------------------
def read_metadata(pdf_path):
    with pikepdf.open(pdf_path) as pdf:
        xmp = pdf.open_metadata()
        if "resume:payload" in xmp:
            return decode_metadata(xmp["resume:payload"])
    return None


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Resume Metadata Editor", layout="wide")
st.title("📄 Resume Parser & Metadata Embedder (OpenAI)")

tabs = st.tabs(["Upload & Edit Resume", "Inspect PDF Metadata"])

# ======================================================
# TAB 1 — Upload, Parse, Edit, Embed
# ======================================================
with tabs[0]:
    uploaded_pdf = st.file_uploader("Upload Resume PDF", type="pdf")

    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        st.success("PDF uploaded")

        if st.button("🔍 Parse Resume using OpenAI"):
            with st.spinner("Extracting & parsing resume..."):
                raw_text = extract_pdf_text(tmp_path)
                print(raw_text)
                parsed_json = parse_resume_with_openai(raw_text)

            st.session_state["parsed_json"] = parsed_json

        if "parsed_json" in st.session_state:
            edited_json = st.text_area(
                "Edit Parsed Resume JSON",
                value=json.dumps(st.session_state["parsed_json"], indent=2),
                height=400
            )

            if st.button("💾 Save & Embed Metadata"):
                final_json = json.loads(edited_json)
                payload = encode_metadata(final_json)

                output_pdf = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ).name

                embed_metadata(tmp_path, payload, output_pdf)

                st.success("Metadata embedded successfully")

                with open(output_pdf, "rb") as f:
                    st.download_button(
                        "⬇️ Download Final PDF",
                        f,
                        file_name="resume_with_metadata.pdf"
                    )

        os.unlink(tmp_path)

# ======================================================
# TAB 2 — Inspect Metadata
# ======================================================
with tabs[1]:
    inspect_pdf = st.file_uploader(
        "Upload PDF to Inspect Metadata",
        type="pdf",
        key="inspect"
    )

    if inspect_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(inspect_pdf.read())
            inspect_path = tmp.name

        metadata = read_metadata(inspect_path)

        if metadata:
            st.success("Metadata found!")
            st.json(metadata)
        else:
            st.warning("No resume metadata found in this PDF")

        os.unlink(inspect_path)

