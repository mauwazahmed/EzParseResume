import streamlit as st
import fitz  # PyMuPDF
import pikepdf
import json
import base64
import zlib
import tempfile
import os
from openai import OpenAI

# ---------------- CONFIG ----------------

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# ---------------- HELPERS ----------------

def extract_visible_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def extract_xmp_metadata(pdf_path):
    with pikepdf.open(pdf_path) as pdf:
        xmp = pdf.open_metadata()
        payload = xmp.get("resume:payload")
        if not payload:
            return None

        decoded = zlib.decompress(base64.b64decode(payload)).decode()
        return json.loads(decoded)

def embed_xmp_metadata(pdf_path, metadata_json):
    encoded = base64.b64encode(
        zlib.compress(json.dumps(metadata_json).encode())
    ).decode()

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        with pdf.open_metadata() as meta:
            meta["resume:version"] = "1"
            meta["resume:format"] = "json+zlib+base64"
            meta["resume:payload"] = encoded
        pdf.save(pdf_path)


def parse_resume_with_openai(raw_text: str) -> dict:
    ASSISTANT_ID = st.secrets["ASSISTANT_ID"]
    prompt = f"""{raw_text}"""
    
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=prompt)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    reply = messages.data[0].content[0].text.value
    print(reply)

    return json.loads(reply)
# ---------------- UI ----------------

st.set_page_config(page_title="IRIS",page_icon="📄",layout="centered")
st.title("IRIS — Intelligent Resume Interchange Standard")
tab1, tab2 = st.tabs(["Candidate", "Recruiter / ATS Docs"])

# ======================================================
# TAB 1 — CANDIDATE
# ======================================================

with tab1:
    st.subheader("Welcome to IRIS!")
    st.markdown(
    """
    **Make your resume work smarter.**

    See how your resume looks after parsing, add richer information without clutter,
    and ensure it is both **human-friendly** and **machine-readable**.

    Stop worrying about incomplete or erroneous parsing and
    **simplify your entire resume workflow** — from upload to optimization.
    """
)
    uploaded = st.file_uploader("Upload PDF resume", type=["pdf"], help="Upload your resume in PDF format only")
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            pdf_path = tmp.name

        text = extract_visible_text(pdf_path)
        metadata = extract_xmp_metadata(pdf_path)
        if metadata is not None:
            st.success("✅ Parsed metadata detected in resume")

        else:
            st.warning("❌ The resume does not have parsed metadata.")
            metadata = {}

            if st.button("🔍 Parse Resume with AI"):
                with st.spinner("Parsing resume..."):
                    metadata = parse_resume_with_openai(text)

        st.subheader("📦 Resume Metadata (Editable)")
        metadata_str = st.text_area(
            "Edit metadata JSON",
            json.dumps(metadata, indent=4),
            height=400
        )

        if st.button("💾 Save & Embed Metadata"):
            try:
                final_metadata = json.loads(metadata_str)
                embed_xmp_metadata(pdf_path, final_metadata)
                st.success("Metadata embedded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇ Download Updated Resume",
                f,
                file_name=uploaded.name,
                mime="application/pdf"
            )
        st.divider()
        st.subheader("Feedback")
        st.markdown(
        """
        <div style="text-align: center; margin-top: 30px;">
            <a href="https://forms.gle/L5RQULW5y4TEraG39" target="_blank">
                <button style="
                    background-color:#4CAF50;
                    color:white;
                    padding:10px 20px;
                    border:none;
                    border-radius:6px;
                    font-size:16px;
                    cursor:pointer;">
                    💬 Share Feedback
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
        )

# ======================================================
# TAB 2 — RECRUITER / ATS DOCS
# ======================================================

with tab2:
    doc_text= """
How Recruiter ATS Can Use IRIS
------------------------------

Overview
----
This resume may contain machine-readable structured metadata embedded
inside the PDF using XMP metadata.

The purpose of this metadata is to:
- Reduce resume parsing errors
- Eliminate repeated parsing
- Improve candidate data accuracy
- Remain fully backward-compatible with existing ATS systems


Recommended ATS Pipeline
-----------------------
    Resume Upload
        ↓
    Check for XMP Metadata
        ↓
    IF FOUND:
        - Extract XMP metadata
        - Decode structured JSON payload
        - Skip re-parsing
    ELSE:
        - Use existing resume parser


    
Metadata Storage Format
----------------------
Storage Method:
- XMP metadata embedded inside the PDF
- Namespace-based keys
- Versioned payload

Encoding Method:
- JSON
- zlib compression
- Base64 encoding



Why This Works?
--------------
- No breaking changes to ingestion pipelines
- Metadata extraction is cheap and deterministic
- Eliminates layout-based parsing failures
- Resume remains human-readable simultaneously becoming machine-readable
- Helps candidates add/edit information in metadata without cluttering the visual resume
- Saves cost on the AI parsing side for recruiters



"""
    st.markdown(doc_text)
    st.subheader("Python Code to Extract & Decode Metadata")
    st.code('''
import pikepdf
import base64
import zlib
import json

with pikepdf.open("resume.pdf") as pdf:
    meta = pdf.open_metadata()
    encoded = meta.get("resume:payload")

    if encoded:
        decoded = zlib.decompress(base64.b64decode(encoded))
        data = json.loads(decoded)
        print(data)''')
    st.subheader("XMP Metadata")
    st.code('''
<rdf:Description
    xmlns:resume="https://example.com/ai-resume"
    resume:version="1"
    resume:format="json+zlib+base64"
    resume:payload="ENCODED_JSON_PAYLOAD"
/>''')
    st.subheader("Decoded JSON Metadata Example")
    st.code('''
{
  "name": "Candidate Name",
  "email": "candidate@email.com",
  "phone": "+91-XXXXXXXXXX",
  "skills": [
    "Python",
    "SQL",
    "Machine Learning"
  ],
  "experience": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "duration": "2022 - 2024"
    }
  ],
  "education": [
    {
      "degree": "Bachelor of Technology",
      "institution": "University Name"
    }
  ]
}
''')














