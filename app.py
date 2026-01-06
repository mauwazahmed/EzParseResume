import streamlit as st
import fitz  # PyMuPDF
import pikepdf
import json
import base64
import zlib
import tempfile
import os
from openai import OpenAI
import time

# ---------------- CONFIG ----------------

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
    schema = {
    "basics": {
      "type": "object",
      "required": [
        "name",
        "contact"
      ],
      "properties": {
        "name": {
          "type": "object",
          "properties": {
            "first": {
              "type": "string"
            },
            "middle": {
              "type": "string"
            },
            "last": {
              "type": "string"
            },
            "full": {
              "type": "string"
            }
          }
        },
        "contact": {
          "type": "object",
          "properties": {
            "email": {
              "type": "string",
              "format": "email"
            },
            "phone": {
              "type": "string"
            },
            "alternate_phone": {
              "type": "string"
            }
          }
        },
        "location": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            },
            "state": {
              "type": "string"
            },
            "country": {
              "type": "string"
            }
          }
        },
        "summary": {
          "type": "string"
        },
        "portfolio": {
          "type": "array",
          "items": {
            "type": "string",
            "format": "uri"
          }
        }
      }
    },
    "skills": {
      "type": "object",
      "properties": {
        "technical/tools": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "soft": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "languages": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      }
    },
    "work_experience": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "company",
          "role"
        ],
        "properties": {
          "company": {
            "type": "string"
          },
          "role": {
            "type": "string"
          },
          "employment_type": {
            "type": "string",
            "enum": [
              "intern",
              "fulltime",
              "contract",
              "freelance",
              "part-time"
            ]
          },
          "location": {
            "type": "string"
          },
          "start_date": {
            "type": "string",
            "format": "date"
          },
          "end_date": {
            "type": "string",
            "format": "date"
          },
          "is_current": {
            "type": "boolean"
          },
          "description": {
            "type": "string"
          }
        }
      }
    },
    "education": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "institution",
          "course"
        ],
        "properties": {
          "institution": {
            "type": "string"
          },
          "course": {
            "type": "string"
          },
          "start_date": {
            "type": "string",
            "format": "date"
          },
          "end_date": {
            "type": "string",
            "format": "date"
          },
          "grade": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string"
              },
              "value": {
                "type": "number"
              },
              "scale": {
                "type": "number"
              }
            }
          },
          "main_subjects": {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        }
      }
    },
    "projects": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "title"
        ],
        "properties": {
          "title": {
            "type": "string"
          },
          "description": {
            "type": "string"
          },
          "tools/technologies": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "link": {
            "type": "string",
            "format": "uri"
          }
        }
      }
    },
    "extra_curricular": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "role": {
            "type": "string"
          },
          "organization": {
            "type": "string"
          },
          "description": {
            "type": "string"
          }
        }
      }
    },
    "achievements": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "certifications": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "name"
        ],
        "properties": {
          "name": {
            "type": "string"
          },
          "provider": {
            "type": "string"
          },
          "issue_date": {
            "type": "string",
            "format": "date"
          },
          "expiry_date": {
            "type": "string",
            "format": "date"
          },
          "credential_link": {
            "type": "string",
            "format": "uri"
          }
        }
      }
    },
    "hobbies": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }}
    system_prompt = f"""You are a resume parser. Extract all resume details including headers and corresponding information.
                For dates or years, use start_date and end_date fields. Use clear key-value pairs.
                Do not return any information which is not there in the user text. Return the output in JSON format only.\n\n
                """
    response = client.responses.create(
          model="gpt-4.1-nano",
          input=[{"role":"system","content":system_prompt},{"role":"user","content":raw_text}], 
          text={"format": {"type": "json_schema","name":"iris", "strict": True,"additionalProperties":False,"schema":schema}}
    )
    return json.loads(response.output_text)
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







































