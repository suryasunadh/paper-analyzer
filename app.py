import os
import re
from flask import Flask, request, render_template, session
import fitz  # PyMuPDF
import requests

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.secret_key = "7248119045971c255ad845e76bbd9388c53b15103de0cd52ca8c5d51bf3d4141"  # Replace securely for production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

headers = {"Authorization": f"Bearer {API_TOKEN}"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_content(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_key_sections(pdf_text):
    sections = {}
    current_section = None
    collected_lines = []
    target_sections = ['abstract', 'introduction', 'methodology', 'methods', 'conclusion', 'discussion', 'summary']

    for line in pdf_text.split('\n'):
        header = line.strip().lower()
        if header in target_sections:
            if current_section and collected_lines:
                sections[current_section] = '\n'.join(collected_lines).strip()
                collected_lines = []
            current_section = header
        elif current_section:
            collected_lines.append(line)
    if current_section and collected_lines:
        sections[current_section] = '\n'.join(collected_lines).strip()

    combined = ""
    for sec in target_sections:
        if sec in sections:
            combined += sections[sec] + "\n\n"
    return combined.strip()

def call_summarization_api(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    json_resp = response.json()
    if isinstance(json_resp, list) and "summary_text" in json_resp[0]:
        return json_resp[0]["summary_text"]
    elif isinstance(json_resp, dict) and "summary_text" in json_resp:
        return json_resp["summary_text"]
    else:
        return str(json_resp)

def summarize_with_llm(text):
    truncated_text = text[:1000] if len(text) > 1000 else text
    return call_summarization_api(truncated_text)

def gaps_with_llm(pdf_text):
    key_text = extract_key_sections(pdf_text)
    if not key_text:
        key_text = pdf_text[:1000]

    prompt_text = (
        f"Identify the main research gaps and directions for future work in the following research paper:\n{key_text}"
    )
    truncated_prompt = prompt_text[:1000]

    return call_summarization_api(truncated_prompt)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    summary = None
    gaps = None

    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            file.save(file_path)
            text = extract_pdf_content(file_path)
            summary = summarize_with_llm(text)
            gaps = gaps_with_llm(text)

            history = session['history']
            history.append({
                "filename": file.filename,
                "summary": summary,
                "gaps": gaps,
            })
            session['history'] = history

    return render_template('index.html', summary=summary, gaps=gaps, history=session.get('history'))

if __name__ == '__main__':
    app.run(debug=True)