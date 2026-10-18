# LocalInvoiceAI 🧾

LocalInvoiceAI is a locally running AI-powered invoice extraction and management application built with Python and Streamlit.

It helps users extract information from invoices, review the results, validate basic financial consistency, and manage invoice records.

## Features

- Extract invoice information using a local AI model.
- Process PDFs and invoice images using text extraction and OCR.
- Review extracted fields before saving.
- Check basic consistency between subtotals, taxes, and totals.
- Store invoice records in a local SQLite database.
- View invoice records and export data to CSV.

## Technology Stack

- Python
- Streamlit
- Ollama
- Qwen2.5 3B
- Tesseract OCR
- pypdf
- Pillow
- pandas
- SQLite

## Prerequisites

Install the following before running the application:

1. Python 3.11 or a compatible Python version.
2. [Ollama](https://ollama.com/).
3. [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

Pull the model used by the application:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before using AI extraction.

## Installation

Clone the repository:

```bash
git clone https://github.com/Anirban846/LocalInvoiceAI.git
cd LocalInvoiceAI
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure Tesseract

If Tesseract is not detected automatically, configure its executable path in `app.py` to match your installation.

For example, on Windows:

```python
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
```

The path above is an example and may need to be changed on another computer.

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will display the local address where the application is available.

## Privacy

The application is designed to use a local Ollama model and a local SQLite database. Review your configuration and any logging or storage behavior before processing sensitive documents.

Local processing does not automatically guarantee complete privacy or data security.

## Limitations

- AI and OCR can misread invoice fields.
- Financial validation cannot guarantee that an invoice is authentic or correct.
- Users should compare extracted information against the original invoice before saving.
- Tesseract OCR and Ollama must be installed and configured separately.

## License

No open-source license has been selected yet. Until a license is added, do not assume that the repository grants permission to reuse, modify, or redistribute the code.
