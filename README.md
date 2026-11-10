
# Machine Translation Pipeline (English → Hindi)

## Setup and Installation

Follow these steps to run the project on **VS Code**.

### 1. Get Gemini API Key

- Visit [Google AI Studio](https://aistudio.google.com/).
- Generate a free Gemini API key.
- Keep the API key ready for the translation stage.

### 2. Extract the Project

Unzip the project folder and navigate to the project directory:

```powershell
cd project
```

### 3. Create and Activate Virtual Environment

Create a Python virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

**If script execution is blocked**, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate the environment again.

### 4. Install Dependencies

Install PyTorch with CUDA 12.4 support for NVIDIA RTX 4060:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

Download the required spaCy English language model:

```powershell
python -m spacy download en_core_web_sm
```

### 5. Run the Translation Pipeline

Execute the following scripts in the specified order.

**Step 1: Data Preparation**
```powershell
python prep.py
```

**Step 2: Model Training**
```powershell
python train.py
```

**Step 3: Configure Gemini API Key**

Set your Gemini API key as an environment variable:

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

**Step 4: Run Translation**

```powershell
python translate.py 50
```

Here, `50` represents the number of sentences to translate.

For a quick test, use:

```powershell
python translate.py 20
```

**Step 5: Compare Translation Results**

```powershell
python compare.py
```

### 6. Generate the Project Report

Ensure that the following files and folders are arranged together:

```text
project/
│
├── prep.py
├── train.py
├── translate.py
├── compare.py
├── report.tex
├── requirements.txt
└── results/
```

## Important Notes

### Gemini API Usage
- `translate.py 50` processes 50 sentences.
- Each sentence is evaluated using 3 different prompts.
- This results in approximately **150 Gemini API calls**.
- A 4-second delay is introduced between calls to help stay within free-tier rate limits.
- Expected execution time: approximately **10–15 minutes**, depending on API quota and response times.

### MarianMT Model Download
- The first execution of `translate.py` downloads the MarianMT translation model.
- The model requires a few hundred MB of disk space.
- Subsequent runs can reuse the downloaded model.

### Report Customization
- The institute name is obtained from the report template header.
- Replace the red placeholder team details with your actual information.
- Tables and charts are automatically generated using data from the `results/` directory.
- Replace the red `[ ... ]` placeholders with your own analysis and commentary.
- Include observations about plots, translation quality, and examples of translation errors.

---

## Hardware Requirements

| Component | Configuration |
|---|---|
| Operating System | Windows |
| IDE | Visual Studio Code |
| Python | Python 3 |
| GPU | NVIDIA RTX 4060 |
| CUDA | 12.4 |
| Translation API | Google Gemini |
| Translation Model | MarianMT |


