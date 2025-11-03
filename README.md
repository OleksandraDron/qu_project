# Machine Translation Quality Evaluation for NLI Data

This project focuses on evaluating the quality of machine translations for Natural Language Inference (NLI) data. It uses OpenAI's GPT models for translation and various metrics (BLEU, chrF, and COMET) to assess translation quality.

## Project Overview

The project workflow consists of the following steps:

1. **Translation**: Translate English NLI data to German using GPT-4.1-mini
2. **Backtranslation**: Translate the German text back to English using GPT-4.1
3. **Evaluation**: Assess translation quality using multiple metrics:
   - BLEU and chrF scores (comparing original and backtranslated text)
   - COMET-QE scores (evaluating the quality of translations directly)

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Translation (English to German)

```
python machine_translation.py
```

This script reads NLI data from `data/esnli_selected.xlsx`, translates it to German using GPT-4.1-mini, and saves the results to `esnli_de_translated.xlsx`.

### Backtranslation (German to English)

```
python machine_backtranslation.py
```

This script reads the translated data from `data/esnli_de_gpt4_1_mini.xlsx`, translates it back to English using GPT-4.1, and saves the results to `esnli_en_gpt4_1.xlsx`.

### Evaluation with BLEU and chrF

```
python metrics.py
```

This script compares the original English text with the backtranslated English text using BLEU and chrF scores, providing both overall scores and scores for each field (Sentence1, Sentence2, Explanation_1).

### Evaluation with COMET

```
python mectric_comet.py
```

This script evaluates the quality of the English to German translations using the COMET model, providing scores for each field.

## File Structure

- `machine_translation.py`: Translates English NLI data to German using GPT-4.1-mini
- `machine_backtranslation.py`: Translates German data back to English using GPT-4.1
- `metrics.py`: Evaluates translation quality using BLEU and chrF scores
- `mectric_comet.py`: Evaluates translation quality using COMET-QE
- `requirements.txt`: Lists all dependencies
- `data/`: Directory containing input data files
  - `esnli_selected.xlsx`: Original English NLI data
  - `esnli_de_gpt4_1_mini.xlsx`: Translated German data
  - `esnli_en_gpt4_1.xlsx`: Backtranslated English data

## Dependencies

Key dependencies include:
- `openai`: For accessing GPT models
- `pandas`: For data manipulation
- `sacrebleu`: For BLEU and chrF score calculation
- `unbabel-comet`: For COMET metric calculation
- `python-dotenv`: For environment variable management