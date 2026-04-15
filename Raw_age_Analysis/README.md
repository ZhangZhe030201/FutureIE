# Raw_age Analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Open `run.py`
2. Modify the following paths:
   - `ANALYSIS_INPUT_DATA_FILE`
   - `MAPPING_FILE`
   - `ANALYSIS_OUTPUT_FILE`
   - `STATS_INPUT_FILE`
3. Run:

```bash
python run.py
```

## Inputs

The analysis input CSV should contain at least:

- `fws_triple`
- `abstract_triple`
- `year`

The mapping CSV should contain at least:

- `entity`
- `year`

## Outputs

Running `python run.py` will generate:

- the analysis CSV file at `ANALYSIS_OUTPUT_FILE`
- the statistical plot `statistical_tests_distribution.png`
- log files under `logs/`
