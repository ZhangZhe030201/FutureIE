<<<<<<< HEAD
# FWS Recognition

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run commands from the project root.

### 1. Train sklearn model
```bash
python run.py --mode train --model svm --train_csv data/train1.csv --model_path models/fws_svm.pkl --do_balance
```

### 2. Evaluate sklearn model
```bash
python run.py --mode eval --model svm --train_csv data/train1.csv --cv 10 --do_balance
```

### 3. Evaluate a saved model
```bash
python run.py --mode eval_saved --model logistic-regression --model_path models/fws_logistic.pkl --eval_csv data/train1.5.csv
```

### 4. Predict with a saved sklearn model
```bash
python run.py --mode predict --model svm --model_path models/fws_svm.pkl --input_csv data/test.csv --out_csv reports/predictions.csv
```

### 5. Train transformer model
```bash
python run.py --mode train --model scibert --hf_ckpt microsoft/deberta-v3-large --train_csv data/train1.csv --model_path models/deberta_fws --do_balance
```

### 6. Predict with a saved transformer model
```bash
python run.py --mode predict --model scibert --model_path path/to/checkpoint --input_csv data/test.csv --out_csv reports/scibert_predictions.csv --hf_ckpt microsoft/deberta-v3-large
```

## Inputs

The labeled training or evaluation CSV should contain:

- `text`
- `label`

The prediction CSV should contain:

- `text`

## Outputs

Running the commands above will generate model files, prediction CSV files, and evaluation outputs under `models/`, `reports/`, and `logs/` if enabled.
=======
# FutureIE
>>>>>>> 45e8d4c83c2b71368598ae05f10a118a71129762
