# FutureIE: A Large-Scale Dataset for Fine-Grained Entity and Relation Extraction from Future Work Sentences

This is the official repository for the dataset and code accompanying the paper:

**"FutureIE: A Large-Scale Dataset for Fine-Grained Entity and Relation Extraction from Future Work Sentences"**

FutureIE investigates future work sentences (FWS) in scientific papers and organizes them into a structured resource for frontier-oriented analysis. Based on NLP conference papers from 1952 to 2025, the project combines future work sentence recognition and entity temporal novelty analysis.

---

## 📚 FutureIE

FutureIE is constructed from NLP conference papers collected from ACL-related venues.  
The corpus integrates:

- **73,285 papers** with metadata and full text from [The ACL OCL Corpus](https://github.com/shauryr/ACL-anthology-corpus) (Shaurya Rohatgi, 2022) . covering **1952–2022**
- **31,555 papers** covering **2023–2025** collected from [the ACL Anthology](https://aclanthology.org/) PDFs and parsed into editable text through [DeepSeek OCR](https://github.com/deepseek-ai/DeepSeek-OCR)


On top of this corpus, FutureIE identifies future work sentences through rule-based filtering and a fine-tuned [SciBERT](https://github.com/allenai/scibert) classifier, and then extracts entities and semantic relations using [HGERE](https://github.com/yanzhh/HGERE).

The resulting dataset includes:

- **552,676** future work sentences from **40,773** papers
- **213,768** entities extracted from **34,975** papers
- **38,383** relation triples extracted from **11,875** papers

The repository currently contains two main code modules:

- **FWS_Recognition/**: code for future work sentence recognition
- **Raw_age_analysis/**: code for entity-year matching and temporal novelty analysis

---

## 📦 Repository Structure

```text
FutureIE/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ FWS_Recognition/
│  ├─ README.md
│  ├─ run.py
│  ├─ requirements.txt
│  ├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ logs/
│  └─ reports/
└─ Raw_age_analysis/
   ├─ README.md
   ├─ run.py
   ├─ requirements.txt
   ├─ src/
   ├─ data/
   ├─ logs/
   └─ reports/
```

---

## 🔍 Data Format

The released resources may include processed CSV files and code outputs related to:

- future work sentence recognition
- temporal novelty statistics

### FWS_Recognition output

The prediction results of **FWS_Recognition** are saved in CSV format.  
A typical output file contains the following fields:

- `acl_id`: paper identifier
- `text`: input candidate sentence
- `pred_label`: predicted label
- `prob_0`: predicted probability for class 0
- `prob_1`: predicted probability for class 1


For Raw_age analysis, typical tabular outputs may contain fields such as:

- `acl_id`: paper identifier
- `year`:publication year
- `fws_avg_age`:average Raw_age in future work sentences
- `abs_avg_age`:average Raw_age in abstract
- `sim`:similarity information for fuzzy matching
- `fws_triple`extracted entities from future work sentences
- `abstract_triple`:extracted entities from abstract

Please refer to the README files inside each submodule for the exact input and output format used by that module.

---

## 📈 Main Results

### Future work sentence recognition results

| **Method Category** | **Model** | **Sample Ratio** | **Precision** | **Recall** | **F1** |
|---------------------|-----------|------------------|---------------|------------|--------|
| Traditional ML | SVM | 1.0 | 0.8613 | 0.8492 | 0.8460 |
| Traditional ML | NB | 1.0 | 0.8363 | 0.8359 | 0.8358 |
| Traditional ML | RF | 1.0 | 0.8696 | **0.8624** | 0.8614 |
| Traditional ML | Logistic | 1.0 | 0.8604 | 0.8558 | 0.8551 |
| Pretrained Model | SciBERT | 1.0 | **0.8900** | 0.8464 | **0.8677** |
| Pretrained Model | DeBERTa-v3-large | 1.0 | 0.9966 | 0.9641 | 0.9801 |
| Large Language Model | Qwen3-8B-Instruct | 1.0 | 0.8918 | 0.9967 | 0.9414 |

In the subsequent large-scale recognition stage, the fine-tuned **SciBERT** model was used to identify future work sentences from **3,843,771** candidate sentences, yielding **552,676** future work sentences from **40,773** scientific papers.

### Entity-relation extraction performance

The project adopts **HGERE** for fine-grained entity and relation extraction.  
On the SciNLP test set, the model reports the following F1 scores:

| **Model** | **NER** | **Rel** | **Rel+** |
|-----------|---------|---------|----------|
| HGERE | 79.53 | 49.28 | 47.64 |

### Entity temporal novelty results

To measure entity temporal novelty, the project compares entities extracted from future work sentences with those extracted from abstracts within the same paper.

| **Source** | **Entity Counts** | **Mean Raw_age** | **Median Raw_age** | **SD** |
|------------|-------------------|------------------|--------------------|--------|
| Abstract | 567,540 | 20.06 | 20.00 | 7.08 |
| FWS | 213,768 | 16.97 | 15.00 | 11.32 |

These results indicate that entities in future work sentences tend to appear closer to their first occurrence year and therefore exhibit stronger temporal novelty.


---

## 🚀 Quick Start

### Run future work sentence recognition

```bash
cd FWS_Recognition
python run.py --mode eval --model svm --train_csv data/train1.csv --cv 10 --do_balance
```

### Predict with a saved SciBERT model

```bash
cd FWS_Recognition
python run.py --mode predict --model scibert --model_path models/transformers/deberta_fws/checkpoint-600 --input_csv data/test.csv --out_csv reports/scibert_predictions.csv
```

### Run entity temporal novelty analysis

```bash
cd entity_novelty_analysis
python run.py
```

Before running the novelty analysis module, please edit the input and output paths in `run.py` if necessary.

---

## 📖 References

If you use this repository, please cite the corresponding paper and the major upstream resources used in this project.

```bibtex
@misc{futureie,
  title={FutureIE: A Large-Scale Dataset for Fine-Grained Entity and Relation Extraction from Future Work Sentences},
  author={Anonymous},
  year={2026},
  note={Dataset and code repository}
}

@inproceedings{rohatgi2023aclocl,
  author    = {Rohatgi, Shaurya and Qin, Yujia and Aw, Benjamin and Unnithan, Nitin and Kan, Min-Yen},
  title     = {The ACL OCL Dataset: advancing Open science in Computational Linguistics},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year      = {2023}
}

@inproceedings{duan2025scinlp,
  author    = {Duan, Decheng and Zhang, Yingyi and Peng, Jitong and Zhang, Chengzhi},
  title     = {SciNLP: A Domain-Specific Benchmark for Full-Text Scientific Entity and Relation Extraction in NLP},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025}
}

@inproceedings{Beltagy2019SciBERT,
  title={SciBERT: Pretrained Language Model for Scientific Text},
  author={Iz Beltagy and Kyle Lo and Arman Cohan},
  year={2019},
  booktitle={EMNLP},
  Eprint={arXiv:1903.10676}
}

@misc{yan2023joint,
      title={Joint Entity and Relation Extraction with Span Pruning and Hypergraph Neural Networks}, 
      author={Zhaohui Yan and Songlin Yang and Wei Liu and Kewei Tu},
      year={2023},
      eprint={2310.17238},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

## 📬 Contact

For questions about the code, dataset organization, or reproduction details, please open an issue in this repository.
