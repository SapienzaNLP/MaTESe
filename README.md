<div align="center">

# MaTESe: Machine Translation Evaluation as a Sequence Tagging Problem

![Static Badge](https://img.shields.io/badge/Python%203.9-blue?style=for-the-badge&logo=python&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorch%201.13.1-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/Streamlit%201.22.0-success?style=for-the-badge&logo=streamlit&logoColor=white)


</div>

This repository contains the implementation of the MaTESe metrics, which have been introduced in the paper "MaTESe: Machine Translation Evaluation as a Sequence Tagging Problem" presented at WMT 2022 ([read it here](https://aclanthology.org/2022.wmt-1.51/)).

## About MaTESe

MaTESe metrics tag the error spans of a translation, assigning to them a level of severity that can be either 'Major' or 'Minor'. Additionally, the evaluation produces a numerical quality score, which is derived from combining the penalties linked to each error span.  We have created two metrics: MaTESe and MaTESe-QE. The former requires references to conduct the evaluation, whereas the latter enables a reference-free evaluation.


## How to Use

### Prerequisites

- Python 3.9 or later

### Installation

Clone the repository and install the required dependencies:

```bash
cd MaTESe
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
```

Download the checkpoints of the models and put them in the `checkpoints` directory

MaTESe:
```
https://drive.google.com/file/d/1rcrNFuR6gZfWPCfSm4HJHfjtALBEaXe6/view?usp=sharing
```

MaTESe-QE:
```
https://drive.google.com/file/d/1amAFlHQVpZXqtZWS014RzQ0xpdb6Mymn/view?usp=sharing
```

### Usage

MaTESe can be used in two ways:

1. **Data file evaluation**: To evaluate sentences stored in data files, you need to populate three files: `sources.txt`, `candidates.txt`, and `references.txt`, which are stored in the `data` directory. Each line in `sources.txt`, `candidates.txt`, and `references.txt` must contain respectively a sentence in the source language, its candidate translation, and the corresponding reference translation (`references.txt` is not needed if you are using MaTESe-QE).

   To run the evaluation, use the following command:

    ```bash
    python src/matese/predict.py
    ```

   This command will populate the files `data/output.scores.txt` and `data/output.spans.txt` with the result of the evaluation.

2. **Interactive mode**: If you prefer an interactive mode, you can use MaTESe with Streamlit:

    ```bash
    streamlit run src/demo.py
    ```

   This will start the Streamlit app. You can follow the instructions in the app to evaluate your translations.


## Contact

For any questions or concerns, feel free to open an issue or contact the repository maintainer directly.
