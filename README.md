# MaTESe: Machine Translation Evaluation as a Sequence Tagging Problem

Welcome to the MaTESe GitHub repository!

## About MaTESe

MaTESe metrics have been introduced in the paper "MaTESe: Machine Translation Evaluation as a Sequence Tagging Problem" presented at WMT 2022 ([read it here](https://aclanthology.org/2022.wmt-1.51/)). 

MaTESe metrics tag the spans of a translation that contain an error, assigning it a level of severity that can be either 'Major' or 'Minor'. Additionally, the evaluation produces a numerical quality score, which is derived from combining the penalties linked to each erroneous span.  We've created two metrics: MaTESe and MaTESe-QE. The latter requires references to conduct the evaluation, whereas the former enables a reference-free evaluation.

Note: Differently from what is said in the paper, the current version of MaTESe takes as input both the source and the reference, together with the candidate translation.
## How to Use

### Prerequisites

- Python 3.9 or later

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sapienzanlp/MaTESe.git
cd MaTESe
pip install -r requirements.txt
```

Download the checkpoint of the models and put them in the `checkpoints` directory

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

1. **Data file evaluation**: To evaluate sentences stored in data files, you need to populate three files: `sources.txt`, `candidates.txt`, and `references.txt`, that are stored in the `data` directory. Each line in `sources.txt`, `candidates.txt` and `references.txt` must contain respectively a sentence in the source language, its candidate translation and the corresponding reference translation (`references.txt` is not needed if you are using MaTESe-QE).
    
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