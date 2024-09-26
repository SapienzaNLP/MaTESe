<div align="center">

# MaTESe: Machine Translation Evaluation as a Sequence Tagging Problem

![Static Badge](https://img.shields.io/badge/Python%203.9-blue?style=for-the-badge&logo=python&logoColor=white)
![Static Badge](https://img.shields.io/badge/PyTorch%201.13.1-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/Streamlit%201.22.0-success?style=for-the-badge&logo=streamlit&logoColor=white)


</div>

This repository contains the implementation of the MaTESe metrics, introduced in the paper "MaTESe: Machine Translation Evaluation as a Sequence Tagging Problem" presented at WMT 2022 ([read it here](https://aclanthology.org/2022.wmt-1.51/)). 

NOTE: the checkpoints in this repository correspond to the metrics submitted to WMT 2023, with the exception of MaTESe-QE, which was re-trained but not re-submitted. These metrics were trained using [MQM assessments](https://github.com/google/wmt-mqm-human-evaluation) from WMT20 to WMT22.

## About MaTESe

MaTESe metrics tag the error spans of a translation, assigning to them a level of severity that can be either 'Major' or 'Minor'. Additionally, the evaluation produces a numerical quality score, which is derived from combining the penalties linked to each error span.  We have created two metrics: MaTESe and MaTESe-QE. The former requires references to conduct the evaluation, whereas the latter enables a reference-free evaluation.

If you find our paper or code useful, please reference this work in your paper:

```bibtex
@inproceedings{perrella-etal-2022-matese,
    title = "{M}a{TES}e: Machine Translation Evaluation as a Sequence Tagging Problem",
    author = "Perrella, Stefano  and
      Proietti, Lorenzo  and
      Scir{\`e}, Alessandro  and
      Campolungo, Niccol{\`o}  and
      Navigli, Roberto",
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.51",
    pages = "569--577",
}
```


## How to Use

### Prerequisites

- Python 3.9 or later

### Installation

Clone the repository and install the required dependencies:

```bash
cd MaTESe
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
pip install -e ./src
```

Download the checkpoints of the models and put them in the `checkpoints` directory

MaTESe (English only):
```
https://drive.google.com/file/d/12LmxaQP_s42RKORHeII97hJlNyUS2Pgg/view?usp=sharing
```

MaTESe (Supports English, German and Russian as target languages):
```
https://drive.google.com/file/d/1uajyhYYCu3qPfHNIU3RR2NXsvNIyYL4L/view?usp=sharing
```

MaTESe-QE (Supports the language pairs `en-de`, `zh-en` and `en-ru`):
```
https://drive.google.com/file/d/1ZFYTNroMijr9-vYyc1WL0DPnmyrJfwa_/view?usp=sharing
```

### Usage

MaTESe can be used in several ways:

1. **From the command line**: You need to populate two out of the following three files: `data/sources.txt`, `data/candidates.txt`, and `data/references.txt`. Each line of these files must contain respectively a sentence in the source language, its candidate translation, and the corresponding reference translation (`sources.txt` is not needed if you are using MaTESe, `references.txt` is not needed if you are using MaTESe-QE).

   To run the evaluation using MaTESe, use the following command:

    ```bash
    python src/matese.py
    ```
   
   For the English-only version instead:
    
    ```bash
    python src/matese.py --metric matese-en
    ```

   And if you want to use MaTESe-QE:
   
    ```bash
    python src/matese.py --metric matese-qe
    ```

   These commands will create the files `data/output.scores.txt` and `data/output.spans.txt` with the result of the evaluation.

1. **Interactively**: If you prefer an interactive mode, you can use MaTESe with Streamlit:

    ```bash
    streamlit run src/demo.py
    ```

   This will start the Streamlit app. You can follow the instructions in the app to evaluate your translations.

1. **Programmatically**: In the following example you can see how it is possible to use MaTESe metrics in a Python program:

   ```python
   from matese.metric import MaTESe

   candidates = ["This is a wrong translation in English"]
   references = ["This is a sentence in English"]

   metric = MaTESe.load_metric('matese-en') # pass 'matese' or 'matese-qe' for the other versions
   assessments = metric.evaluate(candidates, references=references)

   print(assessments[0])
   ```
   
   ```
   {'spans': [{'offset': (9, 27), 'error': 'Major'}], 'score': -5}
   ```


## License

This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
