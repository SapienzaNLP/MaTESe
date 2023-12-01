from typing import *
import logging
import json
from tqdm import tqdm

import torch
import transformers.tokenization_utils
from transformers import AutoConfig
from pytorch_lightning import LightningModule

from matese.data import data_utils
from matese.model.model import MateseModule
from matese.utils import utils


class MaTESe:

    def __init__(
            self,
            model: LightningModule,
            minimum_score: Optional[int] = -25,
    ):
        super(MaTESe, self).__init__()

        self.model = model

        self.reference_less = self.model.reference_less
        self.tokenizer = self.model.tokenizer
        self.minimum_score = minimum_score

    @classmethod
    def load_metric(
            cls,
            metric_name: str,
            device: str = "cuda"
    ):
        device = torch.device(device)

        if metric_name == "matese":
            checkpoint = torch.load(
                utils.get_root_dir().joinpath('checkpoints', 'matese.ckpt'),
                map_location='cpu'
            )
            with open(utils.get_root_dir().joinpath('configs', 'matese.json'), "r") as f:
                config = json.load(f)
        elif metric_name == "matese-en":
            checkpoint = torch.load(
                utils.get_root_dir().joinpath('checkpoints', 'matese-en.ckpt'),
                map_location='cpu'
            )
            with open(utils.get_root_dir().joinpath('configs', 'matese-en.json'), "r") as f:
                config = json.load(f)
        elif metric_name == "matese-qe":
            checkpoint = torch.load(
                utils.get_root_dir().joinpath('checkpoints', 'matese-qe.ckpt'),
                map_location='cpu'
            )
            with open(utils.get_root_dir().joinpath('configs', 'matese-qe.json'), "r") as f:
                config = json.load(f)
        else:
            print("Supported metrics: ['matese', 'matese-en', 'matese-qe']")
            exit()

        tokenizer = utils.get_tokenizer(config['transformer_model_name'])
        ordered_labels = list(checkpoint["hyper_parameters"]["ordered_labels"])
        module = MateseModule(**config, tokenizer=tokenizer, ordered_labels=ordered_labels)
        module.load_state_dict(checkpoint["state_dict"])
        module.to(device)

        metric = MaTESe(module)

        return metric

    @classmethod
    def batch_data(cls, data, batch_size: int = 32) -> List[List[transformers.tokenization_utils.BatchEncoding]]:
        batches = []
        for idx in range(0, len(data), batch_size):
            batches.append(data[idx:idx + batch_size])
        return batches

    def evaluate(
            self,
            candidates: List[str],
            sources: Optional[List[str]] = None,
            references: Optional[List[str]] = None,
            batch_size: int = 32,
    ) -> List[Dict]:
        """

        Args:
            candidates: a list of candidate translations
            sources: a list of source sentences
            references: a list of reference translations
            batch_size: the batch size to use for the evaluation

        Returns: a list of dictionaries of the form
            {
                "spans": List[Tuple[int, int]],
                "score": int
            }

        """

        if candidates is None:
            raise ValueError("You have not provided any candidate translations!")
        if self.reference_less:
            if sources is None:
                raise ValueError("MaTESe-QE requires the source sentences for the evaluation!")
        else:
            if references is None:
                raise ValueError("MaTESe requires the reference translations for the evaluation!")

        data = data_utils.preprocessing_pipeline(
            self.tokenizer,
            candidates,
            self.reference_less,
            sources,
            references,
        )

        batches = self.batch_data(data, batch_size)
        with torch.no_grad():
            output = []
            for batch in tqdm(batches, total=len(batches)):
                output += [self.model.batch_predict(batch)]

        predictions: List[List[Dict]] = [
            sparse_pred for batch_output in output for sparse_pred in batch_output['sparse_predictions']
        ]
        scores = self.compute_scores_from_spans(predictions)

        return [{
            "spans": predictions[idx],
            "score": scores[idx]
        } for idx in range(len(predictions))]

    def compute_scores_from_spans(self, predictions: List[List[Dict]]) -> List[int]:
        scores = []
        for prediction in predictions:
            score = 0
            for span_prediction in prediction:
                if span_prediction['error'] == data_utils.MQMSeverities.major:
                    score -= 5
                elif span_prediction['error'] == data_utils.MQMSeverities.minor:
                    score -= 1
                else:
                    logging.error("The prediction is not among the allowed classes\n"
                                  f"Span prediction: {span_prediction}\n"
                                  f"Prediction: {prediction}")
                    exit()

            if self.minimum_score is not None:
                score = score if score > self.minimum_score else self.minimum_score
            scores.append(score)

        return scores
