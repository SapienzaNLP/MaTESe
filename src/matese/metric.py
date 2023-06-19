from typing import *
import logging
import json

import torch
import transformers.tokenization_utils
from pytorch_lightning import LightningModule

from matese.data import data_utils
from matese.model.model import MateseModule
from matese.utils import utils


class MaTESe:

    def __init__(
            self,
            model: LightningModule,
            batch_size: int = 32,
            minimum_score: Optional[int] = -25,
    ):
        super(MaTESe, self).__init__()

        self.model = model
        assert(model.eval())

        self.reference_less = self.model.reference_less
        self.tokenizer = self.model.tokenizer
        self.minimum_score = minimum_score
        self.batch_size = batch_size

    @classmethod
    def load_metric(cls, metric_name: str, device: str = "cuda"):
        device = torch.device(device)

        if metric_name == "matese":
            checkpoint = torch.load(
                utils.get_root_dir().joinpath('checkpoints', 'matese.ckpt'),
                map_location=device
            )
            with open(utils.get_root_dir().joinpath('configs', 'matese.json'), "r") as f:
                config = json.load(f)
        elif metric_name == "matese-qe":
            checkpoint = torch.load(
                utils.get_root_dir().joinpath('checkpoints', 'matese-qe.ckpt'),
                map_location=device
            )
            with open(utils.get_root_dir().joinpath('configs', 'matese-qe.json'), "r") as f:
                config = json.load(f)
        else:
            print("Supported metrics: ['matese', 'matese-qe']")
            exit()

        tokenizer = utils.get_tokenizer(config['transformer_model_name'])
        ordered_labels = list(checkpoint["hyper_parameters"]["ordered_labels"])
        module = MateseModule(**config, tokenizer=tokenizer, ordered_labels=ordered_labels)
        module.load_state_dict(checkpoint["state_dict"])

        metric = MaTESe(module)

        return metric

    def batch_data(self, data) -> List[List[transformers.tokenization_utils.BatchEncoding]]:
        batches = []
        for idx in range(0, len(data), self.batch_size):
            batches.append(data[idx:idx + self.batch_size])
        return batches

    def evaluate(
            self,
            candidates: List[str],
            sources: Optional[List[str]] = None,
            references: Optional[List[str]] = None,
    ) -> List[Dict]:
        """

        Args:
            candidates: a list of candidate translations
            sources: a list of source sentences
            references: a list of reference translations

        Returns: a list of dictionaries of the form
            {
                "spans": List[Tuple[int, int]],
                "score": int
            }

        """

        if self.reference_less:
            assert sources
        else:
            assert(sources and references)

        data = data_utils.preprocessing_pipeline(
            self.tokenizer,
            candidates,
            self.reference_less,
            sources,
            references,
        )

        batches = self.batch_data(data)
        with torch.no_grad():
            output = [self.model.batch_predict(batch) for batch in batches]

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

            # setting the minimum score to -25 as it is in mqm
            if self.minimum_score is not None:
                score = score if score > self.minimum_score else self.minimum_score
            scores.append(score)

        return scores
