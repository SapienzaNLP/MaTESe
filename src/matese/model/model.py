from typing import List, Dict, Callable, Tuple

import torch
import transformers
from transformers import AutoModel, PreTrainedTokenizer
from pytorch_lightning import LightningModule

from matese.model import model_utils


class MateseModule(LightningModule):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        reference_less: bool,
        transformer_model_name: str,
        ordered_labels: List[str],
        encoder_layers: int,
        encoder_attention_heads: int = 8,
        dropout_rate: float = 0.1,
        use_last_n_layers: int = 4,
    ):

        super().__init__()

        self.transformer_model_name = transformer_model_name
        self.use_last_n_layers = use_last_n_layers
        self.dropout_rate = dropout_rate
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layers = encoder_layers
        self.tokenizer = tokenizer
        self.reference_less = reference_less
        self.ordered_labels = ordered_labels

        self.transformer_model = AutoModel.from_pretrained(
            self.transformer_model_name,
            output_hidden_states=True,
            return_dict=True,
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.transformer_model.config.hidden_size,
            nhead=self.encoder_attention_heads,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.encoder_layers,
        )

        self.transformer_model.resize_token_embeddings(len(self.tokenizer))

        linear_size = self.encoder.layers[0].linear2.out_features

        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(linear_size, 5, bias=False),
        )

    def postprocess_predictions(
            self,
            predictions: torch.tensor,
            candidate_indices: List[List[int]],
            token_to_chars: List[Callable]
    ) -> Tuple[List[List[str]], List[List[Dict]]]:

        predictions = [
            prediction[: len(candidate_indices[idx])]
            for idx, prediction in enumerate(predictions)
        ]

        string_predictions: List[List[str]] = [
            [self.ordered_labels[int_prediction] for int_prediction in prediction]
            for idx, prediction in enumerate(predictions)
        ]

        sparse_predictions: List[List[Dict]] = [
            model_utils.sparsify_prediction(string_prediction)
            for string_prediction in string_predictions
        ]
        sparse_char_based_predictions: List[List[Dict]] = [
            model_utils.bpe2char(sparse_prediction, token_to_chars[idx])
            for idx, sparse_prediction in enumerate(sparse_predictions)
        ]

        return string_predictions, sparse_char_based_predictions

    def forward(
            self,
            input_ids,
            attention_mask: torch.tensor,
            candidate_indices: List[List[int]],
    ) -> Dict[str, torch.tensor]:

        input_ids = input_ids.to(self.transformer_model.device)
        attention_mask = attention_mask.to(self.transformer_model.device)

        encoder_output = self.transformer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.use_last_n_layers > 1:
            encoded_bpes = torch.stack(
                encoder_output.hidden_states[-self.use_last_n_layers :],
                dim=-1,
            ).sum(-1)
        else:
            encoded_bpes = encoder_output.last_hidden_state

        src_key_padding_mask = attention_mask != 1
        encoded_bpes = self.encoder(
            src=encoded_bpes, src_key_padding_mask=src_key_padding_mask
        )
        candidates_bpes = [
            torch.stack(
                [
                    encoded_bpes[idx][candidate_idx]
                    for candidate_idx in candidate_indices[idx]
                ]
            )
            for idx in range(len(candidate_indices))
        ]

        candidates_bpes = torch.nn.utils.rnn.pad_sequence(
            candidates_bpes, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        logits = self.classification_head(candidates_bpes)

        predictions = logits.argmax(dim=-1)
        probabilities = logits.softmax(dim=-1)
        return {
            "logits": logits,
            "predictions": predictions,
            "probabilities": probabilities,
        }

    def predict(
            self, encoded_input: transformers.tokenization_utils.BatchEncoding
    ) -> Dict:
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        candidate_indices = [encoded_input["candidate_indices"]]
        token_to_chars = [encoded_input["token_to_chars"]]

        predictions = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_indices=candidate_indices,
        )["predictions"]

        string_predictions, sparse_predictions = self.postprocess_predictions(
            predictions, candidate_indices, token_to_chars
        )

        return {
            "prediction": predictions[0],
            "string_prediction": string_predictions[0],
            "sparse_prediction": sparse_predictions[0],
        }

    def batch_predict(self, batch: List) -> Dict:

        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        candidate_indices = [item["candidate_indices"] for item in batch]
        token_to_chars = [item["token_to_chars"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_masks,
            candidate_indices=candidate_indices,
        )

        predictions = output["predictions"]

        (
            string_predictions,
            sparse_char_based_predictions,
        ) = self.postprocess_predictions(predictions, candidate_indices, token_to_chars)

        probabilities = output["probabilities"]
        probabilities = [
            probability[: len(candidate_indices[idx])]
            for idx, probability in enumerate(probabilities)
        ]

        return {
            "predictions": predictions,
            "string_predictions": string_predictions,
            "sparse_predictions": sparse_char_based_predictions,
            "probabilities": probabilities,
        }
