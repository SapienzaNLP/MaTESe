from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

SOURCE_SPECIAL_TOKEN = "<source>"
CANDIDATE_SPECIAL_TOKEN = "<candidate>"
REFERENCE_SPECIAL_TOKEN = "<reference>"
EMPTY_STRING_SPECIAL_TOKEN = "<empty-string>"


@dataclass
class MQMSeverities:
    major: str = "Major"
    minor: str = "Minor"
    neutral: str = "Neutral"

    # no-error severity is a placeholder for when the error was a no-error
    no_error: str = "No-error"


def preprocessing_pipeline(
    tokenizer: PreTrainedTokenizer,
    candidates: List[str],
    reference_less: bool,
    sources: Optional[List[str]] = None,
    references: Optional[List[str]] = None,
) -> List[BatchEncoding]:

    assert sources or references, "Either sources or references must be not None!"

    data = [
        {
            "candidate": candidates[idx],
            "source": sources[idx] if sources is not None else None,
            "reference": references[idx] if references is not None else None,
        }
        for idx in range(len(candidates))
    ]

    data = tokenize(tokenizer, data, reference_less=reference_less)

    return data


def materialize_labels(labels: List[Dict], candidate_indices: List[int]) -> Dict:
    """
    Transforms sparse labels (those in the form {'offset': offset, 'span_label': span_label}) into dense labels,
    namely strings with a label for each position

        returns: a dictionary where original labels are those that still contain the macro category, micro category
            and severity, while "labels" are those that only contain the pre-decided "error" field
    """

    dense_labels = ["O" for _ in range(len(candidate_indices))]
    dense_original_labels = ["O" for _ in range(len(candidate_indices))]

    first_candidate_idx = candidate_indices[0]

    for label in labels:
        start_t, end_t = label["offset"]
        # Depending on where is the candidate, we need to shift the indices of the labels to start from 0
        start_t, end_t = start_t - first_candidate_idx, end_t - first_candidate_idx
        error = label["error"]

        dense_labels[start_t] = "B-" + error
        dense_original_labels[start_t] = (
            "B-" + label["category"] + "-" + label["severity"]
        )
        for t_idx in range(start_t + 1, end_t):
            dense_labels[t_idx] = "I-" + error
            dense_original_labels[t_idx] = (
                "I-" + label["category"] + "-" + label["severity"]
            )

    return {"label": dense_labels, "original_label": dense_original_labels}


def tokenize(
    tokenizer: PreTrainedTokenizer,
    data: List[Dict],
    reference_less: bool,
) -> List[BatchEncoding]:
    """
    Returns: the tokenized data along a list of indices to determine which samples have been filtered out
        and which have been kept

    """

    init_data, data_indices = [], []
    (
        source_special_token_id,
        candidate_special_token_id,
        reference_special_token_id,
    ) = tokenizer.convert_tokens_to_ids(
        [SOURCE_SPECIAL_TOKEN, CANDIDATE_SPECIAL_TOKEN, REFERENCE_SPECIAL_TOKEN]
    )

    logging.debug("Source special token id: " + str(source_special_token_id))
    logging.debug("Candidate special token id: " + str(candidate_special_token_id))
    logging.debug("Reference special token id: " + str(reference_special_token_id))

    num_truncated_samples = 0
    for idx, sample in enumerate(data):
        candidate = sample["candidate"]
        source = sample["source"]
        reference = sample["reference"]

        if candidate is None or candidate == "":
            candidate = EMPTY_STRING_SPECIAL_TOKEN
            print("Warning: the provided candidate translation is empty!")
        if reference is None or reference == "":
            reference = EMPTY_STRING_SPECIAL_TOKEN
        if source is None or source == "":
            source = EMPTY_STRING_SPECIAL_TOKEN

        if reference_less:
            input_string = (
                    CANDIDATE_SPECIAL_TOKEN
                    + " "
                    + candidate
                    + " "
                    + SOURCE_SPECIAL_TOKEN
                    + " "
                    + source
            )
        else:
            input_string = (
                    CANDIDATE_SPECIAL_TOKEN
                    + " "
                    + candidate
                    + " "
                    + REFERENCE_SPECIAL_TOKEN
                    + " "
                    + reference
            )

        encoded_input = tokenizer(input_string, return_tensors="pt", truncation=True)

        encoded_input["input_ids"] = encoded_input["input_ids"][0]
        encoded_input["attention_mask"] = encoded_input["attention_mask"][0]
        encoded_input["source"] = source
        encoded_input["candidate"] = candidate
        encoded_input["reference"] = reference

        candidate_start_idx, reference_start_idx, source_start_idx = None, None, None
        try:
            candidate_start_idx = list(encoded_input.input_ids).index(
                candidate_special_token_id
            )
            if reference_less:
                source_start_idx = list(encoded_input.input_ids).index(
                    source_special_token_id
                )
            else:
                reference_start_idx = list(encoded_input.input_ids).index(
                    reference_special_token_id
                )

        except ValueError as e:
            logging.debug(e)
            logging.debug("This sample has been truncated due to being too long!")
            logging.debug("Candidate: ", candidate)
            logging.debug("Source: ", source)
            logging.debug("Reference: ", reference)
            logging.debug("Tokens: ", encoded_input.tokens())
            num_truncated_samples += 1

        if reference_less:
            encoded_input["reference_indices"] = []
            if source_start_idx is not None:
                encoded_input["candidate_indices"] = list(
                    range(candidate_start_idx + 1, source_start_idx)
                )
                encoded_input["source_indices"] = list(
                    range(source_start_idx + 1, len(encoded_input.tokens()))
                )
            else:
                encoded_input["candidate_indices"] = list(
                    range(candidate_start_idx + 1, len(encoded_input.tokens()))
                )
                encoded_input["source_indices"] = []
        else:
            encoded_input["source_indices"] = []
            if reference_start_idx is not None:
                encoded_input["candidate_indices"] = list(
                    range(candidate_start_idx + 1, reference_start_idx)
                )
                encoded_input["reference_indices"] = list(
                    range(reference_start_idx + 1, len(encoded_input.tokens()))
                )
            else:
                encoded_input["candidate_indices"] = list(
                    range(candidate_start_idx + 1, len(encoded_input.tokens()))
                )

        encoded_input["tokens"] = encoded_input.tokens()
        encoded_input["candidate_special_token"] = CANDIDATE_SPECIAL_TOKEN
        encoded_input["reference_special_token"] = REFERENCE_SPECIAL_TOKEN
        encoded_input["source_special_token"] = SOURCE_SPECIAL_TOKEN
        encoded_input["empty_string_special_token"] = EMPTY_STRING_SPECIAL_TOKEN
        encoded_input["token_to_chars"] = encoded_input.token_to_chars

        init_data.append(encoded_input)

    if num_truncated_samples > 0:
        logging.warning(
            f"Truncation: {num_truncated_samples} samples have been severely truncated (up to losing completely one "
            f"of the sentences that compose the input)"
        )

    return init_data


def char2bpe(char_labels: List[Dict], encoded_input: BatchEncoding) -> List[Dict]:
    """

    Args:
        char_labels: labels with offsets based on characters (of the candidate) positions
        encoded_input: a function mapping from characters positions to tokens positions

    Returns: a token-based list of labels, namely labels in which offsets are based on token positions
    """

    token_labels = []
    for label in char_labels:
        start_c, end_c = label["offset"]
        category, severity = label["category"], label["severity"]
        error = label["error"]

        token_label = {}
        for char_idx in range(start_c, end_c):
            token_idx = encoded_input.char_to_token(char_idx)
            if token_idx is None:
                continue

            if token_label.get("offset", None) is None:
                token_label["offset"] = (token_idx, token_idx + 1)
                token_label["category"] = category
                token_label["severity"] = severity
                token_label["error"] = error
            else:
                token_label["offset"] = (token_label["offset"][0], token_idx + 1)
        token_labels.append(token_label)

    return token_labels
