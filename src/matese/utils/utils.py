from typing import Optional
from pathlib import Path

from transformers import PreTrainedTokenizer, AutoTokenizer, AddedToken


from matese.data.data_utils import (
    SOURCE_SPECIAL_TOKEN,
    CANDIDATE_SPECIAL_TOKEN,
    REFERENCE_SPECIAL_TOKEN,
    EMPTY_STRING_SPECIAL_TOKEN,
)


def get_root_dir():
    return Path(__file__).parent.parent.parent.parent


def get_tokenizer(transformer_model_name: str, model_max_length: int = 512) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

    tokenizer.model_max_length = model_max_length

    source_special_token = AddedToken(
        SOURCE_SPECIAL_TOKEN,
        lstrip=True,
        rstrip=False,
    )
    candidate_special_token = AddedToken(
        CANDIDATE_SPECIAL_TOKEN,
        lstrip=True,
        rstrip=False,
    )
    reference_special_token = AddedToken(
        REFERENCE_SPECIAL_TOKEN,
        lstrip=True,
        rstrip=False,
    )
    empty_string_special_token = AddedToken(
        EMPTY_STRING_SPECIAL_TOKEN,
        lstrip=True,
        rstrip=False,
    )

    special_tokens = {
        "additional_special_tokens": [
            source_special_token,
            candidate_special_token,
            reference_special_token,
            empty_string_special_token,
        ]
    }

    tokenizer.add_special_tokens(special_tokens)

    return tokenizer
