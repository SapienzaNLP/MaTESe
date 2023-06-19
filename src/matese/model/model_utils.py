from typing import List, Dict, Callable
import logging
import sys

from matese.data.data_utils import CANDIDATE_SPECIAL_TOKEN


def bpe2char(bpe_predictions: List[Dict], token_to_chars: Callable) -> List[Dict]:
    """
    Args:
        bpe_predictions: predictions with offsets based on bpes (of the candidate) positions
        token_to_chars: a function mapping from bpe positions to char positions

    Returns: a char-based prediction, namely predictions in which offsets are based on char positions

    We need to take into account that the first char considered by token_to_chars is the candidate special token. For
    this reason, I need to shift the offsets by its length (+1, to account for the space after the special token)
    """

    assert(token_to_chars(0) is None)

    char_predictions = []
    for idx, bpe_prediction in enumerate(bpe_predictions):
        start_t, end_t = bpe_prediction['offset']

        char_prediction = {}
        for token_idx in range(start_t, end_t):
            if token_to_chars(token_idx) is not None:
                start_c, end_c = token_to_chars(token_idx)
                start_c, end_c = start_c - len(CANDIDATE_SPECIAL_TOKEN) - 1, end_c - len(CANDIDATE_SPECIAL_TOKEN) - 1
                if char_prediction.get('offset', None) is None:
                    char_prediction['offset'] = (start_c, end_c)
                    char_prediction['error'] = bpe_prediction['error']
                else:
                    char_prediction['offset'] = (char_prediction['offset'][0], end_c)

        assert(char_prediction.get('offset', None) is not None)
        char_predictions.append(char_prediction)

    return char_predictions


def sparsify_prediction(prediction: List[str]) -> List[Dict]:
    """

    Args:
        prediction: a list of prediction labels as returned by the model. The BOS token is not included, nor the
            candidate special token

    Returns: token_based predictions. For consistency with token-based labels, I need to consider also
        the BOS token and the candidate special token. For this reason, the first token is indexed as 2 and the last
        one as len(prediction)+1

    """
    sparse_prediction = []
    current_predicted_span = {}
    for token_idx, token_pred in enumerate(prediction):
        if token_pred == 'O':
            if current_predicted_span.get('offset', None) is not None:
                sparse_prediction.append(current_predicted_span)
                current_predicted_span = {}

        elif token_pred.split('-')[0] == 'B':
            if current_predicted_span.get('offset', None) is not None:
                sparse_prediction.append(current_predicted_span)

            current_predicted_span = {
                'offset': (token_idx+2, token_idx + 3),
                'error': token_pred.split('-')[1],
            }

        elif token_pred.split('-')[0] == 'I':
            if current_predicted_span.get('offset', None) is None:
                current_predicted_span = {
                    'offset': (token_idx+2, token_idx + 3),
                    'error': token_pred.split('-')[1],
                }
            else:
                current_predicted_span['offset'] = (current_predicted_span['offset'][0], token_idx + 3)
        else:
            logging.warning("This prediction is ill-formed!", prediction)
            sys.exit(1)

    if current_predicted_span.get('offset', None) is not None:
        sparse_prediction.append(current_predicted_span)

    return sparse_prediction
