from typing import List, Dict
import json
from pathlib import Path

from matese.metric import MaTESe


def read_data_file(filepath: str) -> List[str]:
    if filepath is None:
        return []
    with open(filepath, 'r') as fin:
        lines = [line.strip() for line in fin.readlines()]

    return lines


def write_scores(filepath: str, scores: List[int]):
    with open(filepath, 'w') as fout:
        for score in scores:
            fout.write(str(score))
            fout.write("\n")


def write_spans(filepath: str, spans: List[List[Dict]]):
    with open(filepath, 'w') as fout:
        for sample_spans in spans:
            json.dump(sample_spans, fout)
            fout.write("\n")


def main(args):
    device = 'cuda' if not args.cpu else 'cpu'
    metric = MaTESe.load_metric(args.metric, device)

    references = read_data_file(args.references) if not metric.reference_less else None
    sources = read_data_file(args.sources) if metric.reference_less else None
    candidates = read_data_file(args.candidates)

    if (sources and candidates) or (references and candidates):

        predictions = metric.evaluate(candidates, sources, references)
        spans = [prediction["spans"] for prediction in predictions]
        scores = [prediction["score"] for prediction in predictions]

        print(f"Scores: {scores}")
        print(f"Spans: {spans}")

        output_path_prefix = str(Path(args.output).with_suffix(""))
        write_scores(output_path_prefix + ".scores.txt", scores)
        write_spans(output_path_prefix + ".spans.txt", spans)
