import streamlit as st

import pandas as pd

from matese.utils import utils
from matese.metric import MaTESe
from matese.data import data_utils


@st.cache_resource
def load_metric(metric_name: str, device: str) -> MaTESe:
    metric = MaTESe.load_metric(metric_name, device)
    return metric


def main():
    st.title("MaTESe")

    metric_name = st.text_input(
        "Write the metric name (either 'matese' or 'matese-qe')",
    )
    device = st.selectbox(
        "Choose a device",
        ["cpu", "cuda"],
        index=0
    )

    metric = load_metric(metric_name, device)

    source, reference, candidate = None, None, None
    candidate = st.text_input("Insert the candidate translation")
    if metric.reference_less:
        source = st.text_input("Insert the source sentence")
    else:
        source = st.text_input("Insert the source sentence")
        reference = st.text_input("Insert the reference translation")

    if (metric.reference_less and source and candidate) or \
            (not metric.reference_less and source and reference and candidate):

        predictions = metric.evaluate([candidate], [source], [reference])
        spans = [prediction['spans'] for prediction in predictions][0]

        errors = pd.DataFrame({"offset": [], "error": []})
        for idx, span in enumerate(spans[-1::-1]):
            start_c, end_c = span["offset"]
            error = span["error"]
            errors = pd.concat([errors, pd.DataFrame(
                {"offset": [(start_c, end_c)], "error": [error]}
            )], ignore_index=True)
            if error == data_utils.MQMSeverities.minor:
                candidate = (
                    candidate[:start_c]
                    + "<b><span style='color:orange'>"
                    + candidate[start_c:end_c]
                    + "</span></b>"
                    + candidate[end_c:]
                )
            elif error == data_utils.MQMSeverities.major:
                candidate = (
                    candidate[:start_c]
                    + "<b><span style='color:red'>"
                    + candidate[start_c:end_c]
                    + "</span></b>"
                    + candidate[end_c:]
                )

        st.markdown(
            "<h1 style='text-align: center; font-size: 28px;'>Prediction</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(candidate, unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; font-size: 28px;'>Error spans</h1>",
            unsafe_allow_html=True,
        )

        st.table(errors)


if __name__ == "__main__":
    main()
