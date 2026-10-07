"""A stand-in for a billable language model, so the example shows cost metering.

It is a logistic regression on the text channel that records one priced call per row
it scores, against the fictional ``example-llm-small`` in prices.yaml.
"""

from sklearn.linear_model import LogisticRegression

from accbench.methods import SklearnMethod, register


class StubLLM(SklearnMethod):
    name = "stub_llm"
    channel_kinds = frozenset({"text"})

    def make(self, seed, threads):
        return LogisticRegression(C=0.5, max_iter=1000, random_state=seed)

    def predict_proba(self, X, *, meter):
        for _ in range(X.shape[0]):
            meter.record("example-llm-small", input_tokens=400, output_tokens=5)
        return super().predict_proba(X, meter=meter)


register(StubLLM)
