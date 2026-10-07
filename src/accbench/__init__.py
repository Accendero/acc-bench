"""accbench: seven checked layers between a model and a claim.

Each unit enforces one requirement and writes one artifact:

    0 construct   construct.yaml       R0 meaning
    1 fixtures    fixtures/<hash>/     R1 sameness
    2 channels    channels.yaml        R2 isolation
    3 runner      runs/<cell>.json     R3 completeness
    4 resolution  resolution.json      R4 resolution
    5 rules       verdicts.json        R5 sound conclusions
    6 claims      claims.jsonl         R6 accountable claims

The provenance library (X1) stamps every artifact on write and refuses a
stale one on read.
"""

__version__ = "0.1.0.dev0"

UNITS = (
    "construct",
    "fixtures",
    "channels",
    "runner",
    "resolution",
    "rules",
    "claims",
)
