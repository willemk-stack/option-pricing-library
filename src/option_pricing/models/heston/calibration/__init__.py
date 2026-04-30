from typing import Literal

type ObjectiveKind = Literal[
    "price",
    "vega_price",
    "bid_ask_price",
    "iv",
    "relative_price",
]
