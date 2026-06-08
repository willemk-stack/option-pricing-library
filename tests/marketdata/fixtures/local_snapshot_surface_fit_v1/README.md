# local_snapshot_surface_fit_v1

Provider-neutral synthetic option snapshot for model-validation bundle and
surface market-fit workflow tests.

The fixture is fully synthetic and redistributable. It contains one synthetic
underlying (`SYNTH`), one synthetic market-input row, and 21 vanilla option
quotes across three expiries. Prices and implied volatilities were generated
from a small smooth synthetic volatility surface, then rounded for stable CSV
storage.

The fixture intentionally contains no provider payloads, screenshots, raw
option-chain exports, or provider-derived market data. Its purpose is to prove
API and artifact flow from local fixture to model-ready bundle to SVI/eSSVI
workflow preparation and fitting.
