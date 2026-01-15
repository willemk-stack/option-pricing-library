from option_pricing import PricingContext
from option_pricing.instruments.vanilla import VanillaOption

# expiry: float,
# strike: float,
# kind: OptionType,


def plot_pde_bs_errorbars(
    inst: VanillaOption,
    sigma: float,
    ctx: PricingContext,
):
    # price_bs = bs_price_instrument_from_ctx(inst=inst, sigma=sigma, ctx=ctx)

    return None


def plot_pde_convergence():
    return None


def plot_pde_error_scaling():
    return None
