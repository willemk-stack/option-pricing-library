import numpy as np
import pandas as pd


def surface_domain_report(surface, *, quotes_df=None, forward=None, atol=1e-12):
    rows = []
    q = quotes_df.copy() if quotes_df is not None else None
    for s in surface.smiles:
        T = float(s.T)
        y_model_min = float(s.y_min)
        y_model_max = float(s.y_max)
        row = {
            "T": T,
            "y_model_min": y_model_min,
            "y_model_max": y_model_max,
            "y_data_min": np.nan,
            "y_data_max": np.nan,
            "slack_left": np.nan,
            "slack_right": np.nan,
        }
        if forward is not None:
            row["F"] = float(forward(T))
        if q is not None and not q.empty:
            if "T" in q.columns:
                qq = q.loc[
                    np.isclose(q["T"].astype(float), T, atol=atol, rtol=0.0)
                ].copy()
            else:
                qq = q
            if not qq.empty:
                if "y" in qq.columns:
                    y_data = qq["y"].astype(float).to_numpy()
                elif "K" in qq.columns:
                    if forward is None:
                        raise ValueError(
                            "quotes_df has K but not y; forward(T) is required"
                        )
                    F = float(forward(T))
                    K = qq["K"].astype(float).to_numpy()
                    y_data = np.log(K / F)
                else:
                    raise ValueError("quotes_df must contain 'y' or 'K'")
                y0 = float(np.min(y_data))
                y1 = float(np.max(y_data))
                row["y_data_min"] = y0
                row["y_data_max"] = y1
                row["slack_left"] = y0 - y_model_min
                row["slack_right"] = y_model_max - y1
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("T").reset_index(drop=True)
    return df
