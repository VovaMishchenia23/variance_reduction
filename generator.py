from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_samples(
        mu_metric: float,
        sigma_metric: float,
        epsilon: float,
        treatment_effect: float,
        size: int,
        cov_mu_eps: Optional[List[Tuple[float, float]]] = None,
        non_linear: Optional[List[bool]] = None,
        p_binomial: float = 0.5,
        random_epsilon: float = 1.0,
        seed: int = None,
) -> pd.DataFrame:

    if seed:
        np.random.seed(seed)

    metric_before = np.random.normal(loc=mu_metric, scale=sigma_metric, size=size)

    error_term = np.random.normal(loc=0, scale=epsilon, size=size)

    treatment_assignment = np.random.binomial(n=1, p=p_binomial, size=size)

    metric_during = metric_before + error_term + treatment_assignment * treatment_effect

    if cov_mu_eps:
        covariates = np.zeros((len(cov_mu_eps), size))
        for i, mu_eps in enumerate(cov_mu_eps):
            covariates[i, :] = np.random.normal(mu_eps[0], mu_eps[1], size)
            if non_linear and non_linear[i]:
                metric_during = metric_during + covariates[i, :] ** 2
            else:
                metric_during = metric_during + covariates[i, :]
    else:
        covariates = []

    d = {
        "Y": metric_during,
        "T": treatment_assignment,
        "Y_before": metric_before,
    }
    d_cov = {f"X_{i+1}": c for i, c in enumerate(covariates)}
    r_cov = {"R_1": np.random.normal(loc=0, scale=random_epsilon, size=size)}
    return pd.DataFrame({**d, **d_cov, **r_cov})
#%%
