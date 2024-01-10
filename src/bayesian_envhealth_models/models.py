import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam
from jaxtyping import Array, jaxtyped, Int
import jax.numpy as jnp
from typing import Optional
from beartype import beartype


@jaxtyped(typechecker=beartype)
def model_intercept(
    population: Int[Array, "data"],
    deaths: Optional[Int[Array, "data"]] = None,
) -> None:
    """Model with intercept only. Binomial likelihood."""
    N = len(population)

    # hyperparameters
    intercept = numpyro.sample("intercept", dist.Normal(loc=0.0, scale=10.0))

    # likelihood
    with numpyro.plate("N", size=N):
        numpyro.sample(
            "deaths",
            dist.Binomial(total_count=population, logits=intercept),
            obs=deaths,
        )


@numpyro.handlers.reparam(
    config={
        k: LocScaleReparam(0)
        for k in [
            "age_drift",
            "age_time_drift",
        ]
    }
)
@jaxtyped(typechecker=beartype)
def model_age_time_interaction(
    age_id: Int[Array, "data"],
    time_id: Int[Array, "data"],
    population: Int[Array, "data"],
    deaths: Optional[Int[Array, "data"]] = None,
) -> None:
    """
    Model with:
    - intercept (first term of random walk over age effect)
    - slope over time
    - random walk over age effect
    - age-specific random walk over time (type II interaction)
    - Binomial likelihood
    """
    N = len(population)
    N_age = len(np.unique(age_id))
    N_t = len(np.unique(time_id))

    # plates
    age_plate = numpyro.plate("age_groups", size=N_age, dim=-2)
    time_plate = numpyro.plate("time", size=(N_t - 1), dim=-1)

    # hyperparameters
    slope = numpyro.sample("slope", dist.Normal(loc=0.0, scale=1.0))
    sigma_rw_age = numpyro.sample("sigma_rw_age", dist.HalfNormal(1.0))
    sigma_rw_age_time = numpyro.sample("sigma_rw_age_time", dist.HalfNormal(1.0))

    # slope over time is the same as adding slope at each timestep
    slope_cum = slope * jnp.arange(N_t)

    # random walk over age
    with age_plate:
        age_drift_scale = jnp.pad(
            jnp.broadcast_to(sigma_rw_age, N_age - 1),
            (1, 0),
            constant_values=10.0,  # pad so first term is the intercept, prior N(0, 10)
        )[:, jnp.newaxis]
        age_drift = numpyro.sample("age_drift", dist.Normal(0, age_drift_scale))
        age_effect = jnp.cumsum(age_drift, -2)

    # age-time random walk (type II) interaction
    with age_plate, time_plate:
        age_time_drift = numpyro.sample(
            "age_time_drift", dist.Normal(0, sigma_rw_age_time)
        )
        age_time_effect = jnp.pad(jnp.cumsum(age_time_drift, -1), [(0, 0), (1, 0)])

    latent_rate = slope_cum + age_effect + age_time_effect

    # likelihood
    with numpyro.plate("N", size=N):
        mu_logit = latent_rate[age_id, time_id]
        numpyro.sample(
            "deaths",
            dist.Binomial(total_count=population, logits=mu_logit),
            obs=deaths,
        )
