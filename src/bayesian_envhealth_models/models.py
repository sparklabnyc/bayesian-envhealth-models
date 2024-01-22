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


@numpyro.handlers.reparam(
    config={
        k: LocScaleReparam(0)
        for k in [
            "age_drift",
            "space_s1",
            "space_effect",
            "time_drift",
            "race_effect",
        ]
    }
)
@jaxtyped(typechecker=beartype)
def model_age_space_time_race(
    age_id: Int[Array, "data"],
    s1_id: Int[Array, "data"],
    space_id: Int[Array, "data"],
    lookup: Int[Array, "space"],
    time_id: Int[Array, "data"],
    race_id: Int[Array, "data"],
    population: Int[Array, "data"],
    deaths: Optional[Int[Array, "data"]] = None,
):
    """
    Model with:
    - intercept (first term of random walk over age effect)
    - random walk over age effect
    - two-tier hierarchy over space intercepts
    - random effect for race
    - Binomial likelihood
    """
    N = len(population)
    N_age = len(np.unique(age_id))
    N_s1 = len(np.unique(s1_id))  # states
    N_space = len(np.unique(space_id))  # counties
    N_t = len(np.unique(time_id))
    N_race = len(np.unique(race_id))

    # plates
    age_plate = numpyro.plate("age_groups", size=N_age, dim=-3)
    space_plate = numpyro.plate("space", size=N_space, dim=-2)
    race_plate = numpyro.plate("races", size=N_race, dim=-4)
    time_plate = numpyro.plate("time", size=(N_t - 1), dim=-1)

    # hyperparameters
    sigma_age = numpyro.sample("sigma_age", dist.HalfNormal(1.0))
    sigma_s1 = numpyro.sample("sigma_s1", dist.HalfNormal(1.0))
    sigma_space = numpyro.sample("sigma_space", dist.HalfNormal(1.0))
    sigma_race = numpyro.sample("sigma_race", dist.HalfNormal(1.0))
    sigma_time = numpyro.sample("sigma_time", dist.HalfNormal(1.0))

    # age
    with age_plate:
        age_drift_scale = jnp.pad(
            jnp.broadcast_to(sigma_age, N_age - 1),
            (1, 0),
            constant_values=10.0,  # pad so first term is alpha0, prior N(0, 10)
        )[:, jnp.newaxis, jnp.newaxis]
        age_drift = numpyro.sample("age_drift", dist.Normal(0, age_drift_scale))
        age_effect = jnp.cumsum(age_drift, -3)

    # spatial hierarchy
    # N_s1 is states
    # N_space is counties
    with numpyro.plate("s1", N_s1, dim=-2):
        space_s1 = numpyro.sample("space_s1", dist.Normal(0, sigma_s1))
    with space_plate:
        space_effect = numpyro.sample(
            "space_effect", dist.Normal(space_s1[lookup], sigma_space)
        )

    # race
    with race_plate:
        race_effect = numpyro.sample("race_effect", dist.Normal(0, sigma_race))

    # time
    # random walk over time
    with time_plate:
        time_drift = numpyro.sample("time_drift", dist.Normal(0, sigma_time))
        time_effect = jnp.pad(jnp.cumsum(time_drift, -1), (1, 0))

    latent_rate = age_effect + space_effect + race_effect + time_effect

    # likelihood
    with numpyro.plate("N", size=N):
        mu_logit = latent_rate[race_id, age_id, space_id, time_id]
        numpyro.sample(
            "deaths",
            dist.Binomial(total_count=population, logits=mu_logit),
            obs=deaths,
        )
