import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float, jaxtyped, Int
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


@jaxtyped(typechecker=beartype)
def model_age_time_interaction(
    age_id: Float[Array, "data"],
    time_id: Float[Array, "data"],
    population: Float[Array, "data"],
    deaths: Optional[Float[Array, "data"]] = None,
) -> None:
    """Model with age effect, random walk over time, and type II interaction. Binomial likelihood."""
    N = len(population)

    # hyperparameters
    intercept = numpyro.sample("alpha", dist.Normal(loc=0.0, scale=10.0))

    # likelihood
    with numpyro.plate("N", size=N):
        numpyro.sample(
            "deaths",
            dist.Binomial(total_count=population, logits=intercept),
            obs=deaths,
        )
