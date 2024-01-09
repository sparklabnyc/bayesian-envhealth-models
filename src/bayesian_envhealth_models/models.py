import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float, jaxtyped
from typing import Any, Optional
from beartype import beartype


@jaxtyped(typechecker=beartype)
def model_intercept(
    population: Float[Array, "data"],
    deaths: Optional[Float[Array, "data"]] = None,
) -> tuple[Any, list]:
    """Model with intercept only. Binomial likelihood."""
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
