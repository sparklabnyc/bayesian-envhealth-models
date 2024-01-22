"""Script to run models from src/bayesian_envhealth_models/model.py."""

import logging

import hydra
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
from omegaconf import DictConfig
from beartype.typing import Callable
from beartype import beartype

from bayesian_envhealth_models.models import (
    model_intercept,
    model_age_time_interaction,
    model_age_space_time_race,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

log = logging.getLogger(__name__)


@beartype
def model_factory(model_name: str) -> Callable:
    if model_name == "model_intercept":
        return model_intercept
    if model_name == "model_age_time_interaction":
        return model_age_time_interaction
    if model_name == "model_age_space_time_race":
        return model_age_space_time_race
    else:
        raise ValueError("Invalid model name")


@beartype
def var_factory(model_name: str) -> list[str]:
    if model_name == "model_intercept":
        return ["deaths", "population"]
    if model_name == "model_age_time_interaction":
        return ["age_id", "time_id", "deaths", "population"]
    if model_name == "model_age_space_time_race":
        # s1 is state, space is county, lookup is from state to county, time is year
        return [
            "age_id",
            "s1_id",
            "space_id",
            "time_id",
            "race_id",
            "lookup",
            "deaths",
            "population",
        ]
    else:
        raise ValueError("Invalid model name")


@beartype
def load_data(data_path: str, vars: list[str]) -> dict[str, jnp.ndarray]:
    """Load data from data_path."""
    data_list = [jnp.load("{}{}.npy".format(data_path, var)) for var in vars]
    data = dict(zip(vars, data_list))
    return data


@beartype
def print_model_shape(
    model: Callable,
    data: dict[str, jnp.ndarray],
) -> None:
    """Helper function to get model shapes."""
    with numpyro.handlers.seed(rng_seed=1):
        trace = numpyro.handlers.trace(model).get_trace(**data)
    print(numpyro.util.format_shapes(trace))


@beartype
def run_inference(
    model: Callable,
    data: dict[str, jnp.ndarray],
    rng_key: jnp.ndarray,
    args,
) -> numpyro.infer.mcmc.MCMC:
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        thinning=args.thin,
        chain_method=args.chain_method,
        progress_bar=True,
    )
    mcmc.run(rng_key, **data)
    # mcmc.print_summary()

    extra_fields = mcmc.get_extra_fields()
    log.info("Number of divergences: {}".format(jnp.sum(extra_fields["diverging"])))

    return mcmc


@hydra.main(version_base=None, config_path="conf", config_name="model_config")
def main(cfg: DictConfig) -> None:
    numpyro.set_platform(cfg.run.device)
    numpyro.set_host_device_count(cfg.model.sampler_config.chains)

    numpyro.enable_x64()

    model = model_factory(cfg.model.name)
    vars = var_factory(cfg.model.name)

    log.info("Fetching data...")

    data = load_data(
        data_path=cfg.dir.data,
        vars=vars,
    )

    print_model_shape(
        model=model,
        data=data,
    )

    log.info("Starting inference...")
    rng_key = random.PRNGKey(cfg.model.sampler_config.seed)
    samples = run_inference(
        model=model,
        data=data,
        rng_key=rng_key,
        args=cfg.model.sampler_config,
    )

    if cfg.run.save:
        import arviz as az

        samples = az.from_numpyro(samples)
        # only save posterior samples
        # do not save observed data, which contains deaths
        samples["posterior"].to_netcdf(
            "{}{}_samples.nc".format(cfg.dir.output, cfg.model.name)
        )


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.13.2")

    main()
