"""Script to run models from src/bayesian_envhealth_models/model.py."""

import logging
from typing import Callable, Any

import hydra
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from omegaconf import DictConfig

from bayesian_envhealth_models.models import model_intercept

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

log = logging.getLogger(__name__)


def model_factory(model_name: str) -> Callable:
    if model_name == "model_intercept":
        return model_intercept
    else:
        raise ValueError("Invalid model name")


def run_inference(
    model: Callable,
    data: dict[str, jnp.ndarray],
    rng_key: jnp.ndarray,
    args,
) -> Any:
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thin,
        chain_method=args.chain_method,
        progress_bar=True,
    )
    mcmc.run(rng_key, **data)
    # mcmc.print_summary()

    extra_fields = mcmc.get_extra_fields()
    log.info("Number of divergences: {}".format(jnp.sum(extra_fields["diverging"])))

    return mcmc.get_samples(group_by_chain=True)


@hydra.main(version_base=None, config_path="conf", config_name="model_config")
def main(cfg: DictConfig) -> None:
    log.info(cfg)
    numpyro.set_platform(cfg.run.device)
    numpyro.set_host_device_count(cfg.model.sampler_config.chains)

    numpyro.enable_x64()

    # model = model_factory(cfg.model.name)
    # rng_key = random.PRNGKey(cfg.run.seed)

    pass


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.13.2")

    main()
