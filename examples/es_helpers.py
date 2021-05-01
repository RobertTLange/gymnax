import jax
import jax.numpy as jnp
from jax import jit, vmap
import functools
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(context='poster', style='white', palette='Paired',
        font='sans-serif', font_scale=1.05, color_codes=True, rc=None)


def init_cma_es(mean_init, sigma_init, population_size, mu):
    ''' Initialize evolutionary strategy & learning rates. '''
    n_dim = mean_init.shape[0]
    weights_prime = jnp.array(
        [jnp.log((population_size + 1) / 2) - jnp.log(i + 1)
         for i in range(population_size)])
    mu_eff = ((jnp.sum(weights_prime[:mu]) ** 2) /
               jnp.sum(weights_prime[:mu] ** 2))
    mu_eff_minus = ((jnp.sum(weights_prime[mu:]) ** 2) /
                     jnp.sum(weights_prime[mu:] ** 2))

    # lrates for rank-one and rank-μ C updates
    alpha_cov = 2
    c_1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
    c_mu = jnp.minimum(1 - c_1 - 1e-8, alpha_cov * (mu_eff - 2 + 1 / mu_eff)
              / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2))
    min_alpha = min(1 + c_1 / c_mu,
                    1 + (2 * mu_eff_minus) / (mu_eff + 2),
                    (1 - c_1 - c_mu) / (n_dim * c_mu))
    positive_sum = jnp.sum(weights_prime[weights_prime > 0])
    negative_sum = jnp.sum(jnp.abs(weights_prime[weights_prime < 0]))
    weights = jnp.where(weights_prime >= 0,
                       1 / positive_sum * weights_prime,
                       min_alpha / negative_sum * weights_prime,)
    weights_truncated = jax.ops.index_update(weights, jax.ops.index[mu:], 0)
    c_m = 1

    # lrate for cumulation of step-size control and rank-one update
    c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
    d_sigma = 1 + 2 * jnp.maximum(0, jnp.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
    chi_d = jnp.sqrt(n_dim) * (
        1.0 - (1.0 / (4.0 * n_dim)) + 1.0 / (21.0 * (n_dim ** 2)))

    # Initialize evolution paths & covariance matrix
    p_sigma = jnp.zeros(n_dim)
    p_c = jnp.zeros(n_dim)
    C, D, B = jnp.eye(n_dim), None, None

    memory = {"p_sigma": p_sigma, "p_c": p_c, "sigma": sigma_init,
              "mean": mean_init, "C": C, "D": D, "B": B,
              "generation": 0}

    params = {"mu_eff": mu_eff,
              "c_1": c_1, "c_mu": c_mu, "c_m": c_m,
              "c_sigma": c_sigma, "d_sigma": d_sigma,
              "c_c": c_c, "chi_d": chi_d,
              "weights": weights,
              "weights_truncated": weights_truncated,
              "pop_size": population_size,
              "n_dim": n_dim,
              "tol_x": 1e-12 * sigma_init,
              "tol_x_up": 1e4,
              "tol_fun": 1e-12,
              "tol_condition_C": 1e14,
              "min_generations": 10}
    return params, memory

@functools.partial(jax.jit, static_argnums=(4, 5))
def sample(rng, memory, B, D, n_dim, pop_size):
    """ Jittable Gaussian Sample Helper. """
    z = jax.random.normal(rng, (n_dim, pop_size)) # ~ N(0, I)
    y = B.dot(jnp.diag(D)).dot(z)               # ~ N(0, C)
    y = jnp.swapaxes(y, 1, 0)
    x = memory["mean"] + memory["sigma"] * y    # ~ N(m, σ^2 C)
    return x


@jax.jit
def eigen_decomposition(C, B, D):
    """ Perform eigendecomposition of covariance matrix. """
    if B is not None and D is not None:
        return C, B, D
    C = (C + C.T) / 2
    D2, B = jnp.linalg.eigh(C)
    D = jnp.sqrt(jnp.where(D2 < 0, 1e-20, D2))
    C = jnp.dot(jnp.dot(B, jnp.diag(D ** 2)), B.T)
    return C, B, D


def check_termination(values, params, memory):
    """ Check whether to terminate CMA-ES loop. """
    dC = jnp.diag(memory["C"])
    C, B, D = eigen_decomposition(memory["C"], memory["B"], memory["D"])

    # Stop if generation fct values of recent generation is below thresh.
    if (memory["generation"] > params["min_generations"]
        and jnp.max(values) - jnp.min(values) < params["tol_fun"]):
        print("TERMINATE ----> Convergence/No progress in objective")
        return True

    # Stop if std of normal distrib is smaller than tolx in all coordinates
    # and pc is smaller than tolx in all components.
    if jnp.all(memory["sigma"] * dC < params["tol_x"]) and np.all(
        memory["sigma"] * memory["p_c"] < params["tol_x"]):
        print("TERMINATE ----> Convergence/Search variance too small")
        return True

    # Stop if detecting divergent behavior.
    if memory["sigma"] * jnp.max(D) > params["tol_x_up"]:
        print("TERMINATE ----> Stepsize sigma exploded")
        return True

    # No effect coordinates: stop if adding 0.2-standard deviations
    # in any single coordinate does not change m.
    if jnp.any(memory["mean"] == memory["mean"] + (0.2 * memory["sigma"] * jnp.sqrt(dC))):
        print("TERMINATE ----> No effect when adding std to mean")
        return True

    # No effect axis: stop if adding 0.1-standard deviation vector in
    # any principal axis direction of C does not change m.
    if jnp.all(memory["mean"] == memory["mean"] + (0.1 * memory["sigma"]
                                * D[0] * B[:, 0])):
        print("TERMINATE ----> No effect when adding std to mean")
        return True

    # Stop if the condition number of the covariance matrix exceeds 1e14.
    condition_cov = jnp.max(D) / jnp.min(D)
    if condition_cov > params["tol_condition_C"]:
        print("TERMINATE ----> C condition number exploded")
        return True
    return False


def init_logger(top_k, num_params):
    evo_logger = {"top_values": jnp.zeros(top_k) + 1e10,
                  "top_params": jnp.zeros((top_k, num_params)),
                  "log_top_1": [],
                  "log_top_mean": [],
                  "log_top_std": [],
                  "log_gen_1": [],
                  "log_gen_mean": [],
                  "log_gen_std": [],
                  "log_sigma": [],
                  "log_gen": []}
    return evo_logger


def update_logger(evo_logger, x, fitness, memory, top_k, verbose=False):
    """ Helper function to keep track of top solutions. """
    # Check if there are solutions better than current archive
    vals = jnp.hstack([evo_logger["top_values"], fitness])
    params = jnp.vstack([evo_logger["top_params"], x])
    concat_top = jnp.hstack([jnp.expand_dims(vals, 1), params])
    sorted_top = concat_top[concat_top[:, 0].argsort()]

    # Importantly: Params are stored as flat vectors
    evo_logger["top_values"] = sorted_top[:top_k, 0]
    evo_logger["top_params"] = sorted_top[:top_k, 1:]
    evo_logger["log_top_1"].append(evo_logger["top_values"][0])
    evo_logger["log_top_mean"].append(jnp.mean(evo_logger["top_values"]))
    evo_logger["log_top_std"].append(jnp.std(evo_logger["top_values"]))
    evo_logger["log_gen_1"].append(jnp.min(fitness))
    evo_logger["log_gen_mean"].append(jnp.mean(fitness))
    evo_logger["log_gen_std"].append(jnp.std(fitness))
    evo_logger["log_sigma"].append(memory["sigma"])
    evo_logger["log_gen"].append(memory["generation"])

    if verbose:
        print(evo_logger["log_gen"][-1], evo_logger["top_values"])
    return evo_logger


def plot_fitness(evo_logger, title, ylims=(0, 10), fig=None, ax=None,
                 no_legend=False):
    """ Plot fitness trajectory from evo logger over generations. """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(evo_logger["log_gen"], evo_logger["log_top_1"],
            label="Top 1")
    ax.plot(evo_logger["log_gen"], evo_logger["log_top_mean"],
            label="Top-k Mean")
    ax.plot(evo_logger["log_gen"], evo_logger["log_gen_1"],
            label="Gen. 1")
    ax.plot(evo_logger["log_gen"], evo_logger["log_gen_mean"],
            label="Gen. Mean")
    ax.set_ylim(ylims)
    if not no_legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Number of Generations")
    ax.set_ylabel("Fitness Score")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def plot_sigma(evo_logger, title, ylims=(0, 1.5), fig=None, ax=None):
    """ Plot sigma trace from evo logger over generations. """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(evo_logger["log_gen"], evo_logger["log_sigma"])
    ax.set_ylim(ylims)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Number of Generations")
    ax.set_ylabel(r"Stepsize: $\sigma$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax
