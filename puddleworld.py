from rlpy.Agents import Agent, LSPI, SARSA, Q_Learning
from rlpy.Domains import PuddleWorld
from rlpy.Representations import gaussian_kernel, iFDD, \
    IndependentDiscretization, KernelizediFDD, RBF, Tabular
from rlpy.Policies import eGreedy
from typing import Optional

from common import run_cli


DOMAIN = PuddleWorld()
MAX_STEPS = 40000


def select_agent(name: Optional[str], seed: int) -> Agent:
    tabular = Tabular(DOMAIN, discretization=20)
    if name is None or name == 'tabular-lspi':
        policy = eGreedy(tabular, epsilon=0.1)
        return LSPI(policy, tabular, DOMAIN.discount_factor, MAX_STEPS, 1000)
    elif name == 'tabular-q':
        policy = eGreedy(tabular, epsilon=0.1)
        return Q_Learning(policy, tabular, DOMAIN.discount_factor, lambda_=0.3)
    elif name == 'tabular-sarsa':
        policy = eGreedy(tabular, epsilon=0.1)
        return SARSA(policy, tabular, DOMAIN.discount_factor, lambda_=0.3)
    elif name == 'ifdd-q':
        lambda_, boyan_N0 = 0.42, 202
        discretization = 18
        initial_rep = IndependentDiscretization(DOMAIN, discretization=discretization)
        ifdd = iFDD(
            DOMAIN,
            discovery_threshold=8.63917,
            initial_representation=initial_rep,
            useCache=True,
            iFDDPlus=True,
        )
        return Q_Learning(
            eGreedy(ifdd, epsilon=0.1),
            ifdd,
            discount_factor=DOMAIN.discount_factor,
            lambda_=lambda_,
            initial_learn_rate=0.7422,
            learn_rate_decay_mode='boyan',
            boyan_N0=boyan_N0,
        )
    elif name == 'kifdd-q':
        lambda_, boyan_N0 = 0.52738, 389.56
        kernel_resolution = 8.567677
        kernel_width = (DOMAIN.statespace_limits[:, 1] - DOMAIN.statespace_limits[:, 0]) \
            / kernel_resolution
        kifdd = KernelizediFDD(
            DOMAIN,
            sparsify=True,
            kernel=gaussian_kernel,
            kernel_args=[kernel_width],
            active_threshold=0.01,
            discover_threshold=0.0807,
            normalization=True,
            max_active_base_feat=10,
            max_base_feat_sim=0.5
        )
        policy = eGreedy(kifdd, epsilon=0.1)
        return Q_Learning(
            policy,
            kifdd,
            discount_factor=DOMAIN.discount_factor,
            lambda_=lambda_,
            initial_learn_rate=0.4244,
            learn_rate_decay_mode='boyan',
            boyan_N0=boyan_N0,
        )
    elif name == 'rbfs-q':
        rbf = RBF(
            DOMAIN,
            num_rbfs=96,
            resolution_max=21.0,
            resolution_min=21.0,
            const_feature=False,
            normalize=True,
            seed=seed,
        )
        policy = eGreedy(rbf, epsilon=0.1)
        return Q_Learning(
            policy,
            rbf,
            discount_factor=DOMAIN.discount_factor,
            lambda_=0.1953,
            initial_learn_rate=0.6633,
            learn_rate_decay_mode='boyan',
            boyan_N0=13444.0,
        )
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    run_cli(
        DOMAIN,
        select_agent,
        default_max_steps=MAX_STEPS,
        default_num_policy_checks=20,
        default_checks_per_policy=100,
    )
