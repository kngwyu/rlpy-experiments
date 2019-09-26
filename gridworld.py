from rlpy.Agents import Agent, LSPI, NaturalActorCritic, Q_Learning
from rlpy.Domains import GridWorld
from rlpy.Representations import iFDDK, IndependentDiscretization, Tabular
from rlpy.Policies import eGreedy, GibbsPolicy
import os
from typing import Optional

from common import run_cli


MAZE = os.path.join(GridWorld.default_map_dir, '11x11-Rooms.txt')
DOMAIN = GridWorld(MAZE, noise=0.3)
MAX_STEPS = 10000


def select_agent(name: Optional[str], _seed: int) -> Agent:
    tabular = Tabular(DOMAIN, discretization=20)
    if name is None or name == 'lspi':
        policy = eGreedy(tabular, epsilon=0.1)
        return LSPI(policy, tabular, DOMAIN.discount_factor, MAX_STEPS, 1000)
    elif name == 'nac':
        return NaturalActorCritic(
            GibbsPolicy(tabular),
            tabular,
            DOMAIN.discount_factor,
            forgetting_rate=0.3,
            min_steps_between_updates=100,
            max_steps_between_updates=1000,
            lambda_=0.7,
            learn_rate=0.1
        )
    elif name == 'tabular-q':
        return Q_Learning(
            eGreedy(tabular, epsilon=0.1),
            tabular,
            discount_factor=DOMAIN.discount_factor,
            lambda_=0.3,
            initial_learn_rate=0.11,
            learn_rate_decay_mode='boyan',
            boyan_N0=100,
        )
    elif name == 'ifddk-q':
        lambda_ = 0.3
        ifddk = iFDDK(
            DOMAIN,
            discovery_threshold=1.0,
            initial_representation=IndependentDiscretization(DOMAIN),
            sparsify=True,
            useCache=True,
            lazy=True,
            lambda_=lambda_,
        )
        return Q_Learning(
            eGreedy(ifddk, epsilon=0.1),
            ifddk,
            discount_factor=DOMAIN.discount_factor,
            lambda_=lambda_,
            initial_learn_rate=0.11,
            learn_rate_decay_mode='boyan',
            boyan_N0=100,
        )
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    run_cli(
        DOMAIN,
        select_agent,
        default_max_steps=MAX_STEPS,
        default_num_policy_checks=10,
        default_checks_per_policy=50,
    )
