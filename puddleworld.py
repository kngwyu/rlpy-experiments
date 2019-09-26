from rlpy.Agents import Agent, LSPI, NaturalActorCritic, SARSA, Q_Learning
from rlpy.Domains import PuddleWorld
from rlpy.Representations import iFDD, IndependentDiscretization, Tabular
from rlpy.Policies import eGreedy, GibbsPolicy
from typing import Optional

from common import run_cli


DOMAIN = PuddleWorld()
MAX_STEPS = 40000


def select_agent(name: Optional[str]) -> Agent:
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
        lambda_ = 0.42
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
            boyan_N0=202.0,
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
