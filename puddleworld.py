from rlpy.Agents import LSPI, NaturalActorCritic, SARSA, Q_Learning
from rlpy.Domains import PuddleWorld
from rlpy.Representations import iFDD, IndependentDiscretization, Tabular
from rlpy.Policies import eGreedy, GibbsPolicy

from common import run_cli


def main():
    domain = PuddleWorld()
    max_steps = 40000

    def agent_selector(name):
        tabular = Tabular(domain, discretization=20)
        if name is None or name == 'tabular-lspi':
            policy = eGreedy(tabular, epsilon=0.1)
            return LSPI(policy, tabular, domain.discount_factor, max_steps, 1000)
        elif name == 'tabular-q':
            policy = eGreedy(tabular, epsilon=0.1)
            return Q_Learning(policy, tabular, domain.discount_factor, lambda_=0.3)
        elif name == 'tabular-sarsa':
            policy = eGreedy(tabular, epsilon=0.1)
            return SARSA(policy, tabular, domain.discount_factor, lambda_=0.3)
        elif name == 'ifdd-q':
            lambda_ = 0.42
            discretization = 18
            initial_rep = IndependentDiscretization(domain, discretization=discretization)
            ifdd = iFDD(
                domain,
                discovery_threshold=8.63917,
                initial_representation=initial_rep,
                useCache=True,
                iFDDPlus=True,
            )
            return Q_Learning(
                eGreedy(ifdd, epsilon=0.1),
                ifdd,
                discount_factor=domain.discount_factor,
                lambda_=lambda_,
                initial_learn_rate=0.7422,
                learn_rate_decay_mode='boyan',
                boyan_N0=202.0,
            )
        else:
            raise NotImplementedError()

    run_cli(
        domain,
        agent_selector,
        default_max_steps=max_steps,
        default_num_policy_checks=20,
        default_checks_per_policy=100,
    )


if __name__ == '__main__':
    main()
