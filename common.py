import click
from rlpy.Agents.Agent import Agent
from rlpy.Domains.Domain import Domain
from rlpy.Experiments import Experiment
from typing import Callable, Optional


def run_cli(
        domain: Domain,
        agent_selector: Callable[[str, int], Agent],
        default_max_steps: int = 1000,
        default_num_policy_checks: int = 10,
        default_checks_per_policy: int = 10,
        **kwargs
) -> None:
    @click.group()
    @click.option('--agent', type=str, default=None, help='The name of agent you want to run')
    @click.option('--seed', type=int, default=1, help='The problem to learn')
    @click.option('--max-steps', type=int, default=default_max_steps,
                  help='Total number of interactions')
    @click.option('--num-policy-checks', type=int, default=default_num_policy_checks,
                  help='Total number of evaluation time')
    @click.option('--checks-per-policy', type=int, default=default_checks_per_policy,
                  help='Number of evaluation per 1 evaluation time')
    @click.option('--log-interval', type=int, default=10, help='Number of seconds')
    @click.option('--log-dir', type=str, default='Results/Temp',
                  help='The directory to be used for storing the logs')
    @click.pass_context
    def experiment(
            ctx: dict,
            agent: Optional[str],
            seed: int,
            max_steps: int,
            num_policy_checks: int,
            checks_per_policy: int,
            log_interval: int,
            log_dir: str,
    ) -> None:
        agent = agent_selector(agent, seed)
        ctx.obj['experiment'] = Experiment(
            agent,
            domain,
            exp_id=seed,
            max_steps=max_steps,
            num_policy_checks=num_policy_checks,
            checks_per_policy=checks_per_policy,
            log_interval=log_interval,
            log_dir=log_dir,
            **kwargs
        )

    @experiment.command(help='Train the agent')
    @click.option('--visualize-performance', default=0, type=int,
                  help='The number of visualization steps during performance runs')
    @click.option('--visualize-learning', is_flag=True,
                  help='Visualize of the learning status before each evaluation')
    @click.option('--visualize-steps', is_flag=True,
                  help='Visualize all steps during learning')
    @click.pass_context
    def train(
            ctx: dict,
            visualize_performance: int,
            visualize_learning: bool,
            visualize_steps: bool,
    ) -> None:
        exp = ctx.obj['experiment']
        exp.run(visualize_performance, visualize_learning, visualize_steps)

    return experiment(obj={})
