"""Human extensions for pinned WBC/mjlab; importing this package runs no simulation.

Heavy dependencies are loaded only by the explicit configuration/registration API.
Unqualified bundles may construct evaluation fixtures, never training fixtures.
"""


def build_human_env_cfg(bundle, num_envs=1, evaluation=True):
    from .task import build_human_env_cfg as build

    return build(bundle, num_envs=num_envs, evaluation=evaluation)


def register_human_task(bundle, num_envs=8192, evaluation=False):
    from .task import register_human_task as register

    return register(bundle, num_envs=num_envs, evaluation=evaluation)


def get_runner_class():
    from .runner import HumanMotionTrackingRunner

    return HumanMotionTrackingRunner
