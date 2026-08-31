from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import pandas as pd

from src.optimization.objective import OptimizationObjective
from src.optimization.proposal import CandidateProposal


class UnsupportedStrategyError(ValueError):
    """Raised when an unrecognized or retired optimization strategy is requested."""


class AcquisitionEvaluationError(RuntimeError):
    """Raised when evaluation of an acquisition function fails on the optimization backend."""


SUPPORTED_STRATEGIES: tuple[str, ...] = (
    "random",
    "greedy",
    "gp_ucb",
    "expected_improvement",
    "noisy_expected_improvement",
    "thompson",
)

STRATEGY_ALIASES: dict[str, str] = {
    "uniform": "random",
    "uniform_random": "random",
    "posterior_mean": "greedy",
    "ucb": "gp_ucb",
    "upper_confidence_bound": "gp_ucb",
    "ei": "expected_improvement",
    "log_ei": "expected_improvement",
    "log_expected_improvement": "expected_improvement",
    "nei": "noisy_expected_improvement",
    "log_nei": "noisy_expected_improvement",
    "log_noisy_expected_improvement": "noisy_expected_improvement",
    "q_nei": "noisy_expected_improvement",
    "ts": "thompson",
    "thompson_sampling": "thompson",
}

RETIRED_STRATEGIES: set[str] = {
    "turbo",
    "turbo_ei",
    "turbo_nei",
    "adaptive",
    "qnehvi",
}


def resolve_strategy(strategy: str) -> str:
    """Validates and maps a requested strategy name or alias to its canonical name.

    Raises:
        UnsupportedStrategyError: If strategy is retired (e.g. TuRBO) or unrecognized.
    """
    key = strategy.strip().lower()
    if key in RETIRED_STRATEGIES:
        raise UnsupportedStrategyError(
            f"Strategy {strategy!r} is retired and not part of the production BoTorch backend. "
            "Use 'noisy_expected_improvement' (or 'nei') for global noisy BO or explicitly use a legacy reference benchmark."
        )
    if key in STRATEGY_ALIASES:
        return STRATEGY_ALIASES[key]
    if key in SUPPORTED_STRATEGIES:
        return key
    raise UnsupportedStrategyError(
        f"Optimization strategy {strategy!r} is not recognized. "
        f"Supported canonical strategies: {list(SUPPORTED_STRATEGIES)}. "
        f"Supported aliases: {sorted(STRATEGY_ALIASES.keys())}."
    )


@runtime_checkable
class OptimizerBackend(Protocol):
    """Protocol defining the clean boundary between AIcoScientist and generic optimization engines.

    The science layer interacts exclusively through this protocol and has zero direct
    knowledge of PyTorch tensors, GPyTorch modules, BoTorch acquisitions, or MLL fitting internals.
    """

    @property
    def name(self) -> str:
        """Returns the canonical backend name (e.g., 'botorch')."""
        ...

    @property
    def version(self) -> str:
        """Returns the backend library version."""
        ...

    def propose(
        self,
        observations: pd.DataFrame | Sequence[Mapping[str, Any]],
        candidate_pool: pd.DataFrame,
        objective: OptimizationObjective | str,
        *,
        feature_columns: Sequence[str] | None = None,
        candidate_id_column: str | None = None,
        n: int = 1,
        seed: int | None = None,
        strategy: str | None = None,
        **kwargs: Any,
    ) -> list[CandidateProposal]:
        """Evaluates surrogate model and acquisition function over candidate pool to propose next experiments.

        Parameters
        ----------
        observations:
            Observed experimental data points with feature coordinates and target values.
        candidate_pool:
            Universe of candidate design points.
        objective:
            Optimization objective (target name, minimization/maximization sense, constraints).
        feature_columns:
            Optional explicit feature column names. Inferred from candidate_pool if omitted.
        candidate_id_column:
            Optional column identifying candidates. Inferred if omitted.
        n:
            Number of candidate proposals to return (default 1).
        seed:
            Random seed for stochastic reproducibility.
        strategy:
            Acquisition strategy name (e.g. 'random', 'greedy', 'gp_ucb', 'ei', 'nei', 'thompson').
        kwargs:
            Optional strategy hyperparameters (e.g. beta for UCB, n_fantasies for NEI).

        Returns
        -------
        list[CandidateProposal]:
            Ranked list of candidate proposals with predicted means, uncertainties, and acquisition scores.
        """
        ...

    def score_candidates(
        self,
        observations: pd.DataFrame | Sequence[Mapping[str, Any]],
        candidate_pool: pd.DataFrame,
        objective: OptimizationObjective | str,
        *,
        feature_columns: Sequence[str] | None = None,
        candidate_id_column: str | None = None,
        seed: int | None = None,
        strategy: str | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluates acquisition scores for candidate design points in the candidate pool.

        Parameters
        ----------
        observations:
            Observed experimental data points with feature coordinates and target values.
        candidate_pool:
            Universe of candidate design points.
        objective:
            Optimization objective (target name, minimization/maximization sense).
        feature_columns:
            Optional explicit feature column names.
        candidate_id_column:
            Optional column identifying candidates.
        seed:
            Random seed for stochastic reproducibility.
        strategy:
            Acquisition strategy name (e.g. 'expected_improvement', 'noisy_expected_improvement', 'gp_ucb').
        kwargs:
            Optional strategy hyperparameters.

        Returns
        -------
        dict[str, float]:
            Mapping from candidate_id to raw acquisition score evaluated by the backend.
        """
        ...
