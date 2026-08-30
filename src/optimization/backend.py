from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import pandas as pd

from src.optimization.objective import OptimizationObjective
from src.optimization.proposal import CandidateProposal


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
