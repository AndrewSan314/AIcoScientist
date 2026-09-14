"""Domain-informed process dependency graph; an architecture prior, not causality."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

from .stages import ProcessStage


@dataclass(frozen=True)
class ProcessDependencyGraph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    stage_map: Mapping[str, ProcessStage]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.nodes or len(set(self.nodes)) != len(self.nodes) or any(not node.strip() for node in self.nodes):
            raise ValueError("process graph requires unique non-empty nodes")
        if any(source not in self.nodes or target not in self.nodes for source, target in self.edges):
            raise ValueError("process graph edge references an unknown node")
        if set(self.stage_map) - set(self.nodes) or any(not isinstance(stage, ProcessStage) for stage in self.stage_map.values()):
            raise ValueError("process graph stage_map must bind known nodes to ProcessStage values")
        if self._has_cycle():
            raise ValueError("process dependency graph must be acyclic")
        payload = {"nodes": self.nodes, "edges": self.edges, "stage_map": {node: stage.value for node, stage in sorted(self.stage_map.items())}}
        object.__setattr__(self, "fingerprint", hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProcessDependencyGraph":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list) or not isinstance(raw.get("edges"), list) or not isinstance(raw.get("stage_map"), dict):
            raise ValueError("process graph YAML requires nodes, edges, and stage_map mappings")
        if any(not isinstance(edge, list) or len(edge) != 2 for edge in raw["edges"]):
            raise ValueError("process graph edges must be [source, target] pairs")
        return cls(tuple(str(node) for node in raw["nodes"]), tuple((str(edge[0]), str(edge[1])) for edge in raw["edges"]), {str(node): ProcessStage(value) for node, value in raw["stage_map"].items()})

    def legal_successor_stages(self, stage: ProcessStage) -> tuple[ProcessStage, ...]:
        starts = {node for node, mapped in self.stage_map.items() if mapped == stage}
        reachable = self._reachable(starts)
        return tuple(sorted({self.stage_map[node] for node in reachable if node in self.stage_map and self.stage_map[node] != stage}, key=lambda item: list(ProcessStage).index(item)))

    def legal_predecessor_stages(self, stage: ProcessStage) -> tuple[ProcessStage, ...]:
        return tuple(item for item in ProcessStage if stage in self.legal_successor_stages(item))

    def validate_transition(self, source: ProcessStage, target: ProcessStage) -> None:
        if target not in self.legal_successor_stages(source):
            raise ValueError(f"process graph forbids transition {source.value} -> {target.value}")

    def path_attribution(self, source_node: str, target_node: str) -> tuple[str, ...]:
        if source_node not in self.nodes or target_node not in self.nodes:
            raise KeyError("process graph attribution requires known nodes")
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source_node, (source_node,))])
        while queue:
            node, path = queue.popleft()
            if node == target_node:
                return path
            queue.extend((child, (*path, child)) for parent, child in self.edges if parent == node and child not in path)
        raise ValueError(f"no directed process-graph path from {source_node} to {target_node}")

    def provenance(self) -> dict[str, str]:
        return {"stage_order_source": "PROCESS_DEPENDENCY_GRAPH", "process_graph_fingerprint": self.fingerprint}

    def _reachable(self, starts: set[str]) -> set[str]:
        seen, queue = set(starts), deque(starts)
        while queue:
            node = queue.popleft()
            for source, target in self.edges:
                if source == node and target not in seen:
                    seen.add(target); queue.append(target)
        return seen - starts

    def _has_cycle(self) -> bool:
        remaining = set(self.nodes)
        while remaining:
            roots = {node for node in remaining if not any(target == node and source in remaining for source, target in self.edges)}
            if not roots:
                return True
            remaining -= roots
        return False
