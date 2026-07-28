"""Persistent DiffAM guided by an online real-PPA surrogate."""
from __future__ import annotations

import copy
import logging
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from Appr_Comp.cellsolver import sim as diff_sim
from Appr_Comp.cellsolver.solver import GradientCellSolver

from .candidate import Candidate
from .diffam_search import (
    CellConfig,
    DiffAMStage2Search,
    FrozenDiffAMProblem,
    _balanced_quota,
    _config_distance,
    _config_key,
)
from .stage2_ppa_proxy import PPAProxyTrainer


_ROLES = ("area", "power", "knee")
_ROLE_WEIGHTS = {
    "area": (1.0, 0.10),
    "power": (0.10, 1.0),
    "knee": (0.50, 0.50),
}


class DiffAMProxyStage2Search(DiffAMStage2Search):
    """Stateful STE search whose PPA gradient comes from real DC labels."""

    def __init__(self, engine, config, run_dir: str, backbones: Sequence[Candidate]):
        super().__init__(engine, config, run_dir)
        self.backbones = list(backbones)
        self.proxy = PPAProxyTrainer(engine, backbones, config, self.device)
        self.persistent_logits: Dict[str, torch.Tensor] = {}
        self._pp12_cache: Dict[str, torch.Tensor] = {}

    def observe_and_fit(self, candidates: Sequence[Candidate]) -> dict:
        added = self.proxy.observe(candidates)
        result = self.proxy.fit()
        result["added"] = int(added)
        logging.info(
            "[stage2/diffam-proxy] proxy update: added=%d samples=%d "
            "ready=%s loss=%s",
            added,
            result.get("samples", 0),
            result.get("ready", False),
            (
                f"{float(result['loss']):.6f}"
                if result.get("loss") is not None
                else "n/a"
            ),
        )
        return result

    @staticmethod
    def _role(round_index: int, backbone_index: int, attempt: int) -> str:
        return _ROLES[(int(round_index) + int(backbone_index) + int(attempt)) % 3]

    def _persistent_key(self, context: FrozenDiffAMProblem, role: str) -> str:
        return f"{context.backbone.structure_hash}|{role}"

    def _train_proxy(
        self,
        context: FrozenDiffAMProblem,
        budget: float,
        seed: int,
        warm_start: Candidate | None,
        role: str,
    ) -> List[Tuple[CellConfig, dict]]:
        self.engine._activate_trunc_profile(context.backbone.k)
        solver = GradientCellSolver(
            self.engine,
            context.tree,
            context.pp_specs,
            budget,
            device=self.device,
            est=context.solver.est,
        )
        solver.est.rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        persistent_key = self._persistent_key(context, role)
        saved = self.persistent_logits.get(persistent_key)
        with torch.no_grad():
            if saved is not None and tuple(saved.shape) == tuple(solver.logits.shape):
                solver.logits.copy_(saved.to(self.device))
                solver.logits.add_(
                    torch.randn(
                        solver.logits.shape,
                        generator=generator,
                        device=self.device,
                    )
                    * float(self.cfg.stage2_proxy_logit_noise)
                )
            else:
                solver.logits.normal_(
                    mean=0.0,
                    std=float(self.cfg.stage2_diffam_init_std),
                    generator=generator,
                )
                solver.logits[:, 0] += float(self.cfg.stage2_diffam_exact_bias)
                if warm_start is not None:
                    warm_config = self._config_from_candidate(
                        warm_start, context.graph
                    )
                    rows = {
                        int(node): row
                        for row, (node, _t, _column) in enumerate(
                            solver.space.slots
                        )
                    }
                    for node, (_cell_kind, cell_type) in warm_config.items():
                        row = rows.get(int(node))
                        if (
                            row is not None
                            and int(cell_type) < solver.logits.shape[1]
                        ):
                            solver.logits[row, int(cell_type)] += float(
                                self.cfg.stage2_diffam_warm_bias
                            )

        optimizer = torch.optim.Adam(
            [solver.logits],
            lr=float(self.cfg.stage2_proxy_diffam_lr),
        )
        lam = float(self.cfg.stage2_diffam_lam0)
        proposals: List[Tuple[CellConfig, dict]] = []
        last_key = None
        steps = int(self.cfg.stage2_proxy_diffam_steps)
        dual_every = max(1, int(self.cfg.stage2_diffam_dual_every))
        role_weights = torch.tensor(
            _ROLE_WEIGHTS[role], dtype=torch.float32, device=self.device
        )
        for step in range(steps):
            tau = max(
                float(self.cfg.stage2_diffam_tau_min),
                float(self.cfg.stage2_proxy_tau_start)
                - (
                    float(self.cfg.stage2_proxy_tau_start)
                    - float(self.cfg.stage2_diffam_tau_min)
                )
                * step
                / max(steps - 1, 1),
            )
            selection = solver.weights(tau)
            selected_tables = solver.sel_dict(selection)
            ratio_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            for a, b, golden, weight in solver.est.train_batch():
                if a is solver.est.a12:
                    pp_bits = self._pp12_cache.get(
                        context.backbone.structure_hash
                    )
                    if pp_bits is None:
                        pp_bits = diff_sim.compute_pp_bits(
                            solver.pp_specs,
                            a,
                            b,
                            self.engine.bit_width,
                            self.device,
                        )
                        self._pp12_cache[
                            context.backbone.structure_hash
                        ] = pp_bits
                else:
                    pp_bits = diff_sim.compute_pp_bits(
                        solver.pp_specs,
                        a,
                        b,
                        self.engine.bit_width,
                        self.device,
                    )
                output = solver.tree.eval_diff(pp_bits, selected_tables)
                ratio_sum = ratio_sum + weight * solver.est._ratio_sum_diff(
                    output, golden
                )
            mred = ratio_sum / solver.est.n_rel
            nominal_area = solver.area_term(selection)
            entropy_probabilities = torch.softmax(
                (solver.logits + solver.mask) / max(tau, 1e-6), dim=-1
            )
            entropy = -(
                entropy_probabilities
                * torch.log(entropy_probabilities.clamp_min(1e-12))
            ).sum(dim=-1).mean()

            if self.proxy.ready:
                features = self.proxy.encoder.selection_features(
                    context.backbone,
                    context.graph,
                    solver.space.slots,
                    selection,
                )
                predicted, uncertainty = self.proxy.predict_features(features)
                predicted = predicted[0]
                uncertainty = uncertainty[0]
                ppa_objective = (
                    role_weights @ predicted[:2]
                    + float(self.cfg.stage2_proxy_uncertainty_weight)
                    * (role_weights @ uncertainty[:2])
                    + float(self.cfg.stage2_proxy_nominal_area_weight)
                    * nominal_area.to(torch.float32)
                )
                anchor_delay = float(
                    self.proxy.anchors[context.backbone.structure_hash][2]
                )
                predicted_delay = anchor_delay * torch.exp(predicted[2])
                delay_penalty = float(self.cfg.stage2_proxy_delay_weight) * torch.relu(
                    predicted_delay / float(self.cfg.delay_limit) - 1.0
                )
            else:
                ppa_objective = nominal_area.to(torch.float32)
                delay_penalty = torch.zeros(
                    (), dtype=torch.float32, device=self.device
                )
            loss = (
                ppa_objective
                + delay_penalty
                + lam * torch.relu(mred / budget - 1.0).to(torch.float32)
                - float(self.cfg.stage2_proxy_entropy_weight) * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([solver.logits], 5.0)
            optimizer.step()

            config = solver.hard_config()
            key = _config_key(config)
            if key != last_key:
                proposals.append(
                    (
                        dict(config),
                        {
                            "source": "trajectory",
                            "step": step + 1,
                            "tau": float(tau),
                            "train_mred": float(mred.detach()),
                            "train_area_fraction": float(nominal_area.detach()),
                            "proxy_ready": bool(self.proxy.ready),
                            "proxy_role": role,
                        },
                    )
                )
                last_key = key
            if (step + 1) % dual_every == 0 or step + 1 == steps:
                if step + 1 == steps:
                    self._pp12_cache.pop(
                        context.backbone.structure_hash, None
                    )
                hard_mred = solver.gate_mred(config)
                lam = max(
                    5.0,
                    lam
                    + float(self.cfg.stage2_diffam_lam_step)
                    * (hard_mred / budget - 1.0),
                )

        self.persistent_logits[persistent_key] = solver.logits.detach().cpu()
        proposals.append(
            (
                dict(solver.hard_config()),
                {
                    "source": "hard_final",
                    "step": steps,
                    "proxy_ready": bool(self.proxy.ready),
                    "proxy_role": role,
                },
            )
        )
        with torch.no_grad():
            masked = solver.logits + solver.mask
            for temperature in (0.35, 0.60, 1.0, 1.6):
                probabilities = torch.softmax(masked / temperature, dim=-1)
                for sample_id in range(int(self.cfg.stage2_diffam_samples)):
                    config: CellConfig = {}
                    for row, (node, cell_kind, _column) in enumerate(
                        solver.space.slots
                    ):
                        cell_type = int(
                            torch.multinomial(
                                probabilities[row],
                                1,
                                generator=generator,
                            ).item()
                        )
                        if cell_type:
                            config[int(node)] = (int(cell_kind), cell_type)
                    proposals.append(
                        (
                            config,
                            {
                                "source": "logit_sample",
                                "temperature": float(temperature),
                                "sample_id": sample_id,
                                "proxy_ready": bool(self.proxy.ready),
                                "proxy_role": role,
                            },
                        )
                    )
        for _config, metadata in proposals:
            metadata.update({"target_mred": float(budget), "seed": int(seed)})
        return proposals

    def _proxy_score(
        self,
        context: FrozenDiffAMProblem,
        candidate: Candidate,
        role: str,
    ) -> Tuple[float, dict]:
        if not self.proxy.ready:
            config = self._config_from_candidate(candidate, context.graph)
            saving = float(context.solver.area_saving(config))
            return -saving, {"proxy_area_saving": saving}
        ppa, uncertainty = self.proxy.predict_candidate(candidate)
        weights = np.asarray(_ROLE_WEIGHTS[role], dtype=np.float64)
        anchor = np.asarray(
            self.proxy.anchors[context.backbone.structure_hash],
            dtype=np.float64,
        )
        log_ratio = np.log(np.maximum(ppa[:2], 1e-15) / anchor[:2])
        score = float(
            weights @ log_ratio
            + float(self.cfg.stage2_proxy_uncertainty_weight)
            * (weights @ uncertainty[:2])
        )
        metadata = {
            "predicted_area": float(ppa[0]),
            "predicted_power": float(ppa[1]),
            "predicted_delay": float(ppa[2]),
            "proxy_uncertainty": [float(value) for value in uncertainty],
            "proxy_score": score,
        }
        return score, metadata

    def _select_proxy(
        self,
        context: FrozenDiffAMProblem,
        raw: Iterable[Tuple[CellConfig, dict]],
        budget: float,
        size: int,
        excluded_hashes: set[str],
        role: str,
    ) -> List[Candidate]:
        unique = {}
        for config, metadata in [({}, {"source": "exact"}), *list(raw)]:
            unique.setdefault(
                _config_key(config), (dict(config), copy.deepcopy(metadata))
            )
        exact_full = context.solver.gate_mred({})
        exact_screen = context.solver.gate_screen({})
        screen_offset = exact_full - exact_screen
        screened = []
        for config, metadata in unique.values():
            candidate = self._candidate_from_config(context, config, metadata)
            if candidate.cell_hash in excluded_hashes:
                continue
            screen_mred = context.solver.gate_screen(config) + screen_offset
            score, proxy_metadata = self._proxy_score(context, candidate, role)
            candidate.metadata.update(proxy_metadata)
            screened.append(
                (config, metadata, candidate, float(screen_mred), float(score))
            )

        screened.sort(
            key=lambda item: (
                0 if item[3] <= budget * 1.10 else 1,
                item[4] if item[3] <= budget * 1.10 else item[3],
                abs(item[3] - budget),
                item[2].cell_hash,
            )
        )
        shortlist = screened[:size]
        gated = []
        for config, _metadata, candidate, screen_mred, score in shortlist:
            gate_mred = context.solver.gate_mred(config)
            candidate.metadata.update(
                {
                    "method": "diffam_proxy",
                    "proxy_role": role,
                    "proxy_screen_mred": float(screen_mred),
                    "proxy_mred": float(gate_mred),
                }
            )
            gated.append((config, candidate, score))
        gated.sort(
            key=lambda item: (
                0
                if float(item[1].metadata["proxy_mred"]) <= budget
                else 1,
                item[2]
                if float(item[1].metadata["proxy_mred"]) <= budget
                else float(item[1].metadata["proxy_mred"]),
                item[1].cell_hash,
            )
        )
        selected: List[Tuple[CellConfig, Candidate, float]] = []
        remaining = list(gated)
        while remaining and len(selected) < size:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            position = max(
                range(len(remaining)),
                key=lambda index: (
                    min(
                        _config_distance(remaining[index][0], old[0])
                        for old in selected
                    ),
                    -remaining[index][2],
                ),
            )
            selected.append(remaining.pop(position))
        output = [candidate for _config, candidate, _score in selected]
        context.solver.clear_gate_pp()
        return output

    def propose(
        self,
        backbones: Sequence[Candidate],
        *,
        size: int,
        round_index: int,
        excluded_hashes: Iterable[str] = (),
        warm_starts: Sequence[Candidate] = (),
    ) -> List[Candidate]:
        unique_backbones = {}
        for backbone in backbones:
            unique_backbones.setdefault(backbone.structure_hash, backbone)
        backbone_list = list(unique_backbones.values())
        if not backbone_list:
            raise ValueError("proxy DiffAM requires at least one backbone")
        selected: List[Candidate] = []
        selected_hashes = set(excluded_hashes)
        warm_by_structure: Dict[str, List[Candidate]] = {}
        for candidate in warm_starts:
            warm_by_structure.setdefault(candidate.structure_hash, []).append(
                candidate
            )
        attempts = 0
        maximum_attempts = max(4, int(self.cfg.stage2_diffam_restarts) + 3)
        while len(selected) < size and attempts < maximum_attempts:
            progress = False
            for backbone_index, backbone in enumerate(backbone_list):
                if len(selected) >= size:
                    break
                remaining = size - len(selected)
                quota = _balanced_quota(
                    remaining, len(backbone_list), backbone_index
                )
                context = self._build_context(backbone)
                band = int(backbone.metadata.get("mred_band", backbone_index))
                budget_index = (
                    band + int(round_index) + attempts
                ) % len(self.budgets)
                budget = float(self.budgets[budget_index])
                role = self._role(round_index, backbone_index, attempts)
                seed = (
                    int(self.cfg.seed)
                    + int(round_index) * 1_000_003
                    + attempts * 10_007
                    + int(backbone.structure_hash[:8], 16)
                )
                warm_pool = warm_by_structure.get(backbone.structure_hash, [])
                warm = warm_pool[attempts % len(warm_pool)] if warm_pool else None
                logging.info(
                    "[stage2/diffam-proxy] round=%d backbone=%s role=%s "
                    "budget=%.3e attempt=%d quota=%d proxy=%s",
                    round_index,
                    backbone.candidate_id,
                    role,
                    budget,
                    attempts + 1,
                    quota,
                    self.proxy.ready,
                )
                raw = self._train_proxy(context, budget, seed, warm, role)
                candidates = self._select_proxy(
                    context,
                    raw,
                    budget,
                    min(quota, remaining),
                    selected_hashes,
                    role,
                )
                for candidate in candidates:
                    if candidate.cell_hash in selected_hashes:
                        continue
                    candidate.metadata.update(
                        {
                            "diffam_round": int(round_index),
                            "diffam_attempt": int(attempts),
                            "generation": int(round_index),
                        }
                    )
                    selected.append(candidate)
                    selected_hashes.add(candidate.cell_hash)
                    progress = True
                    if len(selected) >= size:
                        break
            if not progress:
                break
            attempts += 1
        if len(selected) != size:
            raise RuntimeError(
                f"proxy DiffAM generated only {len(selected)}/{size} unseen candidates"
            )
        return selected

    def state_dict(self) -> dict:
        return {
            "persistent_logits": {
                key: value.detach().cpu()
                for key, value in self.persistent_logits.items()
            },
            "proxy": self.proxy.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.persistent_logits = {
            str(key): value.detach().cpu()
            for key, value in (state.get("persistent_logits") or {}).items()
        }
        if state.get("proxy") is not None:
            self.proxy.load_state_dict(state["proxy"])
