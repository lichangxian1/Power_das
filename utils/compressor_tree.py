import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from collections import deque
from typing import Tuple, List, Dict, Any, Callable
import logging
import time
import json
import torch
import torch.utils
import torch.utils.tensorboard
from abc import ABC, abstractmethod


from . import adder_utils
from .common import lse_gamma

try:
    import pulp
except ImportError:
    print(
        "pulp is not installed. Please install it using 'pip install pulp' to use MIP methods."
    )


def get_target_delay(bit_width):
    if bit_width == 8:
        return [50, 250, 400, 650]
    elif bit_width == 16:
        return [50, 200, 500, 1200]
    elif bit_width == 32:
        return [50, 300, 600, 2000]
    else:
        return [50, 600, 1500, 3000]


def get_full_target_delay(bit_width):
    if bit_width == 8:
        return list(range(50, 1000, 10))
    elif bit_width == 16:
        return list(range(50, 2000, 20))
    elif bit_width == 32:
        return list(range(50, 3000, 20))
    else:
        return list(range(50, 4000, 20))


def get_initial_partial_product(bit_width: int, encode_type: str) -> np.ndarray:
    if encode_type == "and":
        pp = np.zeros([bit_width * 2 - 1])
        for i in range(0, bit_width):
            pp[i] = i + 1
        for i in range(bit_width, bit_width * 2 - 1):
            pp[i] = bit_width * 2 - 1 - i
    elif encode_type == "booth":
        pp = np.zeros([bit_width * 2])
        for column_index in range(bit_width):
            if column_index % 2 == 0:
                pp[column_index] = 2 + column_index // 2
            else:
                pp[column_index] = 1 + column_index // 2
        pp[bit_width] = 1 + bit_width // 2
        pp[bit_width + 1] = 1 + bit_width // 2
        pp[bit_width + 2] = 1 + bit_width // 2
        pp[bit_width + 3] = 1 + bit_width // 2
        for column_index in range(bit_width + 4, bit_width * 2):
            if column_index % 2 == 0:
                pp[column_index] = bit_width // 2 - (column_index - bit_width - 4) // 2
            else:
                pp[column_index] = (
                    bit_width // 2 - (column_index - bit_width - 4) // 2 - 1
                )
    elif encode_type == "and_mac":
        pp = np.zeros([bit_width * 2 - 1])
        for i in range(0, bit_width):
            pp[i] = i + 1
        for i in range(bit_width, bit_width * 2 - 1):
            pp[i] = bit_width * 2 - 1 - i
        for i in range(0, bit_width):
            pp[i] += 1
    elif encode_type == "booth_mac":
        pp = np.zeros([bit_width * 2])
        for column_index in range(bit_width):
            if column_index % 2 == 0:
                pp[column_index] = 2 + column_index // 2
            else:
                pp[column_index] = 1 + column_index // 2
        pp[bit_width] = 1 + bit_width // 2
        pp[bit_width + 1] = 1 + bit_width // 2
        pp[bit_width + 2] = 1 + bit_width // 2
        pp[bit_width + 3] = 1 + bit_width // 2
        for column_index in range(bit_width + 4, bit_width * 2):
            if column_index % 2 == 0:
                pp[column_index] = bit_width // 2 - (column_index - bit_width - 4) // 2
            else:
                pp[column_index] = (
                    bit_width // 2 - (column_index - bit_width - 4) // 2 - 1
                )
        for i in range(0, bit_width):
            pp[i] += 1
    else:
        raise NotImplementedError

    return pp.astype(int)


class CompressorTree(ABC):
    def __init__(self, pp=None, ct32=None, ct22=None):
        self.pp: np.ndarray = pp
        self.ct32: np.ndarray = ct32
        self.ct22: np.ndarray = ct22

    @classmethod
    def from_dict(cls, d):
        ct32 = np.asarray(d["ct32"]).astype(int)
        ct22 = np.asarray(d["ct22"]).astype(int)
        pp = np.asarray(d["pp"]).astype(int)
        return cls(pp, ct32, ct22)

    @classmethod
    def wallace(cls, pp):
        max_stage_num = len(pp)
        stage_num = 0

        sequence_pp = np.zeros([1, len(pp)])
        sequence_pp[0] = copy.deepcopy(pp)
        ct32 = np.zeros([1, len(pp)])
        ct22 = np.zeros([1, len(pp)])
        target = np.asarray([2 for i in range(len(pp))])

        while stage_num < max_stage_num:
            for i in range(0, len(pp)):
                if sequence_pp[stage_num][i] % 3 == 0:
                    ct32[stage_num][i] = sequence_pp[stage_num][i] // 3
                    ct22[stage_num][i] = 0
                elif sequence_pp[stage_num][i] % 3 == 1:
                    ct32[stage_num][i] = sequence_pp[stage_num][i] // 3
                    ct22[stage_num][i] = 0
                elif sequence_pp[stage_num][i] % 3 == 2:
                    ct32[stage_num][i] = sequence_pp[stage_num][i] // 3
                    if stage_num == 0:
                        ct22[stage_num][i] = 0
                    else:
                        ct22[stage_num][i] = 1
            sequence_pp = np.r_[sequence_pp, np.zeros([1, len(pp)])]
            sequence_pp[stage_num + 1][0] = (
                sequence_pp[stage_num][0] - ct32[stage_num][0] * 2 - ct22[stage_num][0]
            )
            for i in range(1, len(pp)):
                sequence_pp[stage_num + 1][i] = (
                    sequence_pp[stage_num][i]
                    + ct32[stage_num][i - 1]
                    + ct22[stage_num][i - 1]
                    - ct32[stage_num][i] * 2
                    - ct22[stage_num][i]
                )
            stage_num += 1
            if (sequence_pp[stage_num] <= target).all():
                break

            ct32 = np.r_[ct32, np.zeros([1, len(pp)])]
            ct22 = np.r_[ct22, np.zeros([1, len(pp)])]

        assert (
            stage_num < max_stage_num
        ), "Exceed max stage num! Set max_stage_num larger"
        ct32 = np.sum(ct32, axis=0).astype(int)
        ct22 = np.sum(ct22, axis=0).astype(int)

        compressor_tree = cls(pp, ct32, ct22)
        return compressor_tree

    @classmethod
    def dadda(cls, pp):
        max_stage_num = len(pp)
        d = []
        d_j = 2
        for j in range(max_stage_num):
            d.append(d_j)
            d_j = int(np.floor(1.5 * d_j))

        remain_pp = copy.deepcopy(pp)

        ct32 = np.zeros(len(pp)).astype(int)
        ct22 = np.zeros(len(pp)).astype(int)

        for j in range(max_stage_num - 1, 0 - 1, -1):
            d_j = d[j]
            i = 0

            while i <= len(remain_pp) - 1:
                if remain_pp[i] <= d_j:
                    i += 1
                    continue
                elif remain_pp[i] == d_j + 1:
                    ct22[i] += 1
                    if i + 1 < len(remain_pp):
                        remain_pp[i + 1] += 1
                    remain_pp[i] -= 1
                    i += 1
                    continue
                else:
                    ct32[i] += 1
                    if i + 1 < len(remain_pp):
                        remain_pp[i + 1] += 1
                    remain_pp[i] -= 2
                    continue
        compressor_tree = cls(pp, ct32, ct22)
        return compressor_tree

    @classmethod
    def gomil(
        cls,
        pp,
        max_stage_num,
        method,
        alpha,
        beta,
        timeLimit,
        n_processing,
    ):
        """
        Replication from paper
            GOMIL: Global Optimization of Multiplier by Integer Linear Programming
            https://umji.sjtu.edu.cn/~wkqian/papers/Xiao_Qian_Liu_GOMIL_Global_Optimization_of_Multiplier_by_Integer_Linear_Programming.pdf
        Solver:
            Pulp framework, use GUROBI_CMD default.
        alpha and beta: area of ct32 and ct22( defaults are extracted from NangateOpenCellLibrary_typical.lib)
        """
        logging.info("Using MIP method for ct_architecture")
        start_time = time.time()
        model = pulp.LpProblem(f"gomil-{len(pp)}", pulp.LpMinimize)
        num_cols = len(pp)

        f = pulp.LpVariable.dicts(
            "f",
            [(i, j) for i in range(max_stage_num) for j in range(num_cols)],
            lowBound=0,
            cat="Integer",
        )
        h = pulp.LpVariable.dicts(
            "h",
            [(i, j) for i in range(max_stage_num) for j in range(num_cols)],
            lowBound=0,
            cat="Integer",
        )
        V = pulp.LpVariable.dicts(
            "V",
            [(i, j) for i in range(max_stage_num) for j in range(num_cols)],
            lowBound=0,
            cat="Integer",
        )

        F = pulp.LpVariable("F", lowBound=0, cat="Integer")
        H = pulp.LpVariable("H", lowBound=0, cat="Integer")

        model += alpha * F + beta * H

        model += (
            pulp.lpSum(f[(i, j)] for i in range(max_stage_num) for j in range(num_cols))
            == F
        )
        model += (
            pulp.lpSum(h[(i, j)] for i in range(max_stage_num) for j in range(num_cols))
            == H
        )

        for i in range(max_stage_num):
            for j in range(num_cols):
                model += 3 * f[(i, j)] + 2 * h[(i, j)] <= V[(i, j)]
        for j in range(num_cols):
            model += V[(0, j)] == pp[j]
        for i in range(1, max_stage_num):
            model += V[(i, 0)] == V[(i - 1, 0)] - 2 * f[(i - 1, 0)] - h[(i - 1, 0)]
            for j in range(1, num_cols):
                model += (
                    V[(i, j)]
                    == V[(i - 1, j)]
                    - 2 * f[(i - 1, j)]
                    - h[(i - 1, j)]
                    + f[(i - 1, j - 1)]
                    + h[(i - 1, j - 1)]
                )
        for j in range(num_cols):
            model += V[(max_stage_num - 1, j)] <= 2
            model += V[(max_stage_num - 1, j)] >= 1

        solver: pulp.LpSolver_CMD = getattr(pulp, method)(
            timeLimit=timeLimit,
            keepFiles=False,
            msg=True,
            options=[("Threads", str(n_processing))],
        )

        model.solve(solver)
        logging.info("Solver Status:", pulp.LpStatus[model.status])
        ct_32 = np.zeros_like(pp)
        ct_22 = np.zeros_like(pp)
        for j in range(num_cols):
            for i in range(max_stage_num):
                ct_32[j] += int(pulp.value(f[(i, j)]))
                ct_22[j] += int(pulp.value(h[(i, j)]))
        V_result = np.zeros((max_stage_num, num_cols), dtype=int)
        for i in range(max_stage_num):
            for j in range(num_cols):
                V_result[i, j] = int(pulp.value(V[(i, j)]))
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

        compressor_tree = cls(pp, ct_32, ct_22)
        return compressor_tree

    @classmethod
    def ufomac(cls, pp):
        pp = pp.astype(int)
        ct32 = np.zeros_like(pp).astype(int)
        ct22 = np.zeros_like(pp).astype(int)
        carry_num = 0
        for column_idx in range(len(pp)):
            num_ports = pp[column_idx] + carry_num
            if num_ports >= 3:
                if num_ports % 2 == 0:
                    ct32[column_idx] = (num_ports - 2) // 2
                else:
                    ct32[column_idx] = (num_ports - 3) // 2
                    ct22[column_idx] = 1
                carry_num = ct32[column_idx] + ct22[column_idx]
        ct = cls(pp, ct32, ct22)
        return ct

    def to_dict(self):
        return {
            "ct32": self.ct32.tolist(),
            "ct22": self.ct22.tolist(),
        }

    def compressor_assignment(self):
        assert len(self.pp) == len(self.ct32) and len(self.pp) == len(self.ct22)
        column_len = len(self.pp)
        ct32_remain = copy.deepcopy(self.ct32)
        ct22_remain = copy.deepcopy(self.ct22)
        current_stage = 0
        ct32_decomposed = np.zeros([1, len(self.ct32)]).astype(int)
        ct22_decomposed = np.zeros([1, len(self.ct22)]).astype(int)

        current_pp = copy.deepcopy(self.pp)
        while (ct32_remain > 0).any() or (ct22_remain > 0).any():
            next_pp = np.zeros_like(current_pp)
            for column_index in range(column_len):
                current_pp_height = int(current_pp[column_index])
                assigned_ct32 = min(
                    int(ct32_remain[column_index]),
                    current_pp_height // 3,
                )
                ct32_remain[column_index] -= assigned_ct32
                current_pp_height -= assigned_ct32 * 3
                assigned_ct22 = min(
                    int(ct22_remain[column_index]),
                    current_pp_height // 2,
                )
                ct22_remain[column_index] -= assigned_ct22
                current_pp_height -= assigned_ct22 * 2
                next_pp[column_index] += (
                    current_pp_height + assigned_ct32 + assigned_ct22
                )
                if column_index < column_len - 1:
                    next_pp[column_index + 1] += assigned_ct32 + assigned_ct22

                ct32_decomposed[current_stage][column_index] = assigned_ct32
                ct22_decomposed[current_stage][column_index] = assigned_ct22
            current_pp = next_pp
            ct32_decomposed = np.r_[ct32_decomposed, np.zeros([1, len(self.ct32)])]
            ct22_decomposed = np.r_[ct22_decomposed, np.zeros([1, len(self.ct22)])]
            current_stage += 1
        ct32_decomposed = ct32_decomposed.astype(int)
        ct22_decomposed = ct22_decomposed.astype(int)
        if np.sum(ct32_decomposed[-1, :]) == 0 and np.sum(ct22_decomposed[-1, :]) == 0:
            ct32_decomposed = ct32_decomposed[:-1]
            ct22_decomposed = ct22_decomposed[:-1]
        return ct32_decomposed, ct22_decomposed

    def compressor_assignment_fused(self):
        dec_ct32, dec_ct22 = self.compressor_assignment()
        assignment = []
        stage_num, col_num = dec_ct32.shape
        ct32_counter = np.zeros_like(self.pp, dtype=int)
        ct22_counter = np.zeros_like(self.pp, dtype=int)

        for s in range(stage_num):
            stage_assignment = []
            for c in range(col_num):
                column_assignment = []
                for ct32_idx in range(dec_ct32[s, c]):
                    compressor_info = (s, c, 0, ct32_counter[c])
                    ct32_counter[c] += 1
                    column_assignment.append(compressor_info)
                for ct22_idx in range(dec_ct22[s, c]):
                    compressor_info = (s, c, 1, ct22_counter[c])
                    ct22_counter[c] += 1
                    column_assignment.append(compressor_info)
                stage_assignment.append(column_assignment)
            assignment.append(stage_assignment)
        assert (ct32_counter == self.ct32).all() and (ct22_counter == self.ct22).all()
        return assignment

    def compressor_assignment_ufomac(
        self,
        max_stage_num,
        M,
        method,
        n_processing,
        timeLimit,
        keepFiles,
    ):
        """
        Replication method from paper
            UFO-MAC: A Unified Framework for Optimization of High-Performance Multipliers and Multiply-Accumulators
            https://arxiv.org/pdf/2408.06935
        Solver:
            Pulp framework, use GUROBI_CMD default.
        """
        logging.info("Using MIP method for compressor_assignment")
        start_time = time.time()
        model = pulp.LpProblem(f"compressor_assignment_{len(self.pp)}", pulp.LpMinimize)

        num_cols = len(self.pp)
        F = self.ct32
        H = self.ct22

        f = pulp.LpVariable.dicts(
            "f",
            [(i, j) for i in range(max_stage_num) for j in range(num_cols)],
            lowBound=0,
            cat="Integer",
        )
        h = pulp.LpVariable.dicts(
            "h",
            [(i, j) for i in range(max_stage_num) for j in range(num_cols)],
            lowBound=0,
            cat="Integer",
        )
        pp = pulp.LpVariable.dicts(
            "pp",
            [(i, j) for i in range(max_stage_num) for j in range(num_cols)],
            lowBound=0,
            cat="Integer",
        )
        y = pulp.LpVariable.dicts(
            "y",
            [(i, j) for i in range(max_stage_num) for j in range(num_cols)],
            cat="Binary",
        )
        S = pulp.LpVariable("S", lowBound=0, cat="Integer")

        model += S

        for j in range(num_cols):
            model += pulp.lpSum(f[(i, j)] for i in range(max_stage_num)) == F[j]

        for j in range(num_cols):
            model += pulp.lpSum(h[(i, j)] for i in range(max_stage_num)) == H[j]

        for j in range(num_cols):
            model += pp[(0, j)] == self.pp[j]
        for i in range(1, max_stage_num):
            model += pp[(i, 0)] == pp[(i - 1, 0)] - 2 * f[(i - 1, 0)] - h[(i - 1, 0)]
            for j in range(1, num_cols):
                model += (
                    pp[(i, j)]
                    == pp[(i - 1, j)]
                    - 2 * f[(i - 1, j)]
                    - h[(i - 1, j)]
                    + f[(i - 1, j - 1)]
                    + h[(i - 1, j - 1)]
                )

        for i in range(max_stage_num):
            for j in range(num_cols):
                model += 3 * f[(i, j)] + 2 * h[(i, j)] <= pp[(i, j)]

        for i in range(max_stage_num):
            for j in range(num_cols):
                model += S >= i * y[(i, j)]

        for i in range(max_stage_num):
            for j in range(num_cols):
                model += M * y[(i, j)] >= f[(i, j)] + h[(i, j)]

        solver: pulp.LpSolver_CMD = getattr(pulp, method)(
            timeLimit=timeLimit,
            keepFiles=keepFiles,
            msg=True,
            options=[("Threads", str(n_processing))],
        )
        model.solve(solver)

        logging.info("Solver Status:", pulp.LpStatus[model.status])
        optimal_s = int(pulp.value(S)) + 1
        f_result = np.zeros((optimal_s, num_cols), dtype=int)
        h_result = np.zeros((optimal_s, num_cols), dtype=int)
        for i in range(optimal_s):
            for j in range(num_cols):
                f_result[i, j] = int(pulp.value(f[(i, j)]))
                h_result[i, j] = int(pulp.value(h[(i, j)]))
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
        return f_result, h_result, optimal_s, pulp.LpStatus[model.status]

    UFO_MAC_CONSTANT = {
        "FA": {
            "Tas": None,
            "Tac": None,
            "Tbs": None,
            "Tbc": None,
            "Tcs": None,
            "Tcc": None,
        },
        "HA": {
            "Tas": None,
            "Tac": None,
            "Tcs": None,
            "Tcc": None,
        },
    }

    def ufomac_router(
        self,
        dec_ct32: np.ndarray,
        dec_ct22: np.ndarray,
        method,
        Z_constant,
        n_processing,
        timeLimit,
        keepFiles,
        json_file,
        **kwargs,
    ) -> Tuple[str, List[deque]]:
        """
        Replication method from paper
            UFO-MAC: A Unified Framework for Optimization of High-Performance Multipliers and Multiply-Accumulators
            https://arxiv.org/pdf/2408.06935
        Solver:
            Pulp framework, use GUROBI_CMD default.

        0   1   2   3  : t_src_list[s] 这个阶段的 pp 数量
         Cell Delays   : UFO_MAC_CONSTANT
        a   b   c   d  : t_list[s] 下个阶段的 pp 数量
        Internconnect  : Z[s]
        0'  1'  2'  3' : t_src_list[s+1] next stage
        """
        start_time = time.time()
        slice_size_mat = np.zeros((dec_ct32.shape[0] + 1, dec_ct32.shape[1]), dtype=int)
        stage_num, col_num = dec_ct32.shape

        slice_size_mat[0, :] = self.pp
        for s in range(1, stage_num + 1):
            slice_size_mat[s, 0] = (
                slice_size_mat[s - 1, 0] - dec_ct32[s - 1, 0] * 2 - dec_ct22[s - 1, 0]
            )
            for c in range(1, col_num):
                slice_size_mat[s, c] = (
                    slice_size_mat[s - 1, c]
                    - dec_ct32[s - 1, c] * 2
                    - dec_ct22[s - 1, c]
                    + dec_ct32[s - 1, c - 1]
                    + dec_ct22[s - 1, c - 1]
                )
        print(slice_size_mat)
        print(dec_ct32)
        print(dec_ct22)

        model = pulp.LpProblem(f"ufomac_router_{method}_{col_num}", pulp.LpMinimize)
        t_src_list = []
        t_list = []
        Z_list = []
        for s in range(stage_num):
            t_src_s = []
            t_s = []
            Z_s = []
            for c in range(col_num):
                slice_size = int(slice_size_mat[s, c])
                next_slice_size = int(slice_size_mat[s + 1, c])
                Z = pulp.LpVariable.dicts(
                    f"Z_{s},{c}",
                    [
                        (i, j)
                        for i in range(next_slice_size)
                        for j in range(next_slice_size)
                    ],
                    cat="Binary",
                )
                Z_s.append(Z)
                t = pulp.LpVariable.dicts(
                    f"output_delay_{s},{c}",
                    [i for i in range(next_slice_size)],
                    cat="Continuous",
                    lowBound=0,
                )
                t_s.append(t)
                t_src = pulp.LpVariable.dicts(
                    f"src_delay_{s},{c}",
                    [i for i in range(slice_size)],
                    cat="Continuous",
                    lowBound=0,
                )
                t_src_s.append(t_src)
            Z_list.append(Z_s)
            t_list.append(t_s)
            t_src_list.append(t_src_s)
        M = pulp.LpVariable("M", lowBound=0, cat="Continuous")
        model += M

        for s in range(stage_num):
            for c in range(col_num):
                if c == 0:
                    last_carry_num = 0
                else:
                    last_carry_num = dec_ct32[s, c - 1] + dec_ct22[s, c - 1]
                for u in range(slice_size_mat[s, c]):
                    if s == 0:
                        model += t_src_list[0][c][u] == 0
                    else:
                        for v in range(slice_size_mat[s, c]):
                            model += t_src_list[s][c][u] - t_list[s - 1][c][
                                v
                            ] <= Z_constant * (1 - Z_list[s - 1][c][u, v])
                            model += t_list[s - 1][c][v] - t_src_list[s][c][
                                u
                            ] <= Z_constant * (1 - Z_list[s - 1][c][u, v])

                t_src_idx = 0
                t_sum_idx = last_carry_num
                t_carry_idx = 0
                for ct32_idx in range(dec_ct32[s, c]):
                    model += (
                        t_list[s][c][t_sum_idx]
                        >= t_src_list[s][c][t_src_idx]
                        + self.UFO_MAC_CONSTANT["FA"]["Tas"]
                    )
                    model += (
                        t_list[s][c][t_sum_idx]
                        >= t_src_list[s][c][t_src_idx + 1]
                        + self.UFO_MAC_CONSTANT["FA"]["Tbs"]
                    )
                    model += (
                        t_list[s][c][t_sum_idx]
                        >= t_src_list[s][c][t_src_idx + 2]
                        + self.UFO_MAC_CONSTANT["FA"]["Tcs"]
                    )
                    if c + 1 < col_num:
                        model += (
                            t_list[s][c + 1][t_carry_idx]
                            >= t_src_list[s][c][t_src_idx]
                            + self.UFO_MAC_CONSTANT["FA"]["Tac"]
                        )
                        model += (
                            t_list[s][c + 1][t_carry_idx]
                            >= t_src_list[s][c][t_src_idx + 1]
                            + self.UFO_MAC_CONSTANT["FA"]["Tbc"]
                        )
                        model += (
                            t_list[s][c + 1][t_carry_idx]
                            >= t_src_list[s][c][t_src_idx + 2]
                            + self.UFO_MAC_CONSTANT["FA"]["Tcc"]
                        )
                    t_src_idx += 3
                    t_sum_idx += 1
                    t_carry_idx += 1

                for ct22_idx in range(dec_ct22[s, c]):
                    model += (
                        t_list[s][c][t_sum_idx]
                        >= t_src_list[s][c][t_src_idx]
                        + self.UFO_MAC_CONSTANT["HA"]["Tas"]
                    )
                    model += (
                        t_list[s][c][t_sum_idx]
                        >= t_src_list[s][c][t_src_idx + 1]
                        + self.UFO_MAC_CONSTANT["HA"]["Tcs"]
                    )
                    if c + 1 < col_num:
                        model += (
                            t_list[s][c + 1][t_carry_idx]
                            >= t_src_list[s][c][t_src_idx]
                            + self.UFO_MAC_CONSTANT["HA"]["Tac"]
                        )
                        model += (
                            t_list[s][c + 1][t_carry_idx]
                            >= t_src_list[s][c][t_src_idx + 1]
                            + self.UFO_MAC_CONSTANT["HA"]["Tcc"]
                        )
                    t_src_idx += 2
                    t_carry_idx += 1
                    t_sum_idx += 1

                while t_src_idx < slice_size_mat[s, c]:
                    model += t_list[s][c][t_sum_idx] == t_src_list[s][c][t_src_idx]
                    t_sum_idx += 1
                    t_src_idx += 1

                for u in range(slice_size_mat[s + 1, c]):
                    model += (
                        pulp.lpSum(
                            Z_list[s][c][u, v] for v in range(slice_size_mat[s + 1, c])
                        )
                        == 1
                    )
                for v in range(slice_size_mat[s + 1, c]):
                    model += (
                        pulp.lpSum(
                            Z_list[s][c][u, v] for u in range(slice_size_mat[s + 1, c])
                        )
                        == 1
                    )
        for c in range(col_num):
            for i in range(len(t_list[-1][c])):
                model += M >= t_list[-1][c][i]

        print(model.objective)

        solver: pulp.LpSolver_CMD = getattr(pulp, method)(
            timeLimit=timeLimit,
            keepFiles=keepFiles,
            msg=True,
            options=[("Threads", str(n_processing))],
        )
        model.solve(solver)
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

        if json_file is not None:
            try:
                Z_result = []
                for s in range(stage_num - 1):
                    Z_s_result = []
                    for c in range(col_num):
                        Z = np.zeros(
                            (slice_size_mat[s + 1, c], slice_size_mat[s + 1, c]), int
                        )
                        for u in range(slice_size_mat[s + 1, c]):
                            for v in range(slice_size_mat[s + 1, c]):
                                Z[u, v] = int(pulp.value(Z_list[s][c][u, v]))
                        Z_s_result.append(Z.tolist())
                    Z_result.append(Z_s_result)
                with open(json_file, "w") as f:
                    json.dump(Z_result, f)
            except Exception as e:
                logging.error(f"Error saving Z result to {json_file}: {e}")

        v_src = ""
        wire_set = set()

        def __add_wire_str(wire_name):
            if wire_name in wire_set:
                return ""
            else:
                wire_set.add(wire_name)
                return self.declare_wire(wire_name)

        for s in range(stage_num):
            last_carry_num = 0
            v_src += f"    // stage {s}\n"
            for c in range(col_num):
                for u in range(slice_size_mat[s, c]):
                    wire_src = f"wire_src_s{s}_c{c}_{u}"
                    v_src += __add_wire_str(wire_src)
                    if s == 0:
                        v_src += f"    assign {wire_src} = pp_{c}[{u}];\n"
                    else:
                        for v in range(slice_size_mat[s, c]):
                            if int(pulp.value(Z_list[s - 1][c][u, v])) == 1:
                                wire = f"wire_s{s-1}_c{c}_{v}"
                                break
                        v_src += __add_wire_str(wire_src)
                        v_src += f"    assign {wire_src} = {wire};\n"

                if c == 0:
                    last_carry_num = 0
                else:
                    last_carry_num = dec_ct32[s, c - 1] + dec_ct22[s, c - 1]

                t_sum_idx = last_carry_num
                t_src_idx = 0
                t_carry_idx = 0
                for ct32_idx in range(dec_ct32[s, c]):
                    wire_sum = f"wire_s{s}_c{c}_{t_sum_idx}"
                    wire_carry = f"wire_s{s}_c{c+1}_{t_carry_idx}"
                    wire_a = f"wire_src_s{s}_c{c}_{t_src_idx}"
                    wire_b = f"wire_src_s{s}_c{c}_{t_src_idx+1}"
                    wire_c = f"wire_src_s{s}_c{c}_{t_src_idx+2}"
                    v_src += __add_wire_str(wire_a)
                    v_src += __add_wire_str(wire_b)
                    v_src += __add_wire_str(wire_c)
                    v_src += __add_wire_str(wire_sum)
                    v_src += __add_wire_str(wire_carry)
                    v_src += self.declare_fa(
                        f"ct32_s{s}_c{c}_{ct32_idx}",
                        [wire_a, wire_b, wire_c],
                        wire_sum,
                        wire_carry,
                    )
                    t_src_idx += 3
                    t_sum_idx += 1
                    t_carry_idx += 1

                for ct22_idx in range(dec_ct22[s, c]):
                    wire_sum = f"wire_s{s}_c{c}_{t_sum_idx}"
                    wire_carry = f"wire_s{s}_c{c+1}_{t_carry_idx}"
                    wire_a = f"wire_src_s{s}_c{c}_{t_src_idx}"
                    wire_c = f"wire_src_s{s}_c{c}_{t_src_idx+1}"
                    v_src += __add_wire_str(wire_a)
                    v_src += __add_wire_str(wire_c)
                    v_src += __add_wire_str(wire_sum)
                    v_src += __add_wire_str(wire_carry)
                    v_src += self.declare_ha(
                        f"ct22_s{s}_c{c}_{ct22_idx}",
                        [wire_a, wire_c],
                        wire_sum,
                        wire_carry,
                    )

                    t_src_idx += 2
                    t_carry_idx += 1
                    t_sum_idx += 1

                while t_src_idx < slice_size_mat[s, c]:
                    wire_remain = f"wire_s{s}_c{c}_{t_sum_idx}"
                    wire_src = f"wire_src_s{s}_c{c}_{t_src_idx}"
                    v_src += __add_wire_str(wire_remain)
                    v_src += __add_wire_str(wire_src)
                    v_src += f"    assign {wire_remain} = {wire_src};\n"
                    t_sum_idx += 1
                    t_src_idx += 1
        routed_wire_list = []
        for c in range(col_num):
            wire_list = []
            for u in range(slice_size_mat[-1, c]):
                wire_list.append(f"wire_s{stage_num-1}_c{c}_{u}")
            routed_wire_list.append(deque(wire_list))
        return v_src, routed_wire_list

    def domac_router(
        self,
        dec_ct32: np.ndarray,
        dec_ct22: np.ndarray,
        gamma,
        lr,
        train_steps,
        bm_steps,
        device,
        weight_wns,
        weight_tns,
        weight_L_D,
        weight_L_BM,
        weight_L_inc,
        tb_logger: torch.utils.tensorboard.writer.SummaryWriter = None,
        json_file=None,
        load_from_json=None,
        **kwargs,
    ):
        """
        Replication method from paper
            DOMAC: Differentiable Optimization for High-Speed Multipliers and Multiply-Accumulators
            https://arxiv.org/abs/2503.23943
        0   1   2   3  : t_src_list[s] 这个阶段的 pp 数量
         Cell Delays   : UFO_MAC_CONSTANT
        a   b   c   d  : t_list[s] 下个阶段的 pp 数量
        Internconnect  : Z[s]
        0'  1'  2'  3' : t_src_list[s+1] next stage
        """
        slice_size_mat = np.zeros((dec_ct32.shape[0] + 1, dec_ct32.shape[1]), dtype=int)
        stage_num, col_num = dec_ct32.shape

        slice_size_mat[0, :] = self.pp
        for s in range(1, stage_num + 1):
            slice_size_mat[s, 0] = (
                slice_size_mat[s - 1, 0] - dec_ct32[s - 1, 0] * 2 - dec_ct22[s - 1, 0]
            )
            for c in range(1, col_num):
                slice_size_mat[s, c] = (
                    slice_size_mat[s - 1, c]
                    - dec_ct32[s - 1, c] * 2
                    - dec_ct22[s - 1, c]
                    + dec_ct32[s - 1, c - 1]
                    + dec_ct22[s - 1, c - 1]
                )
        print(slice_size_mat)

        Z_list = []
        for s in range(stage_num):
            Z_s = []
            for c in range(col_num):
                next_slice_size = int(slice_size_mat[s + 1, c])
                Z = torch.nn.Parameter(
                    torch.randn(
                        (next_slice_size, next_slice_size),
                        dtype=torch.float32,
                        device=device,
                    ),
                    requires_grad=True,
                )
                Z_s.append(Z)
            Z_list.append(Z_s)

        def _get_wns_tns():
            t_set = {}
            t_src_set = {}
            for s in range(stage_num):
                for c in range(col_num):
                    if c == 0:
                        last_carry_num = 0
                    else:
                        last_carry_num = dec_ct32[s, c - 1] + dec_ct22[s, c - 1]
                    if s == 0:
                        for u in range(slice_size_mat[s, c]):
                            t_u = torch.zeros(
                                (1),
                                dtype=torch.float32,
                                device=device,
                            )
                            t_src_set[f"{s}_{c}_{u}"] = t_u
                        t_src = torch.concatenate(
                            [
                                t_src_set[f"{s}_{c}_{u}"]
                                for u in range(slice_size_mat[s, c])
                            ]
                        )
                    else:
                        t = torch.stack(
                            [
                                t_set[f"{s-1}_{c}_{u}"]
                                for u in range(slice_size_mat[s, c])
                            ]
                        )
                        t_src = torch.softmax(Z_list[s - 1][c], dim=-1).T @ t

                    t_src_idx = 0
                    t_sum_idx = last_carry_num
                    t_carry_idx = 0
                    for ct32_idx in range(dec_ct32[s, c]):
                        t_as = t_src[t_src_idx] + self.UFO_MAC_CONSTANT["FA"]["Tas"]
                        t_bs = t_src[t_src_idx + 1] + self.UFO_MAC_CONSTANT["FA"]["Tbs"]
                        t_cs = t_src[t_src_idx + 2] + self.UFO_MAC_CONSTANT["FA"]["Tcs"]
                        t_sum = lse_gamma(torch.stack([t_as, t_bs, t_cs]), gamma)
                        t_set[f"{s}_{c}_{t_sum_idx}"] = t_sum
                        if c + 1 < col_num:
                            t_ac = t_src[t_src_idx] + self.UFO_MAC_CONSTANT["FA"]["Tac"]
                            t_bc = (
                                t_src[t_src_idx + 1]
                                + self.UFO_MAC_CONSTANT["FA"]["Tbc"]
                            )
                            t_cc = (
                                t_src[t_src_idx + 2]
                                + self.UFO_MAC_CONSTANT["FA"]["Tcc"]
                            )
                            t_carry = lse_gamma(torch.stack([t_ac, t_bc, t_cc]), gamma)
                            t_set[f"{s}_{c+1}_{t_carry_idx}"] = t_carry
                        t_src_idx += 3
                        t_sum_idx += 1
                        t_carry_idx += 1
                    for ct22_idx in range(dec_ct22[s, c]):
                        t_as = t_src[t_src_idx] + self.UFO_MAC_CONSTANT["HA"]["Tas"]
                        t_cs = t_src[t_src_idx + 1] + self.UFO_MAC_CONSTANT["HA"]["Tcs"]
                        t_sum = lse_gamma(torch.stack([t_as, t_cs]), gamma)
                        t_set[f"{s}_{c}_{t_sum_idx}"] = t_sum
                        if c + 1 < col_num:
                            t_ac = t_src[t_src_idx] + self.UFO_MAC_CONSTANT["HA"]["Tac"]
                            t_cc = (
                                t_src[t_src_idx + 1]
                                + self.UFO_MAC_CONSTANT["HA"]["Tcc"]
                            )
                            t_carry = lse_gamma(torch.stack([t_ac, t_cc]), gamma)
                            t_set[f"{s}_{c+1}_{t_carry_idx}"] = t_carry
                        t_src_idx += 2
                        t_carry_idx += 1
                        t_sum_idx += 1
                    while t_src_idx < slice_size_mat[s, c]:
                        t_remain = t_src[t_src_idx]
                        t_set[f"{s}_{c}_{t_sum_idx}"] = t_remain
                        t_sum_idx += 1
                        t_src_idx += 1
            last_stage_t_list = []
            for c in range(col_num):
                for u in range(slice_size_mat[-1, c]):
                    last_stage_t_list.append(t_set[f"{stage_num-1}_{c}_{u}"])
            L_WNS = lse_gamma(torch.stack(last_stage_t_list), gamma)
            L_TNS = torch.sum(torch.stack([t for t in last_stage_t_list], dim=0))

            return L_WNS, L_TNS

        def _get_L_BM():
            L_BM: torch.Tensor = 0.0
            for s in range(stage_num):
                for c in range(col_num):
                    Z: torch.Tensor = torch.softmax(Z_list[s][c], dim=-1)
                    for v in range(Z.shape[0]):
                        L_BM += (1 - torch.sum(Z[:, v])) ** 2
            return L_BM

        def _get_L_D():
            L_D: torch.Tensor = 0.0
            for s in range(stage_num):
                for c in range(col_num):
                    Z: torch.Tensor = torch.softmax(Z_list[s][c], dim=-1)
                    L_D += torch.sum(Z.pow(2) * (1 - Z).pow(2))
            return L_D

        optim = torch.optim.Adam(
            [
                *[Z_list[s][c] for s in range(stage_num) for c in range(col_num)],
            ],
            lr=lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=train_steps, eta_min=lr / 10
        )

        if load_from_json is None:
            for step in range(train_steps):
                if step < bm_steps:
                    L_WNS, L_TNS = _get_wns_tns()
                    L_BM = _get_L_BM()
                    L_D = _get_L_D()
                    loss = (
                        weight_wns * L_WNS
                        + weight_tns * L_TNS
                        + weight_L_D * L_D
                        + weight_L_BM * L_BM
                    )
                else:
                    L_BM = _get_L_BM()
                    L_D = _get_L_D()
                    loss = weight_L_D * L_D + weight_L_BM * L_BM

                optim.zero_grad()
                loss.backward()
                optim.step()

                lr_scheduler.step()
                print(
                    f"step {step}: loss: {loss.item():.4f}, L_WNS: {L_WNS.item():.4f}, L_TNS: {L_TNS.item():.4f}, L_BM: {L_BM.item():.4f}, L_D: {L_D.item():.4f}"
                )
                if tb_logger is not None:
                    tb_logger.add_scalar("loss", loss.item(), step)
                    tb_logger.add_scalar("L_WNS", L_WNS.item(), step)
                    tb_logger.add_scalar("L_TNS", L_TNS.item(), step)
                    tb_logger.add_scalar("L_BM", L_BM.item(), step)
                    tb_logger.add_scalar("L_D", L_D.item(), step)
                    tb_logger.add_scalar("lr", optim.param_groups[0]["lr"], step)
                    tb_logger.add_scalar("weight_L", weight_L_inc, step)
                    tb_logger.add_scalar("weight_L_BM", weight_L_BM, step)
                L_BM = L_BM * (1 + weight_L_inc)
                L_D = L_D * (1 + weight_L_inc)

            Z_result = []
            for s in range(stage_num):
                Z_s_result = []
                for c in range(col_num):
                    Z = torch.softmax(Z_list[s][c], dim=-1).cpu().detach().numpy()
                    Z_s_result.append(Z.tolist())
                Z_result.append(Z_s_result)
            if json_file is not None:
                try:
                    with open(json_file, "w") as f:
                        json.dump(Z_result, f)
                except Exception as e:
                    logging.error(f"Error saving Z result to {json_file}: {e}")
        else:
            print(f"Load from {load_from_json}")
            with open(load_from_json, "r") as file:
                routing_info = json.load(file)
            Z_result = []
            for s in range(stage_num):
                Z_s_result = []
                for c in range(col_num):
                    Z = routing_info[s][c]
                    Z_s_result.append(Z)
                Z_result.append(Z_s_result)
            
        v_src = ""
        wire_set = set()

        def __add_wire_str(wire_name):
            if wire_name in wire_set:
                return ""
            else:
                wire_set.add(wire_name)
                return self.declare_wire(wire_name)

        for s in range(stage_num):
            last_carry_num = 0
            v_src += f"    // stage {s}\n"
            for c in range(col_num):
                Z_arr = np.asarray(Z_result[s - 1][c])
                for u in range(slice_size_mat[s, c]):
                    wire_src = f"wire_src_s{s}_c{c}_{u}"
                    v_src += __add_wire_str(wire_src)
                    if s == 0:
                        v_src += f"    assign {wire_src} = pp_{c}[{u}];\n"
                    else:
                        v = np.argmax(Z_arr[u, :])
                        Z_arr[:, v] = -1
                        v_src += __add_wire_str(wire_src)
                        wire = f"wire_s{s-1}_c{c}_{v}"
                        v_src += f"    assign {wire_src} = {wire};\n"

                if c == 0:
                    last_carry_num = 0
                else:
                    last_carry_num = dec_ct32[s, c - 1] + dec_ct22[s, c - 1]

                t_sum_idx = last_carry_num
                t_src_idx = 0
                t_carry_idx = 0
                for ct32_idx in range(dec_ct32[s, c]):
                    wire_sum = f"wire_s{s}_c{c}_{t_sum_idx}"
                    wire_carry = f"wire_s{s}_c{c+1}_{t_carry_idx}"
                    wire_a = f"wire_src_s{s}_c{c}_{t_src_idx}"
                    wire_b = f"wire_src_s{s}_c{c}_{t_src_idx+1}"
                    wire_c = f"wire_src_s{s}_c{c}_{t_src_idx+2}"
                    v_src += __add_wire_str(wire_a)
                    v_src += __add_wire_str(wire_b)
                    v_src += __add_wire_str(wire_c)
                    v_src += __add_wire_str(wire_sum)
                    v_src += __add_wire_str(wire_carry)
                    v_src += self.declare_fa(
                        f"ct32_s{s}_c{c}_{ct32_idx}",
                        [wire_a, wire_b, wire_c],
                        wire_sum,
                        wire_carry,
                    )
                    t_src_idx += 3
                    t_sum_idx += 1
                    t_carry_idx += 1

                for ct22_idx in range(dec_ct22[s, c]):
                    wire_sum = f"wire_s{s}_c{c}_{t_sum_idx}"
                    wire_carry = f"wire_s{s}_c{c+1}_{t_carry_idx}"
                    wire_a = f"wire_src_s{s}_c{c}_{t_src_idx}"
                    wire_c = f"wire_src_s{s}_c{c}_{t_src_idx+1}"
                    v_src += __add_wire_str(wire_a)
                    v_src += __add_wire_str(wire_c)
                    v_src += __add_wire_str(wire_sum)
                    v_src += __add_wire_str(wire_carry)
                    v_src += self.declare_ha(
                        f"ct22_s{s}_c{c}_{ct22_idx}",
                        [wire_a, wire_c],
                        wire_sum,
                        wire_carry,
                    )

                    t_src_idx += 2
                    t_carry_idx += 1
                    t_sum_idx += 1

                while t_src_idx < slice_size_mat[s, c]:
                    wire_remain = f"wire_s{s}_c{c}_{t_sum_idx}"
                    wire_src = f"wire_src_s{s}_c{c}_{t_src_idx}"
                    v_src += __add_wire_str(wire_remain)
                    v_src += __add_wire_str(wire_src)
                    v_src += f"    assign {wire_remain} = {wire_src};\n"
                    t_sum_idx += 1
                    t_src_idx += 1
        routed_wire_list = []
        for c in range(col_num):
            wire_list = []
            for u in range(slice_size_mat[-1, c]):
                wire_list.append(f"wire_s{stage_num-1}_c{c}_{u}")
            routed_wire_list.append(deque(wire_list))
        return v_src, routed_wire_list

    def random_router(
        self, dec_ct32: np.ndarray, dec_ct22: np.ndarray
    ) -> Tuple[str, List[deque]]:
        v_src = ""
        stage_num = len(dec_ct32)
        column_num = len(self.pp)
        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(stage_num):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_wire_list = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for ct32_index in range(int(dec_ct32[stage_index][column_index])):
                    ct32_name = f"ct32_s{stage_index}_c{column_index}_{ct32_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(3)
                    ]
                    sum_name = f"sum_from_{ct32_name}"
                    carry_name = f"carry_from_{ct32_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_fa(
                        ct32_name, input_wires, sum_name, carry_name
                    )
                for ct22_index in range(int(dec_ct22[stage_index][column_index])):
                    ct22_name = f"ct22_s{stage_index}_c{column_index}_{ct22_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(2)
                    ]
                    sum_name = f"sum_from_{ct22_name}"
                    carry_name = f"carry_from_{ct22_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_ha(
                        ct22_name, input_wires, sum_name, carry_name
                    )
                all_wires = []
                while len(output_wire_list[column_index]) > 0:
                    wire = output_wire_list[column_index].pop()
                    all_wires.append(wire)
                while len(input_wire_list[column_index]) > 0:
                    wire = input_wire_list[column_index].pop()
                    all_wires.append(wire)
                np.random.shuffle(all_wires)
                for wire in all_wires:
                    input_wire_list[column_index].appendleft(wire)
        v_src += "\n"
        routed_wire_list = input_wire_list
        return v_src, routed_wire_list

    def declare_wire(self, wire_name):
        return f"    wire {wire_name};\n"

    def declare_fa(self, fa_name, input_list, sum_name, carry_name, cell=None):
        # cell=None -> 精确 "FA"；传近似 3:2 cell 名(端口同 FA: a,b,cin,sum,cout)即可替换
        v_src = f"    {cell or 'FA'} {fa_name} (\n"
        v_src += f"        .a ({input_list[0]}),\n"
        v_src += f"        .b({input_list[1]}),\n"
        v_src += f"        .cin({input_list[2]}),\n"
        v_src += f"        .sum({sum_name}),\n"
        v_src += f"        .cout({carry_name})\n"
        v_src += "    );\n"
        return v_src

    def declare_ha(self, ha_name, input_list, sum_name, carry_name, cell=None):
        # cell=None -> 精确 "HA"；传近似 2:2 cell 名(端口同 HA: a,cin,sum,cout)即可替换
        v_src = f"    {cell or 'HA'} {ha_name} (\n"
        v_src += f"        .a ({input_list[0]}),\n"
        v_src += f"        .cin({input_list[1]}),\n"
        v_src += f"        .sum({sum_name}),\n"
        v_src += f"        .cout({carry_name})\n"
        v_src += "    );\n"
        return v_src

    def declare_comment(self, comment: str, num_stars=50, align="c"):
        v_src = "/*" + "*" * num_stars + "\n"
        lines = comment.splitlines()
        for line in lines:
            if len(line) > 0:
                if align == "c":
                    v_src += " * " + line.center(num_stars - 2) + "*\n"
                elif align == "l":
                    v_src += " * " + line.ljust(num_stars - 2) + "*\n"
                else:
                    v_src += " * " + line.rjust(num_stars - 2) + "*\n"
        v_src += "*" * num_stars + "*/\n"
        return v_src

    def emit_verilog_backup(self, file_path=None, dec_ct32=None, dec_ct22=None):
        if dec_ct32 is None or dec_ct22 is None:
            dec_ct32, dec_ct22 = self.compressor_assignment()
        stage_num = len(dec_ct32)
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        comment += f"dec_ct32=\n{dec_ct32}\n"
        comment += f"dec_ct22=\n{dec_ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(stage_num):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_wire_list = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for ct32_index in range(int(dec_ct32[stage_index][column_index])):
                    ct32_name = f"ct32_s{stage_index}_c{column_index}_{ct32_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(3)
                    ]
                    sum_name = f"sum_from_{ct32_name}"
                    carry_name = f"carry_from_{ct32_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_fa(
                        ct32_name, input_wires, sum_name, carry_name
                    )
                for ct22_index in range(int(dec_ct22[stage_index][column_index])):
                    ct22_name = f"ct22_s{stage_index}_c{column_index}_{ct22_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(2)
                    ]
                    sum_name = f"sum_from_{ct22_name}"
                    carry_name = f"carry_from_{ct22_name}"
                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_ha(
                        ct22_name, input_wires, sum_name, carry_name
                    )
                while len(output_wire_list[column_index]) > 0:
                    wire = output_wire_list[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
        v_src += "\n"

        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = input_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError

        v_src += f"    assign out = a + b;\n"
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def rlmul_assigner(
        self, dec_ct32: np.ndarray, dec_ct22: np.ndarray
    ) -> Tuple[str, List[deque]]:
        v_src = ""
        stage_num = len(dec_ct32)
        column_num = len(self.pp)
        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(stage_num):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_sum_list_32 = [deque() for _ in range(column_num)]
            output_sum_list_22 = [deque() for _ in range(column_num)]
            output_carry_list_32 = [deque() for _ in range(column_num)]
            output_carry_list_22 = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for ct32_index in range(int(dec_ct32[stage_index][column_index])):
                    ct32_name = f"ct32_s{stage_index}_c{column_index}_{ct32_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(3)
                    ]
                    sum_name = f"sum_from_{ct32_name}"
                    carry_name = f"carry_from_{ct32_name}"
                    output_sum_list_32[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_carry_list_32[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_fa(
                        ct32_name, input_wires, sum_name, carry_name
                    )
                for ct22_index in range(int(dec_ct22[stage_index][column_index])):
                    ct22_name = f"ct22_s{stage_index}_c{column_index}_{ct22_index}"
                    input_wires = [
                        input_wire_list[column_index].pop() for _ in range(2)
                    ]
                    sum_name = f"sum_from_{ct22_name}"
                    carry_name = f"carry_from_{ct22_name}"
                    output_sum_list_22[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_carry_list_22[column_index + 1].appendleft(carry_name)
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)
                    v_src += self.declare_ha(
                        ct22_name, input_wires, sum_name, carry_name
                    )
                while len(output_carry_list_32[column_index]) > 0:
                    wire = output_carry_list_32[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
                while len(output_carry_list_22[column_index]) > 0:
                    wire = output_carry_list_22[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
                while len(output_sum_list_22[column_index]) > 0:
                    wire = output_sum_list_22[column_index].pop()
                    input_wire_list[column_index].append(wire)
                while len(output_sum_list_32[column_index]) > 0:
                    wire = output_sum_list_32[column_index].pop()
                    input_wire_list[column_index].append(wire)
        v_src += "\n"
        routed_wire_list = input_wire_list
        return v_src, routed_wire_list

    def emit_prefix_adder(self, prefix_adder: str):
        v_src = ""
        if prefix_adder is None:
            v_src += f"    assign out = a + b;\n"
        else:
            remain_pp = self.get_remain_pp()
            cell_map = adder_utils.get_init_cell_map(len(remain_pp), prefix_adder)
            v_src += f"    // {prefix_adder} adder\n"
            v_src += adder_utils.emit_fused_verilog(cell_map, remain_pp)
        return v_src

    def emit_verilog(
        self,
        file_path: str = None,
        dec_ct32: np.ndarray = None,
        dec_ct22: np.ndarray = None,
        router: str = None,
        prefix_adder: str = None,
        n_processing=4,
        Z_constant=1000,
        timeLimit=60,
        keepFiles=False,
        method="GUROBI_CMD",
        **kwargs,
    ):
        if dec_ct32 is None or dec_ct22 is None:
            dec_ct32, dec_ct22 = self.compressor_assignment()
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        comment += f"dec_ct32=\n{dec_ct32}\n"
        comment += f"dec_ct22=\n{dec_ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        if router is None or router == "None" or router == "rlmul":
            router_src, routed_wire_list = self.rlmul_assigner(dec_ct32, dec_ct22)
        elif router == "random":
            router_src, routed_wire_list = self.random_router(dec_ct32, dec_ct22)
        elif router == "ufomac":
            router_src, routed_wire_list = self.ufomac_router(
                dec_ct32,
                dec_ct22,
                method,
                Z_constant,
                n_processing,
                timeLimit,
                keepFiles,
                **kwargs,
            )
        elif router == "domac":
            router_src, routed_wire_list = self.domac_router(
                dec_ct32,
                dec_ct22,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Unknown router: {router}")
        v_src += router_src
        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = routed_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError
        v_src += self.emit_prefix_adder(prefix_adder)
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def emit_verilog_fused_assignment(
        self,
        file_path: str = None,
        assignment: List[List[List[Tuple]]] = None,
        prefix_adder: str = None,
        cell_policy: Callable = None,
    ):
        # cell_policy(s, c, t, idx) -> 该压缩器槽要实例化的 cell 名 (None=精确 FA/HA)。
        # 用于近似压缩器替换；不传则与原行为完全一致。
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        input_wire_list = []
        for column_index in range(column_num):
            wire_list = [
                f"pp_{column_index}[{wire_index}]"
                for wire_index in range(int(self.pp[column_index]))
            ]
            input_wire_list.append(deque(wire_list))
        for stage_index in range(len(assignment)):
            v_src += self.declare_comment(f"stage {stage_index}")
            output_wire_list = [deque() for _ in range(column_num)]
            for column_index in range(column_num):
                for compressor_id in assignment[stage_index][column_index]:
                    s, c, t, idx = compressor_id
                    assert s == stage_index and c == column_index
                    if t == 0:
                        ct_name = f"ct32_s{stage_index}_c{column_index}_{idx}"
                        input_wires = [
                            input_wire_list[column_index].pop() for _ in range(3)
                        ]
                        declare_ct: Callable = self.declare_fa
                    elif t == 1:
                        ct_name = f"ct22_s{stage_index}_c{column_index}_{idx}"
                        input_wires = [
                            input_wire_list[column_index].pop() for _ in range(2)
                        ]
                        declare_ct: Callable = self.declare_ha
                    else:
                        raise NotImplementedError(f"Unknown compressor type: {t}")
                    sum_name = f"sum_from_{ct_name}"
                    carry_name = f"carry_from_{ct_name}"
                    v_src += self.declare_wire(sum_name)
                    v_src += self.declare_wire(carry_name)

                    cell = cell_policy(s, c, t, idx) if cell_policy is not None else None
                    v_src += declare_ct(ct_name, input_wires, sum_name, carry_name, cell=cell)

                    output_wire_list[column_index].appendleft(sum_name)
                    if column_index + 1 < column_num:
                        output_wire_list[column_index + 1].appendleft(carry_name)
                while len(output_wire_list[column_index]) > 0:
                    wire = output_wire_list[column_index].pop()
                    input_wire_list[column_index].appendleft(wire)
        v_src += "\n"
        routed_wire_list = input_wire_list

        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = routed_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError

        v_src += self.emit_prefix_adder(prefix_adder)
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def emit_verilog_from_dict(
        self,
        file_path: str = None,
        assignment: dict = None,
        prefix_adder: str = None,
    ):
        column_num = len(self.pp)

        comment = ""
        comment += f"Generated by CompressorTree\n"
        comment += f"ct32={self.ct32}\n"
        comment += f"ct22={self.ct22}\n"
        v_src = self.declare_comment(comment, align="l")

        v_src += "module CompressorTree(\n"
        input_pp_list = []
        for column_index in range(len(self.pp)):
            pp_height = self.pp[column_index]
            if pp_height > 0:
                v_src += f"    input [{pp_height-1}:0] pp_{column_index},\n"
        v_src += ",\n".join(input_pp_list)
        v_src += f"    output [{column_num-1}:0] out"
        v_src += "\n);\n"

        v_src += assignment["router_src"]
        routed_wire_list = assignment["routed_wire_list"]

        v_src += f"    wire [{column_num-1}:0] a;\n"
        v_src += f"    wire [{column_num-1}:0] b;\n"
        for column_index in range(column_num):
            input_wires = routed_wire_list[column_index]
            if len(input_wires) == 1:
                wire = input_wires[0]
                v_src += f"    assign a[{column_index}] = {wire};\n"
                v_src += f"    assign b[{column_index}] = 1'b0;\n"
            elif len(input_wires) == 2:
                wire_1, wire_2 = input_wires
                v_src += f"    assign a[{column_index}] = {wire_1};\n"
                v_src += f"    assign b[{column_index}] = {wire_2};\n"
            else:
                raise ValueError

        v_src += self.emit_prefix_adder(prefix_adder)
        v_src += "endmodule\n"

        if file_path is not None:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w") as f:
                f.write(v_src)
        return v_src

    def legalize(self):
        initial_pp = copy.deepcopy(self.pp)
        ct32 = copy.deepcopy(self.ct32)
        ct22 = copy.deepcopy(self.ct22)
        assert len(ct32) == len(initial_pp) and len(ct22) == len(initial_pp)
        for column_index in range(0, len(initial_pp)):
            ct32[column_index] = max(0, ct32[column_index])
            ct22[column_index] = max(0, ct22[column_index])
            if column_index == 0:
                remain_pp = (
                    initial_pp[column_index]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            else:
                remain_pp = (
                    initial_pp[column_index]
                    + ct32[column_index - 1]
                    + ct22[column_index - 1]
                    - 2 * ct32[column_index]
                    - ct22[column_index]
                )
            if remain_pp < 1:
                if ct22[column_index] + remain_pp >= 1:
                    ct22[column_index] += remain_pp - 1
                else:
                    remain_pp += ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] -= (2 - remain_pp) // 2
                    else:
                        ct32[column_index] -= (1 - remain_pp) // 2
            elif remain_pp > 2:
                if remain_pp - ct22[column_index] <= 2:
                    ct22[column_index] -= remain_pp - 2
                    ct32[column_index] += remain_pp - 2
                else:
                    ct32[column_index] += ct22[column_index]
                    remain_pp -= ct22[column_index]
                    ct22[column_index] = 0
                    if remain_pp % 2 == 0:
                        ct32[column_index] += (remain_pp - 2) / 2
                    else:
                        ct32[column_index] += (remain_pp - 1) / 2

        remain_pp = copy.deepcopy(initial_pp)
        remain_pp[0] = initial_pp[0] - 2 * ct32[0] - ct22[0]
        for column_index in range(1, len(initial_pp)):
            remain_pp[column_index] = (
                initial_pp[column_index]
                + ct32[column_index - 1]
                + ct22[column_index - 1]
                - 2 * ct32[column_index]
                - ct22[column_index]
            )
        remain_pp = np.asarray(remain_pp)
        assert (remain_pp >= 1).all() and (remain_pp <= 2).all(), "legalize fail"
        self.ct32 = ct32
        self.ct22 = ct22
        return ct32, ct22

    def get_remain_pp(self):
        remain_pp = np.zeros_like(self.pp)
        carry_in = 0
        for i in range(len(self.pp)):
            remain_pp[i] = self.pp[i] + carry_in - self.ct32[i] * 2 - self.ct22[i]
            carry_in = self.ct32[i] + self.ct22[i]
        return remain_pp

    def __repr__(self):
        return f"ct32={self.ct32}, ct22={self.ct22}"
