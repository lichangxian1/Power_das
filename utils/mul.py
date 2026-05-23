import os
from .compressor_tree import CompressorTree, get_initial_partial_product
from utils import (
    lef_path,
    lib_path,
    abc_constr,
    yosys_script_template,
    sta_script_template,
    FA_verilog_src,
    HA_verilog_src,
    FA_no_carry_verilog_src,
    HA_no_carry_verilog_src,
)
import utils
import multiprocessing
import matplotlib.pyplot as plt
import copy
import numpy as np
from tqdm import tqdm
import time
import json
import logging


class Mul:
    def __init__(
        self,
        bit_width: int,
        encode_type: str,
        ct: str,
    ):
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.initial_pp = get_initial_partial_product(bit_width, encode_type)
        if isinstance(ct, str):
            if ct == "wallace":
                self.ct = CompressorTree.wallace(self.initial_pp)
            elif ct == "dadda":
                self.ct = CompressorTree.dadda(self.initial_pp)
            else:
                raise ValueError(f"Unknown ct type: {ct}")
        elif isinstance(ct, np.ndarray):
            ct32, ct22 = ct.astype(int)
            self.ct = CompressorTree(self.initial_pp, ct32, ct22)
        elif isinstance(ct, CompressorTree):
            self.ct = ct
        else:
            raise ValueError(f"Unknown ct type: {ct}")

    def random_assignment(self):
        column_len = len(self.initial_pp)
        ct32_remain = copy.deepcopy(self.ct.ct32)
        ct22_remain = copy.deepcopy(self.ct.ct22)
        current_stage = 0
        ct32_decomposed = np.zeros([1, len(self.ct.ct32)])
        ct22_decomposed = np.zeros([1, len(self.ct.ct22)])

        current_pp = copy.deepcopy(self.initial_pp)
        timeout = 999
        while ((ct32_remain > 0).any() or (ct22_remain > 0).any()) and timeout > 0:
            timeout -= 1
            next_pp = np.zeros_like(current_pp)
            for column_index in range(column_len):
                current_pp_height = int(current_pp[column_index])
                max_ct22_num = min(
                    int(ct22_remain[column_index]),
                    current_pp_height // 2,
                )
                assigned_ct22 = np.random.randint(max_ct22_num + 1)
                ct22_remain[column_index] -= assigned_ct22
                current_pp_height -= assigned_ct22 * 2

                max_ct32_num = min(
                    int(ct32_remain[column_index]),
                    current_pp_height // 3,
                )
                assigned_ct32 = max_ct32_num
                ct32_remain[column_index] -= assigned_ct32
                current_pp_height -= assigned_ct32 * 3

                next_pp[column_index] += (
                    current_pp_height + assigned_ct32 + assigned_ct22
                )

                if column_index < column_len - 1:
                    next_pp[column_index + 1] += assigned_ct32 + assigned_ct22
                ct32_decomposed[current_stage][column_index] = assigned_ct32
                ct22_decomposed[current_stage][column_index] = assigned_ct22
            current_pp = next_pp
            ct32_decomposed = np.r_[ct32_decomposed, np.zeros([1, len(self.ct.ct32)])]
            ct22_decomposed = np.r_[ct22_decomposed, np.zeros([1, len(self.ct.ct22)])]
            current_stage += 1
        assert timeout > 0
        return ct32_decomposed, ct22_decomposed

    def emit_pp_encoder(self) -> str:
        """
        and encoding
              _____a[x] & b[y], x + y = column_index
              |
              o o o o  -> a[0]
            o o o o    -> a[1]
          o o o o      -> a[2]
        o o o o        -> a[3]
            |___ a[.] not stating from a[0] after here

        booth encoding:
        5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
        ================================
                n s s o o o o o o o o o
              1 n o o o o o o o o o   s
          1 n o o o o o o o o o   s
        n o o o o o o o o o   s
        o o o o o o o o   s
        |_____| |_____| |_____________|
          s3     s2          s1

        booth encoder:
        0 9 8 7 6 5 4 3 2 1 0
        ======================
        [ 4 ]   [ 2 ]   [ 0 ]
        . . o o o o o o o o .
            [ 3 ]   [ 1 ]
                      |____center index of encoder
        '.': zeros
        'o': bits of input x
        '[*]': encoders
        """
        f_str = "    "
        verilog_str = f"{f_str}// pp_encoder\n"
        initial_pp = get_initial_partial_product(
            self.bit_width, self.encode_type
        ).astype(int)
        if self.encode_type == "and":
            for column_index in range(len(initial_pp)):
                verilog_str += f"{f_str}wire [{int(initial_pp[column_index]) - 1}:0] pp_{column_index};\n"
            verilog_str += "\n"
            for column_index in range(len(initial_pp)):
                for pp_index in range(int(initial_pp[column_index])):
                    offset = max(0, column_index - self.bit_width + 1)
                    verilog_str += f"{f_str}assign pp_{column_index}[{pp_index}] = a[{pp_index + offset}] & b[{column_index - pp_index - offset}];\n"

        elif self.encode_type == "booth":
            encoder_num = self.bit_width // 2 + 1
            verilog_str += f"{f_str}wire [{self.bit_width + 2}:0] shifted_x;\n"
            verilog_str += f"{f_str}assign shifted_x = " + r"{2'b0, a, 1'b0};" + "\n"
            for encoder_index in range(encoder_num):
                center_index = 2 * encoder_index + 1
                verilog_str += (
                    f"\n{f_str}wire [{self.bit_width}:0] refined_y_{encoder_index};\n"
                )
                verilog_str += f"{f_str}wire sgn_{encoder_index};\n"
                verilog_str += (
                    f"{f_str}BoothEncoder booth_selector_{encoder_index}"
                    + f"(.y(b), "
                    + f".x(shifted_x[{center_index + 1}:{center_index - 1}]), "
                    + f".pp(refined_y_{encoder_index}), .sgn(sgn_{encoder_index}));\n"
                )
            for column_index in range(len(initial_pp)):
                verilog_str += f"{f_str}wire [{initial_pp[column_index] - 1}:0] pp_{column_index};\n"

            for column_index in range(self.bit_width):
                if column_index % 2 == 0:
                    for pp_index in range(int(initial_pp[column_index]) - 1):
                        encoder_index = pp_index
                        bit_pos = column_index - 2 * pp_index
                        verilog_str += f"{f_str}assign pp_{column_index}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
                    sgn_index = column_index // 2
                    verilog_str += f"{f_str}assign pp_{column_index}[{initial_pp[column_index] - 1}] = sgn_{sgn_index};\n"
                else:
                    for pp_index in range(int(initial_pp[column_index])):
                        encoder_index = pp_index
                        bit_pos = column_index - 2 * pp_index
                        verilog_str += f"{f_str}assign pp_{column_index}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
            for pp_index in range(int(initial_pp[self.bit_width])):
                encoder_index = pp_index
                bit_pos = self.bit_width - 2 * pp_index
                verilog_str += f"{f_str}assign pp_{self.bit_width}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
            for pp_index in range(int(initial_pp[self.bit_width + 1]) - 1):
                encoder_index = pp_index + 1
                bit_pos = self.bit_width - 1 - 2 * pp_index
                verilog_str += f"{f_str}assign pp_{self.bit_width + 1}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
            verilog_str += f"{f_str}assign pp_{self.bit_width + 1}[{initial_pp[self.bit_width + 1] - 1}] = sgn_0;\n"
            for pp_index in range(int(initial_pp[self.bit_width + 2]) - 1):
                encoder_index = pp_index + 1
                bit_pos = self.bit_width - 2 * pp_index
                verilog_str += f"{f_str}assign pp_{self.bit_width + 2}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
            verilog_str += f"{f_str}assign pp_{self.bit_width + 2}[{initial_pp[self.bit_width + 2] - 1}] = sgn_0;\n"
            for pp_index in range(int(initial_pp[self.bit_width + 3]) - 2):
                encoder_index = pp_index + 2
                bit_pos = self.bit_width - 1 - 2 * pp_index
                verilog_str += f"{f_str}assign pp_{self.bit_width + 3}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
            verilog_str += f"{f_str}assign pp_{self.bit_width + 3}[{initial_pp[self.bit_width + 3] - 1}] = ~sgn_0;\n"
            verilog_str += f"{f_str}assign pp_{self.bit_width + 3}[{initial_pp[self.bit_width + 3] - 2}] = ~sgn_1;\n"

            for column_index in range(self.bit_width + 4, len(initial_pp)):
                if column_index % 2 == 0:
                    for pp_index in range(initial_pp[column_index] - 1):
                        encoder_index = (
                            (column_index - (self.bit_width + 4)) // 2 + 2 + pp_index
                        )
                        bit_pos = self.bit_width - 2 * pp_index
                        verilog_str += f"{f_str}assign pp_{column_index}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
                    verilog_str += f"{f_str}assign pp_{column_index}[{initial_pp[column_index] - 1}] = 1'b1;\n"
                else:
                    for pp_index in range(initial_pp[column_index] - 1):
                        encoder_index = (
                            (column_index - (self.bit_width + 4)) // 2 + 3 + pp_index
                        )
                        bit_pos = self.bit_width - 1 - 2 * pp_index
                        verilog_str += f"{f_str}assign pp_{column_index}[{pp_index}] = refined_y_{encoder_index}[{bit_pos}];\n"
                    sgn_index = (column_index - (self.bit_width + 4)) // 2 + 2
                    verilog_str += f"{f_str}assign pp_{column_index}[{initial_pp[column_index] - 1}] = ~sgn_{sgn_index};\n"
        else:
            raise NotImplementedError

        return verilog_str

    def emit_verilog(
        self,
        rtl_path=None,
        dec_ct32=None,
        dec_ct22=None,
        router=None,
        assignment=None,
        prefix_adder=None,
        **kwargs,
    ):
        verilog_src = ""
        verilog_src += f"module MUL(\n"
        verilog_src += f"    input wire clk,\n"
        verilog_src += f"    input wire [{self.bit_width - 1}:0] a,\n"
        verilog_src += f"    input wire [{self.bit_width - 1}:0] b,\n"
        verilog_src += f"    output wire [{len(self.initial_pp) - 1}:0] out\n"
        verilog_src += f");\n"

        verilog_src += self.emit_pp_encoder()

        verilog_src += f"    CompressorTree ct(\n"
        for column_index in range(len(self.initial_pp)):
            verilog_src += f"        .pp_{column_index}(pp_{column_index}),\n"
        verilog_src += f"        .out(out)\n"
        verilog_src += f"    );\n"

        verilog_src += f"endmodule\n"

        verilog_src += "\n"
        if self.encode_type == "booth":
            verilog_src += self.emit_booth_selector()
            verilog_src += "\n"

        if assignment is not None:
            if isinstance(assignment, dict):
                verilog_src += self.ct.emit_verilog_from_dict(
                    assignment=assignment, prefix_adder=prefix_adder
                )
            else:
                verilog_src += self.ct.emit_verilog_fused_assignment(
                    assignment=assignment, prefix_adder=prefix_adder
                )
        else:
            verilog_src += self.ct.emit_verilog(
                dec_ct32=dec_ct32,
                dec_ct22=dec_ct22,
                router=router,
                prefix_adder=prefix_adder,
                **kwargs,
            )
        verilog_src += "\n"
        verilog_src += FA_verilog_src
        verilog_src += HA_verilog_src
        verilog_src += FA_no_carry_verilog_src
        verilog_src += HA_no_carry_verilog_src

        if rtl_path is not None:
            os.makedirs(os.path.dirname(rtl_path), exist_ok=True)
            with open(rtl_path, "w") as f:
                f.write(verilog_src)
        return verilog_src

    def emit_booth_selector(self) -> str:
        booth_selector = (
            f"""
module BoothEncoder (
    y,
    x,
    pp,
    sgn
);
    parameter bitwidth = {self.bit_width};

    input wire[bitwidth - 1: 0] y;
    input wire[3 - 1: 0]x;
            
    output wire sgn;
    output wire[bitwidth: 0] pp;

    wire[bitwidth: 0] y_extend;
    wire[bitwidth: 0] y_extend_shifted;
"""
            + r"""

    assign y_extend = {1'b0, y};
    assign y_extend_shifted = {y, 1'b0};
"""
            + f"""

    wire single, double, neg;

    assign single = x[0] ^ x[1];
    assign double = (x[0] & x[1] & (~x[2])) | ((~x[0]) & (~x[1]) & x[2]);
    assign neg = x[2];

    wire[bitwidth: 0] single_extend, double_extend, neg_extend;

    genvar i;
    generate
    for (i = 0; i < {self.bit_width} + 1; i = i + 1) begin : bit_assign
        assign single_extend[i] = single;
        assign double_extend[i] = double;
        assign neg_extend[i] = neg;
    end
    endgenerate

    assign pp = neg_extend ^ ((single_extend & y_extend) | (double_extend & y_extend_shifted));
    assign sgn = neg;

endmodule
"""
        )
        return booth_selector

    @staticmethod
    def simulate_worker(
        worker_path, rtl_path, target_delay, worker_id, keep_files=False
    ):
        os.makedirs(worker_path, exist_ok=True)
        yosys_script_path = os.path.join(worker_path, f"yosys.ys")
        sta_script_path = os.path.join(worker_path, f"sta.tcl")
        netlist_path = os.path.join(worker_path, f"netlist.v")
        constr_path = os.path.join(worker_path, f"constr.sdc")
        yosys_out_path = os.path.join(worker_path, f"yosys_out.log")
        sta_out_path = os.path.join(worker_path, f"sta_out.log")

        yosys_script = yosys_script_template.format(
            rtl_path=rtl_path,
            liberty_path=lib_path,
            target_delay=target_delay,
            constr_path=constr_path,
            netlist_path=netlist_path,
        )
        with open(yosys_script_path, "w") as f:
            f.write(yosys_script)
        sta_script = sta_script_template.format(
            lef_path=lef_path,
            lib_path=lib_path,
            verilog_path=netlist_path,
        )
        with open(sta_script_path, "w") as f:
            f.write(sta_script)
        with open(constr_path, "w") as f:
            f.write(abc_constr)

        os.system(f"yosys {yosys_script_path} > {yosys_out_path}")
        os.system(f"openroad  {sta_script_path} > {sta_out_path}")

        with open(sta_out_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                if len(words) > 0:
                    if words[0] == "wns":
                        # CRITICAL: we set 0.2 * 5 input delay in sta script
                        delay = float(words[1]) - 1
                    if words[0] == "Design":
                        area = float(words[2])
                    if words[0] == "Total":
                        power = float(words[-2])

        if not keep_files:
            os.remove(yosys_script_path)
            os.remove(sta_script_path)
            os.remove(netlist_path)
            os.remove(constr_path)
            os.remove(yosys_out_path)
            os.remove(sta_out_path)

        return {
            "delay": delay,
            "area": area,
            "power": power,
            "target_delay": target_delay,
            "worker_id": worker_id,
        }

    def simulate(
        self,
        build_path,
        rtl_path,
        target_delay_list,
        n_processing=1,
        synth="openroad",
        keep_files=False,
    ):

        params = [
            (
                os.path.join(build_path, f"worker_{i}"),
                rtl_path,
                target_delay,
                i,
                keep_files,
            )
            for i, target_delay in enumerate(target_delay_list)
        ]
        if synth == "openroad":
            worker = self.simulate_worker
        else:
            raise ValueError(f"Unknown synthesis tool: {synth}")

        if n_processing == 1:
            result_list = []
            for param in params:
                result = worker(*param)
                result_list.append(result)
        else:
            with multiprocessing.Pool(processes=n_processing) as pool:
                result = pool.starmap_async(worker, params)
                pool.close()
                pool.join()
                result_list = result.get()
            result_list.sort(key=lambda x: x["worker_id"])
        return result_list


class Mac(Mul):
    def emit_pp_encoder(self) -> str:
        """
        and encoding
              _____ a[x] & b[y], x + y = column_index
              |
              o o o o  -> a[0]
            o o o o    -> a[1]
          o o o o      -> a[2]
        o o o o        -> a[3]
            |___ a[.] not stating from a[0] after here

        booth encoding:
        5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
        ================================
                n s s o o o o o o o o o
              1 n o o o o o o o o o   s
          1 n o o o o o o o o o   s
        n o o o o o o o o o   s
        o o o o o o o o   s
        |_____| |_____| |_____________|
          s3     s2          s1

        booth encoder:
        0 9 8 7 6 5 4 3 2 1 0
        ======================
        [ 4 ]   [ 2 ]   [ 0 ]
        . . o o o o o o o o .
            [ 3 ]   [ 1 ]
                      |____center index of encoder
        '.': zeros
        'o': bits of input x
        '[*]': encoders
        """
        f_str = "    "
        verilog_str = f"{f_str}// pp_encoder\n"
        initial_pp = get_initial_partial_product(
            self.bit_width, self.encode_type
        ).astype(int)
        if self.encode_type == "and_mac":
            for column_index in range(len(initial_pp)):
                verilog_str += f"{f_str}wire [{int(initial_pp[column_index]) - 1}:0] pp_{column_index};\n"
            verilog_str += "\n"
            for column_index in range(len(initial_pp)):
                if column_index < self.bit_width:
                    for pp_index in range(int(initial_pp[column_index]) - 1):
                        offset = max(0, column_index - self.bit_width + 1)
                        verilog_str += f"{f_str}assign pp_{column_index}[{pp_index}] = a[{pp_index + offset}] & b[{column_index - pp_index - offset}];\n"
                    verilog_str += f"{f_str}assign pp_{column_index}[{int(initial_pp[column_index]) - 1}] = c[{column_index}];\n"
                else:
                    for pp_index in range(int(initial_pp[column_index])):
                        offset = max(0, column_index - self.bit_width + 1)
                        verilog_str += f"{f_str}assign pp_{column_index}[{pp_index}] = a[{pp_index + offset}] & b[{column_index - pp_index - offset}];\n"
        else:
            raise NotImplementedError

        return verilog_str

    def emit_verilog(
        self,
        rtl_path=None,
        dec_ct32=None,
        dec_ct22=None,
        router=None,
        assignment=None,
        prefix_adder=None,
        **kwargs,
    ):
        verilog_src = ""
        verilog_src += f"module MUL(\n"
        verilog_src += f"    input wire clk,\n"
        verilog_src += f"    input wire [{self.bit_width - 1}:0] a,\n"
        verilog_src += f"    input wire [{self.bit_width - 1}:0] b,\n"
        verilog_src += f"    input wire [{self.bit_width - 1}:0] c,\n"
        verilog_src += f"    output wire [{len(self.initial_pp) - 1}:0] out\n"
        verilog_src += f");\n"

        verilog_src += self.emit_pp_encoder()

        verilog_src += f"    CompressorTree ct(\n"
        for column_index in range(len(self.initial_pp)):
            verilog_src += f"        .pp_{column_index}(pp_{column_index}),\n"
        verilog_src += f"        .out(out)\n"
        verilog_src += f"    );\n"

        verilog_src += f"endmodule\n"

        verilog_src += "\n"

        if assignment is not None:
            if isinstance(assignment, dict):
                verilog_src += self.ct.emit_verilog_from_dict(
                    assignment=assignment, prefix_adder=prefix_adder
                )
            else:
                verilog_src += self.ct.emit_verilog_fused_assignment(
                    assignment=assignment, prefix_adder=prefix_adder
                )
        else:
            verilog_src += self.ct.emit_verilog(
                dec_ct32=dec_ct32,
                dec_ct22=dec_ct22,
                router=router,
                prefix_adder=prefix_adder,
                **kwargs,
            )
        verilog_src += "\n"
        verilog_src += FA_verilog_src
        verilog_src += HA_verilog_src
        verilog_src += FA_no_carry_verilog_src
        verilog_src += HA_no_carry_verilog_src

        if rtl_path is not None:
            os.makedirs(os.path.dirname(rtl_path), exist_ok=True)
            with open(rtl_path, "w") as f:
                f.write(verilog_src)
        return verilog_src
