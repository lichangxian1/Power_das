import os

# Change your library and LEF paths accordingly
lib_path = os.path.expanduser("~/library/t28_official/tcbn28hpcplusbwp12t40p140tt0p9v25c.lib")
lef_path = os.path.expanduser("~/library/t28_official/tcbn28hpcplusbwp12t40p140.lef")


verilate_header_template = """
#include "VMUL.h"
#include "verilated.h"
#include <iostream>
#include <memory.h>
#include <memory>
#include <random>

#define INPUT_WIDTH {input_width}
#define TEST_INPUT_WIDTH {test_input_width}
#define OUTPUT_WIDTH {output_width}
"""

verilate_main_template = r"""
int main()
{
    auto top = std::make_shared< VMUL >();
    bool                                      flag = true;
    int                                       cnt  = 0;
    std::random_device                        rd;
    std::mt19937                              gen( rd() );
    std::uniform_int_distribution< uint32_t > dis( 0,
                                                   ( ( ( long long )1 ) << TEST_INPUT_WIDTH ) - 1 );
    for ( long long i = 0; i < 1e6; i += 1 )
    {
        cnt += 1;

        unsigned long long a = dis( gen );
        unsigned long long b = dis( gen );
        top->a               = a;
        top->b               = b;
        top->eval();

        unsigned long long out          = top->out;
        unsigned long long ground_truth = ( ( a * b ) & ( ( ( long )1 << OUTPUT_WIDTH ) - 1 ) );
        if ( out != ground_truth )
        {
            flag = false;
            printf( "a = %lld, b = %lld, output out is %lld, true value is "
                    "%lld\n",
                    a, b, out, ground_truth );
        }
    }

    if ( flag )
    {
        std::cout << "All " << cnt << " tests passed" << std::endl;
    }
    return 0;
}
"""


cmakelists_head_template = r"""
cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)
project(cmake_hello_c)

find_package(verilator HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
if(NOT verilator_FOUND)
    message(
        FATAL_ERROR
        "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable"
    )
endif()
"""


cmakelists_main_template = """
# Create a new executable target that will contain all your sources
add_executable(mul_verilate {source_path})
target_compile_features(mul_verilate PUBLIC cxx_std_14)

# Add the Verilated circuit to the target
verilate(mul_verilate
  INCLUDE_DIRS "verilate"
  SOURCES {rtl_path}
)
"""


abc_constr = """
set_driving_cell BUF_X1
set_load 10.0 [all_outputs]
"""

sta_script_template = """
read_lef {lef_path}
read_lib {lib_path}
read_verilog {verilog_path}
link_design MUL

set period 5
create_clock -period $period [get_ports clk]

set clk_period_factor .2

set clk [lindex [all_clocks] 0]
set period [get_property $clk period]
set delay [expr $period * $clk_period_factor]

set all_paths [find_timing_paths]
puts "Number of paths found: [llength $all_paths]"

set_input_delay $delay -clock $clk [delete_from_list [all_inputs] [all_clocks]]
set_output_delay $delay -clock $clk [delete_from_list [all_outputs] [all_clocks]]

set_max_delay -from [all_inputs] 0
set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
set path_delay [sta::format_time [[$critical_path path] arrival] 4]
puts "wns $path_delay"
report_design_area

set_power_activity -input -activity 0.5
report_power

exit
"""

yosys_script_template = """
read -sv {rtl_path}
synth -top MUL
dfflibmap -liberty {liberty_path}
abc -D {target_delay} -constr {constr_path} -liberty {liberty_path}
write_verilog {netlist_path}
"""

FA_verilog_src = """
module FA (a, b, cin, sum, cout);
    input a;
    input b;
    input cin;
    output sum;
    output cout;
    wire  a_xor_b = a ^ b; 
    wire  a_and_b = a & b; 
    wire  a_and_cin = a & cin; 
    wire  b_and_cin = b & cin; 
    wire  _T_1 = a_and_b | b_and_cin;
    assign sum = a_xor_b ^ cin;
    assign cout = _T_1 | a_and_cin; 
endmodule
"""

FA_no_carry_verilog_src = """
module FA_no_carry (a, b, cin, sum);
    input a;
    input b;
    input cin;
    output sum;

    wire  a_xor_b = a ^ b; 
    assign sum = a_xor_b ^ cin;
endmodule
"""

HA_verilog_src = """
module HA (a, cin, sum, cout);
    input a;
    input cin;
    output sum;
    output cout;
    assign sum = a ^ cin; 
    assign cout = a & cin; 
endmodule
"""

HA_no_carry_verilog_src = """
module HA_no_carry (a, cin, sum);
    input a;
    input cin;
    output sum;

    assign sum = a ^ cin; 
endmodule
"""
