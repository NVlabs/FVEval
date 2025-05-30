# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prompts used for NL2SVA-Human
Subject to modifications and improvements
"""

# NL2SVA-Human Question Prompts
SVAGEN_HEADER = """You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.
"""

SVAGEN_MACHINE_ICL_EXAMPLE_1 = """As an example:

Question: Create a SVA assertion that checks: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
```
"""


SVAGEN_MACHINE_ICL_EXAMPLE_2 = """As an example:

Question: Create a SVA assertion that checks: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
```

Question: Create a SVA assertion that checks: If sig_C contains at least one '1' bit or sig_D is not equal to sig_A, then sig_F must eventually be true
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (|sig_C || (sig_D !== sig_A )) |=> s_eventually(sig_F)
);
```
"""

SVAGEN_MACHINE_ICL_EXAMPLE_3 = """As an example:

Question: Create a SVA assertion that checks: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
```

Question: Create a SVA assertion that checks: If sig_C contains at least one '1' bit or sig_D is not equal to sig_A, then sig_F must eventually be true
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
    (|sig_C || (sig_D !== sig_A )) |=> s_eventually(sig_F)
);
```

Question: Create a SVA assertion that checks: "Whenever the value of sig_J is less than the result of the XOR operation between sig_C and the negation of the bitwise negation of sig_H, and this result is equal to the result of the OR operation between the identity comparison of sig_A and the negation of sig_J and sig_B, the assertion is true
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:
```systemverilog
assert property(@(posedge clk)
	((sig_J < (sig_B == (sig_C ^ ~|sig_H))) == ((|sig_A === !sig_J) || sig_B))
);
```
"""
SVAGEN_HUMAN_ICL_EXAMPLE_1 = """As an example, consider the following SystemVerilog module:

module ShiftRegister #(
  parameter Width = 4
) (
  input  logic             clk,
  input  logic             reset_,
  input  logic             shiftRight,
  input  logic             writeEnable,
  input  logic [Width-1:0] dataIn,
  output logic [Width-1:0] dataOut
);

  logic [Width-1:0] shiftRegisters;
  logic [Width-1:0] shiftRegisters_delay1;

  logic tb_reset = (reset_ == 1'b0)
  always_ff @(posedge clk, negedge reset_) begin
    if (!reset_) begin
      shiftRegisters <= '0;
    end
    else if (writeEnable) begin
        shiftRegisters <= dataIn;
    end
    else if (shiftRight) begin
      shiftRegisters <= {shiftRegisters[0], shiftRegisters[Width-1:1]};
    end
    else begin
      shiftRegisters <= {shiftRegisters[Width-2:0], shiftRegisters[Width-1]};
    end
  end
  always_ff @(posedge clk, negedge reset_) begin
    if (!reset_) begin
      shiftRegisters_delay1 <= '0;
    end
    else begin
      shiftRegisters_delay1 <= shiftRegisters;
    end
  end
  assign dataOut = shiftRegisters;
  

endmodule

Here are examples of assertions about this design.

Question: Create a SVA assertion that checks that in the same cycle that data input is written, the output matches the input.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (writeEnable && (dataIn != dataOut) !== 1'b1)
);
```
"""

SVAGEN_HUMAN_ICL_EXAMPLE_3 = """As an example, consider the following SystemVerilog module:

module ShiftRegister #(
  parameter Width = 4
) (
  input  logic             clk,
  input  logic             reset_,
  input  logic             shiftRight,
  input  logic             writeEnable,
  input  logic [Width-1:0] dataIn,
  output logic [Width-1:0] dataOut
);

  logic [Width-1:0] shiftRegisters;
  logic [Width-1:0] shiftRegisters_delay1;

  logic tb_reset = (reset_ == 1'b0)
  always_ff @(posedge clk, negedge reset_) begin
    if (!reset_) begin
      shiftRegisters <= '0;
    end
    else if (writeEnable) begin
        shiftRegisters <= dataIn;
    end
    else if (shiftRight) begin
      shiftRegisters <= {shiftRegisters[0], shiftRegisters[Width-1:1]};
    end
    else begin
      shiftRegisters <= {shiftRegisters[Width-2:0], shiftRegisters[Width-1]};
    end
  end
  always_ff @(posedge clk, negedge reset_) begin
    if (!reset_) begin
      shiftRegisters_delay1 <= '0;
    end
    else begin
      shiftRegisters_delay1 <= shiftRegisters;
    end
  end
  assign dataOut = shiftRegisters;
  

endmodule

Here are examples of assertions about this design.

Question: Create a SVA assertion that checks that in the same cycle that data input is written, the output matches the input.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (writeEnable && (dataIn != dataOut) !== 1'b1)
);
```


Question: Create a SVA assertion that checks that the total count of ones in the register is invariant to shift operations.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    !writeEnable |=> ($countones(shiftRegisters_delay1) == $countones(shiftRegisters))
);
```

Question: Create a SVA assertion that checks that the shift to right operation correctly shifts bits of the registers.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    shiftRight && !writeEnable |=> (shiftRegisters_delay1[0] == shiftRegisters[Width-1]) && (shiftRegisters_delay1[Width-1:1] == shiftRegisters[Width-2:0]) 
);
```
"""


SVAGEN_HUMAN_ICL_EXAMPLE_3_COT = """As an example, consider the following SystemVerilog module:

module ShiftRegister #(
  parameter Width = 4
) (
  input  logic             clk,
  input  logic             reset_,
  input  logic             shiftRight,
  input  logic             writeEnable,
  input  logic [Width-1:0] dataIn,
  output logic [Width-1:0] dataOut
);

  logic [Width-1:0] shiftRegisters;
  logic [Width-1:0] shiftRegisters_delay1;

  logic tb_reset = (reset_ == 1'b0)
  always_ff @(posedge clk, negedge reset_) begin
    if (!reset_) begin
      shiftRegisters <= '0;
    end
    else if (writeEnable) begin
        shiftRegisters <= dataIn;
    end
    else if (shiftRight) begin
      shiftRegisters <= {shiftRegisters[0], shiftRegisters[Width-1:1]};
    end
    else begin
      shiftRegisters <= {shiftRegisters[Width-2:0], shiftRegisters[Width-1]};
    end
  end
  always_ff @(posedge clk, negedge reset_) begin
    if (!reset_) begin
      shiftRegisters_delay1 <= '0;
    end
    else begin
      shiftRegisters_delay1 <= shiftRegisters;
    end
  end
  assign dataOut = shiftRegisters;
  

endmodule

Question: Create a SVA assertion that checks that in the same cycle that data input is written, the output matches the input.

Thought: The assertion should check that the output data matches the input data when the writeEnable signal is high.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (writeEnable && (dataIn != dataOut) !== 1'b1)
);
```


Question: Create a SVA assertion that checks that the total count of ones in the register is invariant to shift operations.

Thought: The assertion should check that unless there is a write operation, the count of ones in the register remains the same in the following cycle.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    !writeEnable |=> ($countones(shiftRegisters_delay1) == $countones(shiftRegisters))
);
```

Question: Create a SVA assertion that checks that the shift to right operation correctly shifts bits of the registers.

Thought: The assertion should check that when the shiftRight signal is high and there is no write operation, 
the first bit of shiftRegisters_delay1 should be equal to the last bit of shiftRegisters, and the remaining bits should be shifted right.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    shiftRight && !writeEnable |=> (shiftRegisters_delay1[0] == shiftRegisters[Width-1]) && (shiftRegisters_delay1[Width-1:1] == shiftRegisters[Width-2:0]) 
);
```
"""


SVAGEN_TB_PREAMBLE = """Here is the testbench to perform your translation:\n"""

SVAGEN_IC_EX_PREAMBLE = """\n\nMore detailed examples of correct translations from description into an SVA assertion:"""


# NL2SVA-Human Question Prompts
SVAGEN_QUESTION_PREAMBLE = "\nQuestion: Create a SVA assertion that checks: "

SVAGEN_QUESTION_POSTAMBLE = """
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Answer:"""

SVAGEN_QUESTION_POSTAMBLE_COT = """
First produce your thought process for what assertion to write, then provide the SVA code.
When providing the SVA implementation, do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
Thought: the assertion should check that a and b are not both high at the same time.

Answer:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Thought:"""

