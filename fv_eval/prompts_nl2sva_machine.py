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
Prompts used for NL2SVA-Machine
Subject to modifications and improvements
"""

# NL2SVA-Machine Question Prompts
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


SVAGEN_MACHINE_ICL_EXAMPLE_3_COT = """As an example:

Question: Create a SVA assertion that checks: Whenever sig_A is high and sig_B is low, sig_C will be high on the next clock edge.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Thought: the assertion should check that sig_C is high on the next clock edge whenever sig_A is high and sig_B is low.
We can express signal behavior on the next clock edge using the implication operator |->.
The assertion can be written as (sig_A && !sig_B) |-> sig_C.

Answer:
```systemverilog
assert property(@(posedge clk)
    (sig_A && !sig_B) |-> sig_C
);
```

Question: Create a SVA assertion that checks: If sig_C contains at least one '1' bit or sig_D is not equal to sig_A, then sig_F must eventually be true
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Thought: the assertion should check that if sig_C contains at least one '1' bit or sig_D is not equal to sig_A, then sig_F must eventually be true.
The implication operator |=> along with s_eventually can be used to express eventual behavior. 
The assertion can be written as (|sig_C || (sig_D !== sig_A )) |=> s_eventually(sig_F).

Answer:
```systemverilog
assert property(@(posedge clk)
    (|sig_C || (sig_D !== sig_A )) |=> s_eventually(sig_F)
);
```

Question: Create a SVA assertion that checks: Whenever the value of sig_J is less than the result of the XOR operation between sig_C and the negation of the bitwise negation of sig_H, and this result is equal to the result of the OR operation between the identity comparison of sig_A and the negation of sig_J and sig_B.
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Thought: the assertion can be decomposed to two parts. 
First, the we should check that the value of sig_J is less than the result of the XOR operation between sig_C and the negation of the bitwise negation of sig_H.
This can be expressed as sig_J < (sig_B == (sig_C ^ ~|sig_H)).
Next, we should check the OR operation between the identity comparison of sig_A and the negation of sig_J and sig_B.
This can be expressed as (|sig_A === !sig_J) || sig_B.
Finally, we should check that the two results are equal.

Answer:
```systemverilog
assert property(@(posedge clk)
	((sig_J < (sig_B == (sig_C ^ ~|sig_H))) == ((|sig_A === !sig_J) || sig_B))
);
```
"""

SVAGEN_TB_PREAMBLE = """Now here is the testbench to perform your translation:\n"""

SVAGEN_IC_EX_PREAMBLE = """\n\nMore detailed examples of correct translations from description into an SVA assertion:"""


# NL2SVA-Machine Question Prompts
SVAGEN_QUESTION_PREAMBLE = "\nQuestion: Create a SVA assertion that checks: "
SVAGEN_QUESTION_POSTAMBLE = """
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Answer:"""

SVAGEN_QUESTION_POSTAMBLE_ZERO_SHOT = """
Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
assert property (@(posedge clk)
    (sig_A && sig_B) != 1'b1
);
```
Answer:"""

SVAGEN_QUESTION_POSTAMBLE_COT = """
First produce your thought process for what assertion to write, then provide the SVA code.
When providing the SVA implementation, do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

Thought:"""

