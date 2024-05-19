SVAGEN_HEADER = """You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to generate a SystemVerilog testbench with as many appropriate assertions given the design-under-test provided.
"""

SVAGEN_DUT_PREAMBLE = (
    """Here is the design RTL to generate assertions for:\n"""
)

SVAGEN_TB_PREAMBLE = (
    """Here is a partial testbench for you to work on:\n"""
)


SVAGEN_PLANNING_QUESTION = """The following are constraints you need to satisfy in completing the task:
Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.


Question: in words, write a list of all features of the design that should be verified.
Consider as many and diverse functional points of interest.

Answer:
"""

SVAGEN_MODELING_QUESTION = """Question: for each of the features you listed, implement modeling code, including wires, registers, and their assignements,
that is necessary for creating assertions.
Answer:
"""


SVAGEN_QUESTION = """Question: generate an SVA assertion for each of the features you listed. 
Create a single piece of code implementing the testbench module with assertions. Only output a single testbench module.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assertions, implement as concurrent SVA assertions and do not add code to output an error message string.
Enclose your SystemVerilog code with ```systemverilog and ```. 

For example:
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else.

Answer:
"""