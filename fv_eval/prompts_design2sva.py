SVAGEN_HEADER = """You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to generate a SystemVerilog assertion for the design-under-test provided.
"""

SVAGEN_DUT_PREAMBLE = """Here is the design RTL to generate assertions for:\n"""

SVAGEN_TB_PREAMBLE = """Here is a partial testbench for you to work on:\n"""


def get_design2sva_planning_prompt (num_assertions: int = 1) -> str:
    return f"""The following are constraints you need to satisfy in completing the task:
Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.
When implementing the assertion, generate a concurrent SVA assertion and do not add code to output an error message string.
Enclose your SystemVerilog code with ```systemverilog and ```. 

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else.

Question: in words, describe {num_assertions} feature(s) of the design that should be verified.
Answer:"""

SVAGEN_MODELING_QUESTION = """Question: for the feature you listed, implement modeling code, including wires, registers, and their assignements,
that is necessary for creating assertions.
Answer:"""

def get_design2sva_question_prompt (num_assertions: int = 1) -> str:
    return f"""Question: generate a single SVA assertion for the feature you listed. 
If necessary, include any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the te  stbench.

When implementing the assertions, implement as concurrent SVA assertions and do not add code to output an error message string.
Enclose your SystemVerilog code with ```systemverilog and ```. 

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else.
Remember to output only one assertion.

Answer:"""

def get_design2sva_direct_question_prompt (num_assertions: int = 1) -> str:
    return  f"""Question: generate a single SVA assertion for the given design RTL that is most important to verify.
If necessary, produce any extra code, including wires, registers, and their assignments.

Do NOT use signals from the design RTL, only use the module input signals or internal signals you have added.
Do NOT use any 'initial' blocks. This testbench is not for running RTL simulation but for formal verification.
Do NOT instantiate the design module inside the testbench.

When implementing the assertion, generate a concurrent SVA assertion and do not add code to output an error message string.
Enclose your SystemVerilog code with ```systemverilog and ```. 

For example:
```systemverilog
logic [3:0] a, b;
assign a = c & d;
assign b = c | d;

assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Only output the code snippet and do NOT output anything else.
Remember to output only one assertion.

Answer:"""
