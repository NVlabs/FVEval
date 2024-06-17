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
