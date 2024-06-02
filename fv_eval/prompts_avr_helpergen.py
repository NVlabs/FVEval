AGR_HELPERGEN_HEADER = """You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.
Your job is to generate a helper assertion for the specified target assertion.
A helper assertion is an assertion that 
(1) is proven
(2) is more easily proven than the target assertion
(3) and can be used as an assumption to help the proof of the target assertion

"""

AGR_HELPERGEN_DUT_PREAMBLE = """Here is the design RTL:\n"""

AGR_HELPERGEN_TB_PREAMBLE = """Here is the testbench with the target assertion to generate helpers for marked as 'target':\n"""

AGR_HELPERGEN_QUESTION_COT_THOUGHT = """ Question: 
Generate a helper assertion for the specified target assertion.

First, provide reasoning for what assertion should be generated as a helper.

Thought:"""

AGR_HELPERGEN_QUESTION_COT_ANSWER = """ 
Next, generate a single helper assertion as a concurrent SVA assertion that you believe is most appropriate.

When implementing the helper assertions, implement as concurrent SVA assertions, for example:
helper: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
Do not add code to output an error message string.

Enclose your SVA code with ```systemverilog and ```. ONLY output the code snippet and do NOT output anything else.
Answer:"""
