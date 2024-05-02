# FVEval: Evaluating LLMs on Hardware Formal Verification


## Installation
Note: Current version of the repo is based on internally-hosted ADLR chat LM servers.

We plan to migrate to NVCF based inference servers:
https://confluence.nvidia.com/display/CHIPNEMOHW/ChipNemo+Inference
Current plan is to resturcture the LM inference calls based on OpenAI API.

Public-facing versions of our repository will be based on the same OpenAI API to support a wider range of models (and endpoint servers), where only the base_url and API_keys need to be modified.

```{python}
# NVIDIA internal only
pip config set global.index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple
```

```
conda create -n fveval python=3.10
conda activate fveval
pip install -r requirements.txt
pip install -e .
```
##  Running LLM generation on each task
```{python}
# Run the following commands in your conda environment (e.g. fveval)
# You can supply a list of models to test with the --models flag, with model names ;-separated
# Run with --debug to print all input and outputs to and from the LM
# Change LLM decoding temperature with the --temperature flag

# Running LM inference on the NL2SVA-machine (assertion generation from directed NL instructions) task:
python run_svagen_nl2sva.py --mode machine --models "gpt-4;mixtral-chat" 

# Running LM inference on the NL2SVA-Human (assertion generation from testbench and high-level instructions) task:
python run_svagen_nl2sva.py --mode human --models "gpt-4;mixtral-chat" 

# Running LM inference on the Design2SVA (SV testbench generation) task:
python run_svagen_design2sva.py --models "gpt-4;mixtral-chat" 
```



##  Repo Structure
Overview of the repository:
```
fv_eval/
├── fv_eval/
│   ├── evaluation.py (methods for LM response evaluation)
│   ├── fv_tool_execution.py (methods for launching FV tools, i.e. JasperGold)
│   ├── data.py (methods for input/output data processing)
│   ├── prompts_*.py  (default prompts for each subtask)
│   ├── utils.py (misc. util functions)
|
├── data_agr/ 
│   ├── helper_gen/
|       |── data/ 
│       |── rtl/
│       |── tb/ 
│       |── generate_pipelines_helpergen.py 
│       |── generate_arbitration-clouds_helpergen.py 
│       |── generate_fsm_helpergen.py 
|
├── data_svagen/ 
│   ├── design2sva/
|   |   |── data/ 
│   |   |── rtl/ 
│   |   |── tb/ 
│   |   |── generate_pipelines_design2sva.py 
│   |   |── generate_arbitration-clouds_design2sva.py 
│   |   |── generate_fsm_design2sva.py 
│   ├── nl2sva/
│       |── annotated_instructions/ 
│       |── annotated_tb/ 
│       |── data/ 
│       |── machine_tb/ 
│       |── generate_nl2sva_human.py 
│       |── generate_nl2sva_machine.py 
|
├── tool_scripts/ 
│   ├── run_jg_design2sva.tcl
│   ├── run_jg_helpergen.tcl
│   ├── run_jg_nl2sva_human.tcl
│   ├── run_jg_nl2sva_machine.tcl
│   ├── run_jg_helpergen.tcl
|
├── run_agr_helpergen.py
├── run_evaluation.py
├── run_svagen_design2sva.py
├── run_svagen_nl2sva.py
├── run_evaluation.py
|
├── setup.py
└── README.md

Note that in the public-facing versions we may not include the benchmark generation scripts (generat_*.py)
and the raw design/testbench directories, but only share the packaged dataset files (in each data/ subdirectory)

```

## Licenses
Copyright © 2024, NVIDIA Corporation. All rights reserved.
This work is made available under the Apache 2.0 License.