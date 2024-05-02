# JG TCL script for evaluating the HelperGen task results
# (1) elaborate testbench
# (2) run formal proofs of target, helper, and target with helper assumed
# (3) measure formal coverage via the Japser Coverage App
# Possible outcomes:
# 1. Syntax error in the testbench; this will be caught during elaboration and script will immediately exit
# 2. Proof failure: some of the LM generated assertions will be proven false; script proceeds to formal coverage analysis
# 3. Succss: script completes formal coverage analysis, prints to STDOUT formal coverage results

# Analyze property files
clear -all
analyze -clear
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sv
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva

# Elaborate design and properties
elaborate
set top [get_inst_top]
puts "top: $top"
set_reset_max_iterations 1000
clock clk
reset -expression (${top}_tb_inst.tb_reset)

set target [get_property_list -include {name {*target*} type assert}]
set helpers [get_property_list -exclude {name {*target*}} -include {type assert}]
prove -all -engine_mode Hp -time_limit 1m

puts "helper_proofs: [get_status $helpers]"

# set design_info [get_design_info]