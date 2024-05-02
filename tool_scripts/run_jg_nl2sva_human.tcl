# JG TCL script for evaluating the NL2SVA-Human task results
# (1) elaborate testbench generated by a LM via the NL2SVA-Human benchmark
# (2) verify match of signals used in LM-generated assertion vs. the reference assertion
# (3) TODO: check assertion-to-assertion equivalence between LM-generated and reference

# Possible outcomes:
# 1. Syntax error in the testbench; this will be caught during elaboration and script will immediately exit
# 2. Success: script completes assertion signal match, prints to STDOUT the list of signals used in each assertion
# Analyze property files
clear -all
analyze -clear
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva
# Elaborate design and properties
elaborate
# set design_info [get_design_info]

set lm_assertion [get_property_list -include {name {*asrt*} type assert}]
set reference_assertion [get_property_list -include {name {*reference*} type assert}]

set lm_coi [get_fanin -transitive [lindex $lm_assertion 0]]
set ref_coi [get_fanin -transitive [lindex $reference_assertion 0]]

puts "LM_COI: $lm_coi"
puts "REF_COI: $ref_coi"