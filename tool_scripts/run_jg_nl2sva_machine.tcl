# Analyze property files
clear -all
analyze -clear
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva

# Elaborate design and properties
elaborate
# set design_info [get_design_info]

clear -all
include tool_scripts/pec/pec.tcle
set signal_list [split $SIGNAL_LIST ","]
prop_eq_checker $LM_ASSERT_TEXT $REF_ASSERT_TEXT "" "" $signal_list