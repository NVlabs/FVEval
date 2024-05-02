# Analyze property files
clear -all
analyze -clear
analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva

# Elaborate design and properties
elaborate
# set design_info [get_design_info]