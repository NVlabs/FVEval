clear -all
check_bps -init
analyze -clear
 
analyze -sv12 data_agr/arbiter_rr/arbiter_rr_ref.sv
analyze -sv12 data_agr/arbiter_rr/arbiter_rr_tb.sv
elaborate
# set design_info [get_design_info]
clock clk;
reset -expression (arbiter_rr_tb_inst.tb_reset)
assume -enable {*}[get_property_list -include {type assume}]
prove -all -engine_mode Hp 
set lm_assertion [get_property_list -include {name {*asrt*} type assert}]
set reference_assertion [get_property_list -include {name {*reference*} type assert}]
