clear -all
check_bps -init
analyze -clear
 
analyze -sv12 data_svagen/design2sva/fifo_1r1w_ref.sv
analyze -sv12 data_svagen/design2sva/lm_gen.sv

elaborate
# set design_info [get_design_info]
clock clk;
reset -expression (fifo_1r1w_tb_inst.tb_reset)
assume -enable {*}[get_property_list -include {type assume}]
prove -all -engine_mode Hp 
set lm_assertion [get_property_list -include {name {*asrt*} type assert}]
set reference_assertion [get_property_list -include {name {*reference*} type assert}]
