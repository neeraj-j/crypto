#CONFIG_FILE="config_1.yaml"
#CONFIG_FILE="basic_L3.yaml"
#CONFIG_FILE="cbr_L3.yaml"
#CONFIG_FILE="cbr_L4.yaml"
#CONFIG_FILE="cbr_L4_grp.yaml"   # Working
#CONFIG_FILE="cbr_L5_grp_dip.yaml"   # very high loss stopped 
#CONFIG_FILE="cbr_L6_grp_lin.yaml"  # ran out of mem no change in loss
#CONFIG_FILE="cbr_L5_grp_c128.yaml"  # loss 0.037
#CONFIG_FILE="cbr_L5_c64.yaml"  # loss 0.037
CONFIG_FILE="cbr_L5_c64_lin.yaml"  # loss 0.037
#CONFIG_FILE="cbr_L4_c32.yaml"  # 
#CONFIG_FILE="cbr_L5_c64_150k.yaml"  # loss 0.0165
#CONFIG_FILE="cbr_L7_grp.yaml"  
python tools/train.py --cfg experiments/$CONFIG_FILE
