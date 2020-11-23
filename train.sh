#CONFIG_FILE="config_1.yaml"
#CONFIG_FILE="basic_L3.yaml"
#CONFIG_FILE="cbr_L3.yaml"
#CONFIG_FILE="cbr_L4.yaml"
#CONFIG_FILE="cbr_L4_grp.yaml"   # Working
#CONFIG_FILE="cbr_L5_grp_dip.yaml"   # very high loss stopped 
#CONFIG_FILE="cbr_L6_grp_lin.yaml"  # ran out of mem no change in loss
#CONFIG_FILE="cbr_L5_grp_c128.yaml"  # loss 0.037
#CONFIG_FILE="cbr_L5_c64.yaml"  # loss 0.037
#CONFIG_FILE="cbr_L5_c64_lin.yaml"  # loss 0.0755 ; stopped
#CONFIG_FILE="cbr_L4_c32.yaml"  # 
#CONFIG_FILE="cbr_L5_c64_150k.yaml"  # loss 0.0118
#CONFIG_FILE="cbr_L6_c64_150k.yaml"  # loss 
#CONFIG_FILE="cbr_L5_c64_150k_rand.yaml"  # loss 
#CONFIG_FILE="cbr_L5_c64_150k_sum.yaml"  # stopped 
#CONFIG_FILE="cbr_L5_c256_150k_sum.yaml"  # loss 0.0641
#CONFIG_FILE="cbr_L5_c128_vq.yaml"
CONFIG_FILE="aes_L5_c512.yaml"
#python tools/train.py --cfg experiments/$CONFIG_FILE
python tools/train_aes.py --cfg experiments/$CONFIG_FILE
