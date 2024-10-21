sh train_llava_instruct_webdataset_cot.sh 4 1152 3660 110 cot_stp8_in4 cot
sh train_llava_instruct_webdataset_cot.sh 4 1152 3660 110 no_cot_stp8_in4 no_cot

# cot step number
sh train_llava_instruct_webdataset_cot.sh 4 1024 3660 110 cot_stp4_in4 cot-stp4-in4-tn5-sr0.25

sh train_llava_instruct_webdataset_cot.sh 4 1280 3660 110 cot_stp12_in4 cot-stp12-in4-tn5-sr0.25

sh train_llava_instruct_webdataset_cot.sh 4 1536 3660 110 cot_stp16_in4 cot-stp16-in4-tn5-sr0.25

# cot image number 
sh train_llava_instruct_webdataset_cot.sh 2 1024 3660 110 cot_stp8_in2 cot-in2-tn5-sr0.25
sh train_llava_instruct_webdataset_cot.sh 6 1280 3660 110 cot_stp8_in6 cot-in6-tn5-sr0.25
sh train_llava_instruct_webdataset_cot.sh 8 1536 3660 110 cot_stp8_in8 cot-in8-tn5-sr0.25

# cot nd few shot
sh train_llava_instruct_webdataset_cot.sh 4 1152 731 22 cot_stp8_in4_tn1 cot-in4-tn1-sr0.25
sh train_llava_instruct_webdataset_cot.sh 4 1152 1462 44 cot_stp8_in4_tn2 cot-in4-tn2-sr0.25
sh train_llava_instruct_webdataset_cot.sh 4 1152 2195 66 cot_stp8_in4_tn3 cot-in4-tn3-sr0.25
sh train_llava_instruct_webdataset_cot.sh 4 1152 2925 88 cot_stp8_in4_tn4 cot-in4-tn4-sr0.25
