# cot in4
python hf_idefics2_online_client.py --split valid --save_path YOUR_PATH \
--freq_refresh 30 --image_num 4 --action_his_num 8 --use_cot 1 --start_idx 0 --end_idx 27 --no_in_diff 1 --trial_list 0,1,2,3 \
--ip 11.255.125.249 --port 80

# GPT-4V
python hf_idefics2_gpt4o.py --split valid --save_path YOUR_PATH \
--freq_refresh 30 --image_num 4 --action_his_num 8 --use_cot 1 --start_idx 0 --end_idx 27 --trial_list 0,1,2,3

# GPT-4o
python hf_idefics2_gpt4o.py --split valid --save_path YOUR_PATH \
--freq_refresh 30 --image_num 4 --action_his_num 8 --model_name gpt-4o --use_cot 1 --start_idx 0 --end_idx 27 --trial_list 0,1,2,3
