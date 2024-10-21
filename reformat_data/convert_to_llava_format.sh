# pos in4 cot
python convert_to_llava_format_with_pos_cot.py --sample_rate 4 --trial_num 5 --image_num 4 --use_cot 1 --save_eval_data
# pos in8 cot
python convert_to_llava_format_with_pos_cot.py --sample_rate 4 --trial_num 5 --image_num 8 --use_cot 1 --save_eval_data
# pos in2 cot
python convert_to_llava_format_with_pos_cot.py --sample_rate 4 --trial_num 5 --image_num 2 --use_cot 1 --save_eval_data
# pos in6 cot
python convert_to_llava_format_with_pos_cot.py --sample_rate 4 --trial_num 5 --image_num 6 --use_cot 1 --save_eval_data

# pos in4 not cot
python convert_to_llava_format_with_pos_cot.py --sample_rate 4 --trial_num 5 --image_num 4 --use_cot 0 --save_eval_data

# difference-equation prompt
python convert_to_llava_format_with_pos_diff_equation.py --sample_rate 4 --trial_num 5 --image_num 4 --use_cot 1

# gold label of diff in4 cot
python convert_to_llava_format_with_pos_with_gold_label_diff.py --sample_rate 4 --trial_num 5 --image_num 4 --use_cot 1
