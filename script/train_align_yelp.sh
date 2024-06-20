python run_mlm.py \
    --model_name_or_path roberta-large \
    --train_file data/yelp_trainmlm.txt \
    --per_device_train_batch_size 32 \
    --do_train \
    --line_by_line \
    --output_dir model/yelp_mlm \
    --fp16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --save_strategy epoch \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --num_train_epochs 2.0 ;
python run_align.py \
    --model_name_or_path  model/yelp_mlm \
    --train_file data/train_yelp_align.json \
    --validation_file data/valid_yelp_align.json \
    --per_device_train_batch_size 32 \_
    --do_train \
    --do_eval \
    --fp16  \
    --output_dir model/rest_absaauga0neg+a2pretrain \
    --learning_rate 1e-5 \
    --save_total_limit 0 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --num_train_epochs 3.0 ;