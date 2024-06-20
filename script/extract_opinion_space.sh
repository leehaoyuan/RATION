CUDA_VISIBLE_DEVICES=0 Snippext_public-master/python test_tag.py \
  --task hotel_tagging \
  --batch_size 128 \
  --summary_path  summary/space_summary \
  --output_path data/space_tag.json \
  --bert_path model/mixda_task_hotel_tagging_lm_bert_batch_size_32_alpha_aug_0.2_augment_op_token_repl_tfidf_run_id_0_dev.pt;
CUDA_VISIBLE_DEVICES=0 python Snippext_public-master/test_pair.py \
  --task pairing \
  --batch_size 128 \
  --output_path data/space_pair.json \
  --input_path data/space_tag.json \
  --bert_path model/mixda_task_pairing_lm_bert_batch_size_32_alpha_aug_0.2_augment_op_token_repl_tfidf_run_id_0_test.pt;
python gen_sample.py data/space_pair.json data/space_clause.json data/space_clause+pair.json