CUDA_VISIBLE_DEVICES=0 Snippext_public-master/python test_tag.py \
  --task restaurant_tagging \
  --batch_size 128 \
  --summary_path  conv_summary/yelp_summary \
  --output_path data/yelp_tag.json \
  --bert_path model/mixda_task_restaurant_tagging_lm_bert_batch_size_32_alpha_aug_0.2_augment_op_token_repl_tfidf_run_id_0_dev.pt;
CUDA_VISIBLE_DEVICES=0 python Snippext_public-master/test_pair.py \
  --task restaurant_pairing \
  --batch_size 128 \
  --output_path data/yelp_pair.json \
  --input_path data/yelp_tag.json \
  --bert_path model/mixda_task_restaurant_pairing_lm_bert_batch_size_32_alpha_aug_0.2_augment_op_token_repl_tfidf_run_id_0_test.pt;
python SemAE-main/src/inference_absa.py \
        --summary_data data/yelp.json \
        --gold_data SemAE-main/data/space/gold \
	--pair_data data/yelp_pairall.json
        --sentencepiece model/yelp_restv3.model \
        --model model/yelp_restv3_run1_9_model.pt \
        --gpu 0 \
        --run_id yelp_summary \
        --outdir conv_summary \
        --max_tokens 100 \
        --cos_thres 0.5 \
	--min_tokens 2 \
	--no_cut_sents;