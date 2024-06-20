CUDA_VISIBLE_DEVICES=0 Snippext_public-master/python test_tag_json.py \
  --task hotel_tagging \
  --batch_size 128 \
  --summary_path  data/space.json \
  --output_path data/space_tagall.json \
  --bert_path model/mixda_task_hotel_tagging_lm_bert_batch_size_32_alpha_aug_0.2_augment_op_token_repl_tfidf_run_id_0_dev.pt;
CUDA_VISIBLE_DEVICES=0 python Snippext_public-master/test_pair.py \
  --task pairing \
  --batch_size 128 \
  --output_path data/space_pairall.json \
  --input_path data/space_tagall.json \
  --bert_path model/mixda_task_pairing_lm_bert_batch_size_32_alpha_aug_0.2_augment_op_token_repl_tfidf_run_id_0_test.pt;
python SemAE-main/src/inference_absa.py \
        --summary_data data/space.json \
        --gold_data SemAE-main/data/space/gold \
	--pair_data data/space_pairall.json
        --sentencepiece model/spm_unigram_32k.model \
        --model model/spacev2_7_model.pt \
        --gpu 0 \
        --run_id space_summary \
        --outdir conv_summary \
        --max_tokens 100 \
        --cos_thres 0.5 \
	--min_tokens 2 \
	--no_cut_sents;