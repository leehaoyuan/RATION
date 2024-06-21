python clause_from_source.py data/yelp
CUDA_VISIBLE_DEVICES=0 python entail_review_sent.py \
	--entail_model rest_absaauga0neg+a2pretrain \
	--input data/yelp_clause+pair.json \
        --output data/entail_review_sent_yelp.pickle \
        --output_dim 3 ;
python gen_ration_cand.py \
	--entail_map data/entail_review_sent_yelp.pickle \
        --data data/yelp_clause+pair.json \
	--entail_idx 1 \
        --entail_thres 0.986 \
        --self_sim_thres 0.64 \
        --output data/ration_cand_yelp.pickle ;
############evaluation################
CUDA_VISIBLE_DEVICES=0 python eval/eval_ration_cand_summac.py
	--input data/ration_cand_yelp.pickle
CUDA_VISIBLE_DEVICES=0 python eval/eval_ration_cand_silh.py
	--input data/ration_cand_yelp.pickle
