python clause_from_source.py data/space
CUDA_VISIBLE_DEVICES=0 python entail_review_sent.py \
	--entail_model space_absaauga0neg+a2pretrain \
	--input data/space_clause+pair.json \
        --output data/entail_review_sent_space.pickle \
        --output_dim 3 ;
python gen_ration_cand.py \
	--entail_map data/entail_review_sent_space.pickle \
        --data data/space_clause+pair.json \
	--entail_idx 1 \
        --entail_thres 0.59 \
        --self_sim_thres 0.47 \
        --output data/ration_cand_space.pickle;
############evaluation################
CUDA_VISIBLE_DEVICES=0 python eval/eval_ration_cand_summac.py
	--input data/ration_cand_space.pickle
CUDA_VISIBLE_DEVICES=0 python eval/eval_ration_cand_silh.py
	--input data/ration_cand_space.pickle