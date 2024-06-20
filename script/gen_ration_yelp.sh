CUDA_VISIBLE_DEVICES=0 python entail_review_self.py \
	--entail_model model/rest_absaauga0neg+a2pretrain \
	--input_file data/ration_cand_yelp.pickle \
        --output data/entail_review_self_yelp.pickle ;
CUDA_VISIBLE_DEVICES=0 python predict_specificity.py \
	--model model/specificity \
	--input_file data/ration_cand_yelp.pickle \
        --output data/ration_cand_yelp.pickle ;
python gen_ration.py \
	--clique_rela \
	--div_weight 0.1 \
	--div token \
	--popu \
	--spec \
	--k 1 \
	#--k 3 \
	--input_file data/ration_cand_yelp.pickle \
	--entail_file data/entail_review_self_yelp.pickle \
	--entail_thres 0.986 \
	--output rationales/yelp;
############evaluation################
CUDA_VISIBLE_DEVICES=0 python eval/eval_ration_rela_div.py
	--input rationales/yelppopuspecclique_rela0.1token_3rationales.json