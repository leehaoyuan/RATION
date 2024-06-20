CUDA_VISIBLE_DEVICES=0 python entail_review_self.py \
	--entail_model space_absaauga0neg+a2pretrain \
	--input_file ration_cand_space.pickle \
        --output entail_review_self_space.pickle ;
CUDA_VISIBLE_DEVICES=0 python predict_specificity.py \
	--model model/specificity \
	--input_file data/ration_cand_space.pickle \
        --output data/ration_cand_space.pickle ;
python gen_ration.py \
	--clique_rela \
	--div_weight 0.1 \
	--div token \
	--popu \
	--spec \
	--k 1 \
	#--k 3 \
	--input_file data/ration_cand_space.pickle \
	--entail_file data/entail_review_self_space.pickle \
	--entail_thres 0.59 \
	--output rationals/space;
############evaluation################
CUDA_VISIBLE_DEVICES=0 python eval/eval_ration_rela_div.py
	--input rationales/spacepopuspecclique_rela0.1token_3rationales.json
