clear
python hw3/train_transformer_rl.py \
	experiment.name=hw3_transformer_ppo_seed0 \
	r_seed=0 \
	init_checkpoint=hw3/miniGRP_Model/miniGRP.pth \
	rl.algorithm=ppo \
	sim.task_set=libero_spatial \
	sim.eval_tasks=[9]