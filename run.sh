CUDA_VISIBLE_DEVICES=0 python main_incremental.py --exp-name nc_first_50_ntask_6 \
	--datasets cifar100_icarl --num-tasks 6 --nc-first-task 50 --network resnet18_cifar --seed 1993 \
	--nepochs 160 --batch-size 128 --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --decay-mile-stone 80 120 \
	--clipping -1 --results-path results --save-models \
	--approach lucir_cwd_BDR --lamb 5.0 --num-exemplars-per-class 20 --exemplar-selection herding \
	--aux-coef 0.5 --reject-threshold 1 --dist 0.5 \
	--cwd --BDR --m1 0.8 --m2 0.8\