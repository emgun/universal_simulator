.PHONY: test-ddp-oom

test-ddp-oom:
	UPS_SIMULATE_OOM_RANK=1 UPS_SIMULATE_OOM_STEP=0 torchrun --nproc_per_node=2 --nnodes=1 --master_addr=localhost --master_port=29501 scripts/test_ddp_minimal.py --simulate-oom-steps=1
