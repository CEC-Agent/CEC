import hydra

from cec.module import RobomimicImitationModule


@hydra.main(config_name="common", config_path="eval_config")
def main(cfg):
    module = RobomimicImitationModule.load_from_checkpoint(cfg.ckpt_path)
    data_module = hydra.utils.instantiate(cfg.data_module)
    env_meta = data_module.eval_env_meta
    evaluator_ = hydra.utils.instantiate(cfg.evaluator)
    evaluator = evaluator_(env_meta=env_meta)
    results = evaluator.start(module.policy)


if __name__ == "__main__":
    main()
