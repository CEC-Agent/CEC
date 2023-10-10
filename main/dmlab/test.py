import hydra

from cec.module import DMLabImitationModule


@hydra.main(config_name="common", config_path="eval_config")
def main(cfg):
    module = DMLabImitationModule.load_from_checkpoint(cfg.ckpt_path)
    evaluator = hydra.utils.instantiate(cfg.evaluator)
    results = evaluator.start(module.policy)


if __name__ == "__main__":
    main()
