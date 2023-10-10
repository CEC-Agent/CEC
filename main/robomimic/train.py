import hydra


@hydra.main(config_name="common", config_path="train_config")
def main(cfg):
    trainer = hydra.utils.instantiate(cfg.trainer)
    module = hydra.utils.instantiate(cfg.module)
    data_module = hydra.utils.instantiate(cfg.data_module)
    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()
