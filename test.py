from configuration.configuration import TrainConfiguration, deserialize_train_cfg, serialize_train_cfg, train_cfg_is_cached

cfg = TrainConfiguration()

is_cached = train_cfg_is_cached()

serialize_train_cfg(cfg)

cfg = TrainConfiguration(val_split=0.2, test_split=0.4, batch_size=5)

old_cfg = deserialize_train_cfg()
print("finish")