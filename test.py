from configuration.data_configuration import DataConfiguration, deserialize_data_cfg, serialize_data_cfg, data_cfg_is_cached
cfg = DataConfiguration(
            symbols=['symbols'],
            start="2020-12-06",
            end="2021-04-06",
            feedback_metrics=["open", "close", "high", "low", "vwap"],
            stock_news_fetch_limit=500
        )

is_cached = data_cfg_is_cached()

serialize_data_cfg(cfg)

cfg = DataConfiguration(
            symbols=['symbols'],
            start="2020-12-06",
            end="2021-05-06",
            feedback_metrics=["open", "close", "high", "low", "vwap"],
            stock_news_fetch_limit=500
        )

old_cfg = deserialize_data_cfg()
print("finish")