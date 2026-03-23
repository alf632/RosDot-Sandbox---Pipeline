class MergerLoader:
    def discover_and_load(self, operator, config):
        """Loads the Central Merger. Runs in global namespace."""

        cfg = config.get('merger_loader', {})

        merger_params = {
            'output_width': cfg.get('output_res', {}).get('width', 600),
            'output_height': cfg.get('output_res', {}).get('height', 400),
            'temporal_smoothing_factor': cfg.get('temporal_smoothing_factor', 5.0),
            'temporal_min_alpha': cfg.get('temporal_min_alpha', 0.05)
        }
        operator.load_component(
            package='sandbox_components',
            plugin='sandbox_components::CloudMerger',
            name='central_merger',
            params=merger_params,
            namespace="",  # Global namespace so it can easily subscribe to all devices
            use_ipc=True
        )
