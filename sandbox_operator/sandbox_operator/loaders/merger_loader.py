class MergerLoader:
    def discover_and_load(self, operator, config):
        """Loads the Central Merger. Runs in global namespace."""

        cfg = config.get('merger_loader', {})

        merger_params = {
            'output_width':  cfg.get('output_res', {}).get('width', 256),
            'output_height': cfg.get('output_res', {}).get('height', 256),
            # Bilateral spatial filter (preserves sand relief edges)
            'bilateral_d':           cfg.get('bilateral_d', 7),
            'bilateral_sigma_color': cfg.get('bilateral_sigma_color', 0.005),
            'bilateral_sigma_space': cfg.get('bilateral_sigma_space', 3.0),
            # Terrain layer: non-linear exponential smoothing; tracks sand, suppresses noise
            'terrain_smoothing_factor': cfg.get('terrain_smoothing_factor', 5.0),
            'terrain_min_alpha':        cfg.get('terrain_min_alpha', 0.05),
            # Physics layer: linear EMA of terrain layer; extra-smooth for water sim
            'physics_alpha': cfg.get('physics_alpha', 0.02),
            # Fast layer: hands and placed objects
            'fast_alpha':               cfg.get('fast_alpha', 0.40),
            'fast_detection_threshold': cfg.get('fast_detection_threshold', 0.008),
            'fast_decay_alpha':         cfg.get('fast_decay_alpha', 0.10),
        }
        operator.load_component(
            package='sandbox_components',
            plugin='sandbox_components::CloudMerger',
            name='central_merger',
            params=merger_params,
            namespace="",  # Global namespace so it can easily subscribe to all devices
            use_ipc=True
        )
