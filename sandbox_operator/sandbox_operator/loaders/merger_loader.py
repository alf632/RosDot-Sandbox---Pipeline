class MergerLoader:
    def discover_and_load(self, operator, config):
        """Loads the Central Merger and the UDP Streamer. Runs in global namespace."""
        
        cfg = config.get('merger_loader', {})
        
        # 1. Load Merger
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

        # 2. Load UDP Streamer
        udp_params = {
            'udp_ip': cfg.get('udp_ip', '127.0.0.1'),
            'udp_port': cfg.get('udp_port', 5005),
            'z_offset': cfg.get('sandbox_z_offset', 0.25),
            'z_range': cfg.get('sandbox_z_range', 0.50)
        }
        operator.load_component(
            package='sandbox_components',
            plugin='sandbox_components::HeightmapStreamer',
            name='udp_streamer',
            params=udp_params,
            namespace="",
            use_ipc=True
        )
