class StreamerLoader:
    def discover_and_load(self, operator, config):
        """Loads the UDP Heightmap Streamer. Runs in global namespace."""

        cfg = config.get('streamer_loader', {})

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
