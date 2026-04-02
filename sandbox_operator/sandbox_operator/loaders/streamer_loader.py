class StreamerLoader:
    def discover_and_load(self, operator, config):
        """Loads UDP Heightmap Streamer(s). Runs in global namespace.

        Always loads a visual streamer on udp_port (subscribes to /sandbox/heightmap).
        If physics_udp_port is set in config, also loads a physics streamer on that port
        (subscribes to /sandbox/heightmap/physics — ultra-smooth, for water/physics sim).
        """

        cfg = config.get('streamer_loader', {})

        common = {
            'udp_ip':   cfg.get('udp_ip', '127.0.0.1'),
            'z_offset': cfg.get('sandbox_z_offset', 0.25),
            'z_range':  cfg.get('sandbox_z_range', 0.50),
        }

        # Visual streamer — composite heightmap including transient objects/hands
        operator.load_component(
            package='sandbox_components',
            plugin='sandbox_components::HeightmapStreamer',
            name='udp_streamer',
            params={
                **common,
                'udp_port':     cfg.get('udp_port', 5005),
                'source_topic': '/sandbox/heightmap',
            },
            namespace="",
            use_ipc=True
        )

        # Physics streamer — slow layer only; only launched when physics_udp_port is configured
        if 'physics_udp_port' in cfg:
            operator.load_component(
                package='sandbox_components',
                plugin='sandbox_components::HeightmapStreamer',
                name='udp_streamer_physics',
                params={
                    **common,
                    'udp_port':     cfg.get('physics_udp_port'),
                    'source_topic': '/sandbox/heightmap/physics',
                },
                namespace="",
                use_ipc=True
            )
