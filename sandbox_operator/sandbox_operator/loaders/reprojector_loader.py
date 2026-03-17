class ReprojectorLoader:
    def discover_and_load(self, operator, config):
        """Loads a single Reprojector Manager for this device."""
        
        cfg = config.get('repro_loader', {})
        
        params = {
            'sandbox_width': cfg.get('sandbox', {}).get('width', 4.0),
            'sandbox_length': cfg.get('sandbox', {}).get('length', 6.0),
            'output_width': cfg.get('output_res', {}).get('width', 600),
            'output_height': cfg.get('output_res', {}).get('height', 400),
            # Pass the namespace so the C++ node knows its local domain
            'target_namespace': operator.device_namespace 
        }

        operator.load_component(
            package='sandbox_components',
            plugin='sandbox_components::CameraReprojector',
            name='local_reprojector',
            params=params,
            namespace=operator.device_namespace,
            use_ipc=True
        )
