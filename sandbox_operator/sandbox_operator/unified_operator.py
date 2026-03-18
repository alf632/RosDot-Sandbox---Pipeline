import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import socket
from composition_interfaces.srv import LoadNode
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

# Import Loaders
from .loaders.realsense_loader import RealSenseLoader
from .loaders.reprojector_loader import ReprojectorLoader
from .loaders.merger_loader import MergerLoader
from .loaders.tf_loader import TfLoader

class UnifiedOperator(Node):
    def __init__(self):
        super().__init__('unified_operator')
        
        self.declare_parameter('role', 'perception')
        self.declare_parameter('is_controller', False)
        self.declare_parameter('config_file_path', '/config/sandbox.json')

        self.role_name = self.get_parameter('role').value
        self.is_controller = self.get_parameter('is_controller').value
        
        # Unique namespace for this physical device (e.g., "n97_node_1")
        self.device_namespace = f"/{socket.gethostname().replace('-', '_')}"
        
        # Build the targeted service string
        target_service = f"{self.device_namespace}/ComponentManager/_container/load_node"
        
        self.get_logger().info(f"Connecting to local container at: {target_service}")
        
        # Now this client will ONLY talk to the container running on this specific N97
        self.client = self.create_client(LoadNode, target_service)

        # Wait for the local container to actually boot before proceeding
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(f"Waiting for local ComponentManager at {target_service}...")

        self.deployed_components = set()

        self.loaders = {
            "realsense_loader": RealSenseLoader(),
            "repro_loader": ReprojectorLoader(),
            "merger_loader": MergerLoader(),
            "tf_loader": TfLoader()
        }

        if self.is_controller:
            self.create_timer(2.0, self.broadcast_config)
            self.publisher_ = self.create_publisher(String, '/config/sandbox_setup', 10)
            self.get_logger().info(f"Running as CONTROLLER. Device NS: {self.device_namespace}")
        else:
            self.get_logger().info(f"Waiting for CONTROLLER. Device NS: {self.device_namespace}")

        self.create_subscription(String, '/config/sandbox_setup', self.config_callback, 10)

    def config_callback(self, msg):
        try:
            config = json.loads(msg.data)
            role_def = next((r for r in config['roles'] if r['name'] == self.role_name), None)
            if not role_def:
                return

            for loader_key in role_def.get('loaders', []):
                if loader_key in self.loaders:
                    self.loaders[loader_key].discover_and_load(self, config.get('loader_settings', {}))
                else:
                    self.get_logger().error(f"Unknown loader requested: {loader_key}")
        except Exception as e:
            self.get_logger().error(f"Config parse error: {e}")

    def load_component(self, package, plugin, name, params, namespace="", use_ipc=False):
        # Prevent re-deploying the exact same node
        deployment_sig = f"{namespace}/{name}"
        if deployment_sig in self.deployed_components:
            return

        req = LoadNode.Request()
        req.package_name = package
        req.plugin_name = plugin
        req.node_name = name
        req.node_namespace = namespace
        
        for k, v in params.items():
            req.parameters.append(self.make_param(k, v))

        if use_ipc:
            ipc_param = Parameter(name='use_intra_process_comms')
            ipc_val = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=True)
            ipc_param.value = ipc_val
            req.extra_arguments.append(ipc_param)

        self.get_logger().info(f"Loading {plugin} as {deployment_sig}...")
        self.client.call_async(req)
        self.deployed_components.add(deployment_sig)

    def make_param(self, name, value):
        p = Parameter(name=name)
        v = ParameterValue()
        if isinstance(value, bool):
            v.type = ParameterType.PARAMETER_BOOL
            v.bool_value = value
        elif isinstance(value, int):
            v.type = ParameterType.PARAMETER_INTEGER
            v.integer_value = value
        elif isinstance(value, float):
            v.type = ParameterType.PARAMETER_DOUBLE
            v.double_value = value
        else:
            v.type = ParameterType.PARAMETER_STRING
            v.string_value = str(value)
        p.value = v
        return p

    def broadcast_config(self):
        try:
            with open(self.get_parameter('config_file_path').value, 'r') as f:
                self.publisher_.publish(String(data=f.read()))
        except Exception as e:
            self.get_logger().error(f"Failed reading config file: {e}")

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(UnifiedOperator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
