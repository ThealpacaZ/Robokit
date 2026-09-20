"""推理 policy 注册表与统一接口。

Policy 接口（鸭子类型，无需继承）:

    class MyPolicy:
        def __init__(self, **kwargs): ...        # kwargs 来自 deploy_server 的 --policy-arg
        def reset(self): ...                     # 每个新客户端连接时调用
        def infer(self, obs) -> np.ndarray: ...  # 返回 (horizon, action_dim) 动作块

    obs = {"images": {相机名: (H,W,3) uint8 RGB},
           "state":  {臂名: {"joint", "eef_pose", "gripper"}},
           "instruction": str}

新 policy：写好类后在 REGISTRY 加一行，或直接用 "模块路径:类名" 指定。
"""
import importlib

REGISTRY = {
    # 无模型联调：回发当前状态，不需要 GPU
    "dummy": "robokit.policies.dummy:DummyPolicy",
    # MemoryVLA 全量 checkpoint（laMem-VLA 这类 33.5 GB 的 .pt），输出 16 步动作块
    "memoryvla": "robokit.policies.memoryvla:MemoryVLAPolicy",
    # MemoryVLA(CogACT-Large) + LoRA。输出动作块，--horizon 可 >1
    "memvla_lora": "robokit.policies.memvla_lora:MemVLALoRAPolicy",
    "openvla_oft": "robokit.policies.openvla_oft:OpenVLAOFTPolicy",
}


def create_policy(spec, **kwargs):
    """spec: REGISTRY 里的名字，或 "模块路径:类名"。kwargs 透传给 policy 构造函数。"""
    spec = REGISTRY.get(spec, spec)
    if ":" not in spec:
        raise ValueError(f"unknown policy '{spec}', registered: {list(REGISTRY)}; "
                         f"or pass '模块路径:类名'")
    module_path, class_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kwargs)
