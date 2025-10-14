import torch
import torch.nn as nn

class ActorModel(nn.Module):
    def __init__(self, input_dim=380, output_dim=23):
        super().__init__()
        self.actor_module = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
        # std 参数虽然存在，但推理时不用
        self.std = nn.Parameter(torch.ones(output_dim))

    def forward(self, x):
        mean = self.actor_module(x)
        # 推理控制 → 只用 mean
        return mean
import torch
import onnx

# 加载 checkpoint
checkpoint = torch.load("model_9100.pt", map_location="cpu")

# 处理 state_dict key
state_dict = checkpoint["actor_model_state_dict"]
new_state_dict = {k.replace("actor_module.module.", "actor_module."): v for k, v in state_dict.items()}

model = ActorModel(input_dim=380, output_dim=23)
model.load_state_dict(new_state_dict)
model.eval()

# 输入是机器人状态向量 (1, 380)
dummy_input = torch.randn(1, 380)

torch.onnx.export(
    model,
    dummy_input,
    "actor_model_inference.onnx",
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=["obs"],
    output_names=["actions"]
)

print("✅ 导出成功: actor_model_inference.onnx")

