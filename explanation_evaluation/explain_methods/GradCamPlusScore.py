import torch
import torch.nn.functional as F

class GradCamPlusScore:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def get_cam_weights(self, grads):
        return torch.mean(grads, dim=(2, 3), keepdim=True)
    
    # def get_loss(self, output, target):
    #     return torch.sum(output * target)

    def get_loss(self, output, target):
        if isinstance(target, torch.Tensor):
            if target.dim() == 1:
                return F.cross_entropy(output, target)
            elif target.dim() == 2 and target.size(1) == output.size(1):
                return -torch.sum(F.log_softmax(output, dim=1) * target, dim=1).mean()
        elif isinstance(target, int):
            target_tensor = torch.full((output.size(0),), target, dtype=torch.long, device=output.device)
            return F.cross_entropy(output, target_tensor)
        raise ValueError(f"Unsupported target type or shape: {type(target)}, {target.shape if isinstance(target, torch.Tensor) else ''}")

    
    def generate_cam(self, activations, weights):
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        return F.relu(cam)
    
    def normalize_cam(self, cam):
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)
        return cam
    
    def attribute(self, inputs, target, relu_attributions=False):
        # 确保模型处于评估模式
        self.model.eval()
        
        # 前向传播
        output = self.model(inputs)
        num_classes = output.shape[1]
        
        # 处理 target
        if isinstance(target, int):
            target = torch.full((inputs.size(0),), target, dtype=torch.long, device=inputs.device)
        elif isinstance(target, torch.Tensor):
            if target.dim() == 0:
                target = target.expand(inputs.size(0))
            elif target.dim() == 1 and target.size(0) != inputs.size(0):
                raise ValueError(f"Target size ({target.size(0)}) must match input batch size ({inputs.size(0)})")
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")
        
        # 计算损失并反向传播
        loss = self.get_loss(output, target)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        # 获取梯度和激活
        gradients = self.gradients
        activations = self.activations
        
        # 计算GradCAM++权重
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        alpha = grad_2 / (2 * grad_2 + torch.sum(gradients * grad_3, dim=(2, 3), keepdim=True) + 1e-6)
        gradcam_weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # 计算 ScoreCAM 权重
        b, c, h, w = activations.shape
        score_weights = []
        target_one_hot = F.one_hot(target, num_classes=num_classes).float()
        for i in range(c):
            act_mask = F.interpolate(activations[:, i:i+1, :, :], size=inputs.shape[2:], mode='bilinear', align_corners=False)
            masked_input = inputs * act_mask
            mask_output = self.model(masked_input)
            score = torch.sum(mask_output * target_one_hot, dim=1)
            score_weights.append(score.view(b, 1, 1, 1))
        score_weights = torch.cat(score_weights, dim=1)
        
        # 结合GradCAM++和ScoreCAM权重
        combined_weights = gradcam_weights * score_weights
        
        # 生成并标准化CAM
        cam = self.generate_cam(activations, combined_weights)
        cam = self.normalize_cam(cam)
        
        # 应用ReLU（如果需要）
        if relu_attributions:
            cam = F.relu(cam)
        
        return cam

    def interpolate_cam(self, cam, target_shape):
        return F.interpolate(cam, size=target_shape, mode='bilinear', align_corners=False)