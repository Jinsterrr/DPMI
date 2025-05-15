import torch
import torch.nn as nn
import torchvision
import numpy as np

class StabilityEvaluator:
    def __init__(self, model, layer_name, num_perturbations=50, perturbation_std=0.05, eps_min=1e-6): # 50次扰动
        self.model = model
        self.layer_name = layer_name
        self.num_perturbations = num_perturbations
        self.perturbation_std = perturbation_std
        self.intermediate_output = None
        # self._register_hook()
        self.device = next(model.parameters()).device  # Get the device of the model
        self._eps_min = eps_min
    
    def _get_norm_function(self, num_dim):   # 用于计算范数的函数
        if num_dim == 4:
            return lambda arr: np.linalg.norm(
                np.linalg.norm(arr, axis=(-1, -2)), axis=-1
            )
        elif num_dim == 3:
            return lambda arr: np.linalg.norm(arr, axis=(-1, -2))
        elif num_dim == 2:
            return lambda arr: np.linalg.norm(arr, axis=-1)
        else:
            raise ValueError(
                "Relative Stability only supports 4D, 3D and 2D inputs (batch dimension inclusive)."
            )
    
    def _compute_norm(self, arr):
        num_dim = arr.ndim
        norm_function = self._get_norm_function(num_dim)
        return norm_function(arr)


    def _register_hook(self):
        def hook(module, input, output):
            self.intermediate_output = output.detach()
        if self.layer_name == 'gn1':
            self.model.gn1.register_forward_hook(hook)
        elif self.layer_name == 'features[0]':
            self.model.features[0].register_forward_hook(hook)
        elif self.layer_name == 'layer1':
            self.model.layer1.register_forward_hook(hook)
        elif self.layer_name == 'layer2':
            self.model.layer2.register_forward_hook(hook)
        elif self.layer_name == 'layer3':
            self.model.layer3.register_forward_hook(hook)
        
        else:
            raise ValueError(f"Unsupported layer name: {self.layer_name}")

    def evaluate(self, x_batch, y_batch, explain_func, explain_func_kwargs, internal_batch_size=32):
        total_batch_size = x_batch.shape[0]
        num_batches = (total_batch_size + internal_batch_size - 1) // internal_batch_size

        ris_all = []
        ros_all = []
        rrs_all = []

        for i in range(num_batches):
            start_idx = i * internal_batch_size
            end_idx = min((i + 1) * internal_batch_size, total_batch_size)
            
            x_sub_batch = x_batch[start_idx:end_idx].to(self.device)
            y_sub_batch = y_batch[start_idx:end_idx].to(self.device)

            sub_batch_size = x_sub_batch.shape[0]

            ris_sub_batch = np.zeros((sub_batch_size,))
            # ros_sub_batch = np.zeros((sub_batch_size,))
            # rrs_sub_batch = np.zeros((sub_batch_size,))

            # 计算原始子批次的输出、预测标签、解释和隐藏层输出
            with torch.no_grad():
                output = self.model(x_sub_batch)
                # h_x = output.cpu().numpy()  # 输出 logits
                predicted_labels = torch.argmax(output, dim=1) # 预测标签
                # _ = self.model(x_sub_batch)  # This will trigger the hook
                # l_x = self.intermediate_output.cpu().numpy()

            # 使用预测标签作为解释函数的目标
            e_x = explain_func(self.model, x_sub_batch, predicted_labels, **explain_func_kwargs)

            for _ in range(self.num_perturbations):
                # 对子批次添加扰动
                x_perturbed = self.add_perturbation(x_sub_batch)
                
                # 预测扰动后子批次的标签
                with torch.no_grad():
                    output_perturbed = self.model(x_perturbed)
                    predicted_labels_perturbed = torch.argmax(output_perturbed, dim=1)

                # 计算扰动后子批次的解释、预测概率和隐藏层输出
                # _ = self.model(x_perturbed)  # This will trigger the hook
                # l_x_perturbed = self.intermediate_output.cpu().numpy()
                e_x_perturbed = explain_func(self.model, x_perturbed, predicted_labels_perturbed, **explain_func_kwargs)
                # h_x_perturbed = output_perturbed.cpu().numpy()

                # 计算 nominator
                e_x, e_x_perturbed = np.asarray(e_x), np.asarray(e_x_perturbed)
                nominator = (e_x - e_x_perturbed) / (e_x + (e_x == 0) * self._eps_min)
                nominator = self._compute_norm(nominator)

                # 计算 RIS、ROS 和 RRS
                ris_values = self.compute_ris(nominator, x_sub_batch.cpu(), x_perturbed.cpu())
                # ros_values = self.compute_ros(nominator, h_x, h_x_perturbed)
                # rrs_values = self.compute_rrs(nominator, l_x, l_x_perturbed)

                # 更新最大值
                ris_sub_batch = np.maximum(ris_sub_batch, ris_values)
                # ros_sub_batch = np.maximum(ros_sub_batch, ros_values)
                # rrs_sub_batch = np.maximum(rrs_sub_batch, rrs_values)

                # 如果预测标签改变，将对应的值设为 NaN
                label_changed = (predicted_labels != predicted_labels_perturbed).cpu().numpy()
                ris_sub_batch[label_changed] = np.nan
                # ros_sub_batch[label_changed] = np.nan
                # rrs_sub_batch[label_changed] = np.nan

            ris_all.append(ris_sub_batch)
            # ros_all.append(ros_sub_batch)
            # rrs_all.append(rrs_sub_batch)

        # 合并所有子批次的结果
        ris_batch = np.concatenate(ris_all)
        # ros_batch = np.concatenate(ros_all)
        # rrs_batch = np.concatenate(rrs_all)

        # return ris_batch, ros_batch, rrs_batch
        return ris_batch

    def add_perturbation(self, x_batch):
        noise = torch.randn_like(x_batch).to(self.device) * self.perturbation_std
        x_perturbed = x_batch + noise
        return x_perturbed


    def compute_ris(self, nominator, x, x_perturbed):
        x, x_perturbed = x.cpu().numpy(), x_perturbed.cpu().numpy()
        denominator = (x - x_perturbed) / (x + (x == 0) * self._eps_min)
        denominator = self._compute_norm(denominator)
        denominator += (denominator == 0) * self._eps_min
        return nominator / denominator

    def compute_rrs(self, nominator, l_x, l_x_perturbed):
        denominator = (l_x - l_x_perturbed) / (l_x + (l_x == 0) * self._eps_min)
        denominator = self._compute_norm(denominator)
        denominator += (denominator == 0) * self._eps_min
        return nominator / denominator

    def compute_ros(self, nominator, h_x, h_x_perturbed):
        denominator = h_x - h_x_perturbed
        denominator = self._compute_norm(denominator)
        denominator += (denominator == 0) * self._eps_min
        return nominator / denominator