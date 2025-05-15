import torch
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Dict, Union, Sequence, Tuple

class ROAD:
    def __init__(self, model, percentages=None, noise=0.01, internal_batch_size=64):
        self.model = model
        self.percentages = percentages if percentages is not None else list(range(1, 100, 2))
        self.noise = noise
        self.internal_batch_size = internal_batch_size
        self.device = next(model.parameters()).device

    def evaluate(self, x_batch: torch.Tensor, y_batch: torch.Tensor, explain_func, explain_func_kwargs: Dict):
        total_batch_size = x_batch.shape[0]
        num_batches = (total_batch_size + self.internal_batch_size - 1) // self.internal_batch_size
        
        all_results = []

        for i in range(num_batches):
            start_idx = i * self.internal_batch_size
            end_idx = min((i + 1) * self.internal_batch_size, total_batch_size)
            
            x_sub_batch = x_batch[start_idx:end_idx]
            y_sub_batch = y_batch[start_idx:end_idx]

            # 首先获取模型的预测结果
            with torch.no_grad():
                outputs = self.model(x_sub_batch)
                _, predicted = torch.max(outputs, 1)

            # 步骤 1: 计算attribution值，使用模型的预测结果而不是真实标签
            attributions = explain_func(self.model, x_sub_batch, predicted, **explain_func_kwargs)
            
            # 步骤 2: 筛选top-k特征
            sorted_indices = np.argsort(attributions.reshape(attributions.shape[0], -1), axis=1)[:, ::-1]
            
            # 步骤 3-5: 对每个百分比进行扰动和评估
            sub_batch_results = self.evaluate_percentages(x_sub_batch, y_sub_batch, sorted_indices)
            all_results.extend(sub_batch_results)

        # 计算整个测试集的正确率
        return self.calculate_accuracy(all_results)

    def evaluate_percentages(self, x_sub_batch: torch.Tensor, y_sub_batch: torch.Tensor, sorted_indices: np.ndarray) -> List[Dict[str, Union[float, bool]]]:
        results = []
        x_np = x_sub_batch.cpu().numpy()
        
        for idx, (x, y, indices) in enumerate(zip(x_np, y_sub_batch, sorted_indices)):
            instance_results = []
            for percentage in self.percentages:
                num_features = int(np.prod(x.shape[1:]) * percentage / 100)
                top_k_indices = indices[:num_features]
                
                # 步骤 3: 扰动输入
                x_perturbed = self.noisy_linear_imputation(x, top_k_indices)
                
                # 步骤 4: 预测
                with torch.no_grad():
                    output = self.model(torch.from_numpy(x_perturbed).unsqueeze(0).to(self.device))
                    pred = output.argmax(dim=1).item()
                
                # 步骤 5: 记录是否正确
                is_correct = (pred == y.item())
                instance_results.append({"percentage": percentage, "correct": is_correct})
            
            results.append(instance_results)
        
        return results

    def noisy_linear_imputation(self, arr: np.ndarray, indices: Union[Sequence[int], Tuple[np.ndarray]], noise: float = 0.1) -> np.ndarray:
        if len(indices) == 0:
            return np.copy(arr)
        offset_weight = [
            ((1, 1), 1/12), ((0, 1), 1/6), ((-1, 1), 1/12),
            ((1, -1), 1/12), ((0, -1), 1/6), ((-1, -1), 1/12),
            ((1, 0), 1/6), ((-1, 0), 1/6)
        ]
        
        arr_flat = arr.reshape((arr.shape[0], -1))
        mask = np.ones(arr_flat.shape[1])
        mask[indices] = 0
        ind_to_var_ids = np.zeros(arr_flat.shape[1], dtype=int)
        ind_to_var_ids[indices] = np.arange(len(indices))

        a = lil_matrix((len(indices), len(indices)))
        b = np.zeros((len(indices), arr.shape[0]))
        sum_neighbors = np.ones(len(indices))

        for offset, weight in offset_weight:
            x = indices // arr.shape[2] + offset[0]
            y = indices % arr.shape[2] + offset[1]
            valid = ~((x < 0) | (y < 0) | (x >= arr.shape[1]) | (y >= arr.shape[2]))
            off_coords = indices + offset[0] * arr.shape[2] + offset[1]
            off_coords = off_coords[valid]
            valid_ids = np.argwhere(valid).flatten()

            in_mask = mask[off_coords] == 1
            b[valid_ids[in_mask], :] -= weight * arr_flat[:, off_coords[in_mask]].T

            out_mask = mask[off_coords] != 1
            a[valid_ids[out_mask], ind_to_var_ids[off_coords[out_mask]]] = weight

            sum_neighbors[~valid] -= weight

        a[np.arange(len(indices)), np.arange(len(indices))] = -sum_neighbors
        res = np.transpose(spsolve(csc_matrix(a), b))

        arr_flat_copy = np.copy(arr_flat)
        arr_flat_copy[:, indices] = res + noise * np.random.randn(*res.shape)

        return arr_flat_copy.reshape(*arr.shape)

    def calculate_accuracy(self, results: List[Dict[str, Union[float, bool]]]) -> Dict[float, float]:
        accuracy = {p: [] for p in self.percentages}
        for instance_results in results:
            for result in instance_results:
                accuracy[result["percentage"]].append(float(result["correct"]))
        
        return {p: np.mean(acc) for p, acc in accuracy.items()}