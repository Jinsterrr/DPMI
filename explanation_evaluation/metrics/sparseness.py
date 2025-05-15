import torch
import numpy as np
from typing import Dict, List, Optional, Callable

class GiniComplexity:
    def __init__(self, model, abs: bool = True, normalise: bool = False, 
                 normalise_func: Optional[Callable] = None, 
                 normalise_func_kwargs: Optional[Dict] = None,
                 internal_batch_size: int = 64):
        self.model = model
        self.abs = abs
        self.normalise = normalise
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs if normalise_func_kwargs else {}
        self.internal_batch_size = internal_batch_size
        self.device = next(model.parameters()).device

    def evaluate(self, x_batch: torch.Tensor, y_batch: torch.Tensor, 
                 explain_func: Callable, explain_func_kwargs: Dict) -> List[float]:
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

            # 步骤 1: 计算attribution值
            attributions = explain_func(self.model, x_sub_batch, predicted, **explain_func_kwargs)
            

            # 步骤 2: 预处理attribution值
            attributions = self.preprocess_attributions(attributions)
            
            # 步骤 3: 计算Gini指数
            sub_batch_results = self.calculate_gini(attributions)
            all_results.extend(sub_batch_results)

        return all_results

    def preprocess_attributions(self, attributions: np.ndarray) -> np.ndarray:
        if self.abs:
            attributions = np.abs(attributions)
        
        if self.normalise:
            if self.normalise_func:
                attributions = self.normalise_func(attributions, **self.normalise_func_kwargs)
            else:
                attributions = attributions / np.max(np.abs(attributions), axis=1, keepdims=True)
        
        return attributions

    def calculate_gini(self, attributions: np.ndarray) -> List[float]:
        results = []
        for attribution in attributions:
            a = np.sort(attribution.flatten())
            a += 1e-7  # 添加小值以避免除零错误
            index = np.arange(1, len(a) + 1)
            gini = (np.sum((2 * index - len(a) - 1) * a)) / (len(a) * np.sum(a))
            results.append(gini)
        return results