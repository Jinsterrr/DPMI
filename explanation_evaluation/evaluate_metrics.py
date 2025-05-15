import sys
import os
import torch
import torchvision
from torchvision import transforms
import warnings
warnings.simplefilter("ignore")
import numpy as np
from tqdm import tqdm
import csv


current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'models'))
sys.path.append(os.path.join(project_root, 'metrics'))

from models import get_model
from data_factory import DataFactory

from metrics.sparseness import GiniComplexity
from metrics.faithfulness import ROAD
from metrics.robustness import StabilityEvaluator
from explain_func import explain

DATA_ROOT = os.path.join(project_root, 'dataset')



efunc_names = [
    "Saliency",
    "IntegratedGradients", 
      "GradientShap",
    "InputXGradient",
    "GuidedGradCam",
    "FeatureAblation", 
    "FeaturePermutation", 
    "Occlusion",
    "Lime", 
    "KernelShap", 
    "InternalInfluence", 
    "LayerActivation",
    "LayerConductance", 
    "LayerGradientXActivation", 
    "LayerGradCam",
    "FovEx",
    "GradCamPlusScore"
]


def prepare_balanced_batch(testset, num_classes=10, samples_per_class=10):
    # get labels
    all_targets = []
    for _, target in testset:
        all_targets.append(target)
    all_targets = np.array(all_targets)

    np.random.seed(42)
    indices = []
    for i in range(num_classes):
        class_indices = np.where(all_targets == i)[0]
        if len(class_indices) >= samples_per_class:
            selected_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)
        else:
            selected_indices = np.random.choice(class_indices, size=len(class_indices), replace=False)
        indices.extend(selected_indices)

    subset = torch.utils.data.Subset(testset, indices)

    x_batch = []
    y_batch = []

    for i in range(len(subset)):
        x, y = subset[i]
        x_batch.append(x)
        y_batch.append(y)

    x_batch = torch.stack(x_batch).to(device)
    y_batch = torch.tensor(y_batch, dtype=torch.int).to(device)

    return x_batch, y_batch

def evaluate_model(net,model_path, model_name):
    net_type = model_name.partition('_')[0]
    dataset_type = model_name.split('_', 2)[1]
    if net == 'tanh':
        model = get_model(net_type,dataset_type,'tanh').to(device)
    else:
        model = get_model(net_type,dataset_type).to(device)

    if net_type == 'resnet':
        layer_ii = 'model.layer2' 
        layer_other = 'model.layer3'
        layer_rrs = 'gn1'
    elif net_type == 'vgg':
        layer_ii = 'model.features[8]' 
        layer_other = 'model.features[8]'
        layer_rrs = 'features[0]'
    elif net_type == 'simple':
        layer_ii = 'model.conv2' 
        layer_other = 'model.conv3'
        layer_rrs = 'conv1'  
    elif net_type == 'inception':
        layer_ii = 'model.feature[-2]' 
        layer_other = 'model.feature[-1]'
        layer_rrs = 'model.feature[0]'    

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


    # evaluation
    # 1. sparseness (Gini)
    gini_complexity = GiniComplexity(model)
    gini_scores_dict = {}
    for efunc_name in tqdm(efunc_names, desc="Evaluating Complexity"):
        try:
            explain_func_kwargs = {'gc_layer': layer_ii if efunc_name == "InternalInfluence" else layer_other} if efunc_name in ["GuidedGradCam", "InternalInfluence", "LayerActivation", "LayerConductance", "LayerGradientXActivation", "LayerGradCam","GradCamPlusScore"] else {}            
            explain_func_kwargs['reduce_axes'] = (1,)
            gini_scores = gini_complexity.evaluate(x_batch, y_batch, 
                                                   lambda model, inputs, targets, **kwargs: explain(model, inputs, targets, method=efunc_name,  device=device,**kwargs),
                                                   explain_func_kwargs)
            gini_scores_dict[efunc_name] = gini_scores
        except Exception as e:
            print(f"Error in Complexity evaluation for {efunc_name}: {str(e)}")
    np.save(os.path.join(SCORE_ROOT, 'scores_sparseness', f'sparseness_{model_name}.npy'), gini_scores_dict)


    # 2. Faithfulness (ROAD)
    road_evaluator = ROAD(model, percentages=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    with open(os.path.join(SCORE_ROOT, 'scores_faithfulness', f'faithfulness_{model_name}.csv'), 'w', newline='') as f:  
    # add new methods
    # with open(os.path.join(SCORE_ROOT, 'scores_faithfulness', f'faithfulness_{model_name}.csv'), 'a', newline='') as f:  
        writer = csv.writer(f)
        writer.writerow(['Attribution Method'] + [f'{p}%' for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]])
        for efunc_name in tqdm(efunc_names, desc="Evaluating Faithfulness"):
            try:
                explain_func_kwargs = {}
                if efunc_name== "GuidedGradCam":
                    explain_func_kwargs['gc_layer'] = layer_other
                
                if efunc_name in ["InternalInfluence", "LayerActivation", "LayerConductance", 
                                "LayerGradientXActivation", "LayerGradCam","GradCamPlusScore"]:
                    if efunc_name == "InternalInfluence":
                        explain_func_kwargs['gc_layer'] = layer_ii
                    else:
                        explain_func_kwargs['gc_layer'] = layer_other
                    explain_func_kwargs['interpolate'] = (32, 32) 
                    explain_func_kwargs['interpolate_mode'] = 'bilinear'
                explain_func_kwargs['reduce_axes'] =(1,)

                road_scores = road_evaluator.evaluate(x_batch, y_batch, 
                                                      lambda model, inputs, targets, **kwargs: explain(model, inputs, targets, method=efunc_name,  device=device,**kwargs),
                                                      explain_func_kwargs)
                writer.writerow([efunc_name] + [road_scores[p] for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]])

            except Exception as e:
                print(f"Error in Faithfulness evaluation for {efunc_name}: {str(e)}")

    
    # 3. Robustness
    evaluator = StabilityEvaluator(model, layer_rrs)
    robustness_scores_dict = {}
    for efunc_name in tqdm(efunc_names, desc="Evaluating Robustness"):
        try:
            explain_func_kwargs = {'gc_layer': layer_ii if efunc_name == "InternalInfluence" else layer_other} if efunc_name in ["GuidedGradCam", "InternalInfluence", "LayerActivation", "LayerConductance", "LayerGradientXActivation", "LayerGradCam","GradCamPlusScore"] else {}
            explain_func_kwargs['reduce_axes'] = (1,)
            ris_batch = evaluator.evaluate(
                x_batch, y_batch, 
                lambda model, inputs, targets, **kwargs: explain(model, inputs, targets, method=efunc_name, device=device, **kwargs),
                explain_func_kwargs
            )
            robustness_scores_dict[efunc_name] = {'RIS': ris_batch}
        except Exception as e:
            print(f"Error in Robustness evaluation for {efunc_name}: {str(e)}")
    
    np.save(os.path.join(SCORE_ROOT, 'scores_robustness', f'robustness_{model_name}.npy'), robustness_scores_dict)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cifar10
data_factory_cifar10 = DataFactory(which= 'cifar10', data_root=DATA_ROOT)
testset_cifar10 = data_factory_cifar10.getTestSet()
x_batch_cifar10, y_batch_cifar10 = prepare_balanced_batch(testset_cifar10, num_classes=10, samples_per_class=11)

## mnist
# data_factory_mnist = DataFactory(which= 'mnist', data_root=DATA_ROOT)
# testset_mnist = data_factory_mnist.getTestSet()
# x_batch_mnist, y_batch_mnist = prepare_balanced_batch(testset_mnist, num_classes=10, samples_per_class=100)

## fmnist
# data_factory_fmnist = DataFactory(which= 'fmnist', data_root=DATA_ROOT)
# testset_fmnist = data_factory_fmnist.getTestSet()
# x_batch_fmnist, y_batch_fmnist = prepare_balanced_batch(testset_fmnist, num_classes=10, samples_per_class=100)

# # SVHN
# data_factory_svhn = DataFactory(which= 'svhn', data_root=DATA_ROOT)
# testset_svhn = data_factory_svhn.getTestSet()
# x_batch_svhn, y_batch_svhn = prepare_balanced_batch(testset_svhn, num_classes=10, samples_per_class=100)
 
net_list = ['loss','tanh']

for net_i in net_list:
    MODEL_ROOT = os.path.join(project_root, 'trained_net', net_i, 'none')

    SCORE_ROOT = os.path.join(project_root, 'metric_mine', 'scores_110samples_loss_tanh_none',net_i)

    if not os.path.exists(SCORE_ROOT):
        os.makedirs(SCORE_ROOT)

    # 创建保存结果的文件夹
    for metric in ['sparseness', 'faithfulness', 'robustness']:
        os.makedirs(os.path.join(SCORE_ROOT, f'scores_{metric}'), exist_ok=True)
    for model_file in os.listdir(MODEL_ROOT):
        if model_file.endswith('.pt'):
            model_path = os.path.join(MODEL_ROOT, model_file)
            model_name = model_file[:-3]  
            dataset = model_name.split('_', 2)[1]
            if dataset == "cifar10":
                x_batch,y_batch = x_batch_cifar10, y_batch_cifar10
            # elif dataset == "mnist":
            #     x_batch,y_batch = x_batch_mnist, y_batch_mnist
            # elif dataset == "fmnist":
            #     x_batch,y_batch = x_batch_fmnist, y_batch_fmnist
            # elif dataset == "svhn":
            #     x_batch,y_batch = x_batch_svhn, y_batch_svhn                

            print(f"Evaluating model: {model_name}")
            evaluate_model(net_i,model_path, model_name)

print("All evaluations complete.")