import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

types = ['black','white']
datasets=['mnist','fmnist','cifar10','svhn']
eps = [None]
nets = ['simplenn','resnet','inception','vgg']
udas = [True]
def gen_scripts(params):
    for param in params:
        type, net, dataset, uda, e = param
        if(e is None):
            cmd = f"python3 {os.path.join(pwd, 'prepare_dataset.py')} "+f"--type %s --net %s --dataset %s --uda %s" % param[:-1]
        else:
            cmd = f"python3 {os.path.join(pwd, 'prepare_dataset.py')} "+f"--type %s --net %s --dataset %s --uda %s --eps %s " % param
        with open(os.path.join(pwd,'..','..','scripts',"attackset_%s_%s_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product(types,nets,datasets,udas,eps)
# replace when need customization
params = [
    # (type, net, datset, uda, eps)
    ('black', 'resnet', 'cifar10', 'false', None)
]

gen_scripts(params)             