import os
import itertools
pwd = os.path.split(os.path.realpath(__file__))[0]

nets = ['simplenn','resnet','inception','vgg']
# nets = ['simplenn']
datasets = ['svhn','cifar10','fmnist','mnist']
epoch = 500
eps = [0.2,0.3,0.4,0.5,1,2,4,8,100,1000,None]

def gen_scripts(params):
    for param in params:
        net, dataset, e,  = param
        if(e is None):
            cmd = f"python3 -u {os.path.join(pwd, 'uda_main.py')} --net %s --dataset %s --epoch {epoch} --n_query 100" % param[:-1]
        else:
            cmd = f"python3 -u {os.path.join(pwd, 'uda_main.py')} --net %s --dataset %s --epoch {epoch} --eps %s --n_query 100" % param
        with open(os.path.join(pwd,'..','..','scripts',"pate_uda_stu_%s_%s_%s.sh"%param),'wt') as f:
            f.write(cmd)

params = itertools.product(nets,datasets,eps)
# params = [

# ]

gen_scripts(params)