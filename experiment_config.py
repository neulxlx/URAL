from yacs.config import CfgNode as CN

e0 = CN()
e0.IDs_train = ['09', '15', '17', '04', '07', '08', '18', '11']
e0.IDs_meta_eval = ['13']
e0.IDs_eval = ['01', '02', '03', '05', '06', '10', '12', '14', '16']
e0.IDs_test = ['01', '02', '03', '05', '06', '10', '12', '14', '16']

EXPERIMENTS = (e0)

       