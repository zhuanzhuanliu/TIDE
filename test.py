import pickle
import json
from utils.extract_fourclass_data import read_single_xml
import os
import numpy as np

p = 0.66
r = 0.62
f = 2*p*r/(p+r)
print(f)




def revise_data(path):
    old_pre = '/sdb/nlp21/Project/physical/depression-main/raw_data'
    new_pre = '/sdb/nlp21/Project/physical/depression-main/raw_data/revise_data/2022/DATA'
    new_path = os.path.join(new_pre)
    old_path = os.path.join(old_pre, path)

    file_list = os.listdir(old_path)
    for file in file_list:
        if not file.endswith('.xml'):
            continue
        with open(os.path.join(new_path, file),'w') as file_revise:
            with open(os.path.join(old_path, file), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line=line.replace('&','and')
                    file_revise.write(line)


