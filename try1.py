
import json

a64 = json.load(open('/home/cuonghoang/Desktop/codedict/RNCDL-main/dataset/lvis/lvis_v1_train.json'))

# print(a64['categories'])

for i in a64['categories']:
    if 'book' in i['name']:
        print(i['name'])
        print(i['instance_count'])