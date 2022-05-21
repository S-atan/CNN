import json


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_new = {}
    for k in list(keys)[start:end]:
        dict_new[k] = adict[k]
    return dict_new


with open("./dataset/GIT_zizi.json", 'r') as load_f:
    load_dict = json.load(load_f)
    load_dict = json.loads(load_dict)
    dict_json = dict_slice(load_dict, 3, 4)    # n-1 到 n 的字典切片

print(type(dict_json))
json_str = json.dumps(dict_json)
print(type(json_str))
filename = '0.json'      # ?.json文件
with open(filename, 'w') as file_obj:
    file_obj.write(json_str + '"')

print('END!')
