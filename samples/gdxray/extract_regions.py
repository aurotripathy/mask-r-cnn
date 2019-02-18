import json
import glob

ROOT_DIR = '/home/auro/via/nirmalyalabs-work/'


# https://gist.github.com/douglasmiranda/5127251
def find(key, dictionary):
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result


for name in glob.glob(ROOT_DIR + '/*/*/*.json'):
    print('\nFile name:', name)
    with open(name, "r") as read_file:
        regions = json.load(read_file)

    region_list = list(find('regions', regions))
    print('\nRegion list')
    print(region_list)

    print('\nEnumerate shapes')
    for shape_attrib in region_list[0]:
        print('all_points_x', shape_attrib['shape_attributes']['all_points_x'])
        print('all_points_y', shape_attrib['shape_attributes']['all_points_y'])
