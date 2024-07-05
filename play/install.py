import yaml

data = { 'train' : '/Users/503/play/train/images/',
         'val' : '/Users/503/play/val/images/',
         'test' : '/Users/503/play/test/images/',
         'names' : ['rollercoaster', 'viking', 'merrygoround', 'ferriswheel'],
         'nc' : 4 }

with open('/Users/503/play/coco8.yaml', 'w') as f :
    yaml.dump(data, f)

with open('/Users/503/play/coco8.yaml', 'r') as f : 
    taco_yaml = yaml.safe_load(f)
    print(taco_yaml)