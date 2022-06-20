import os
import sys
from fuzzywuzzy import fuzz
import subprocess

path = "D:\Grad\COMPSCI 589\hw2"


files = os.listdir(path)
pyfiles = []
for root, dirs, files in os.walk(path):
      for file in files:
        if file.endswith('.py'):
              pyfiles.append(os.path.join(root, file))

stopWords = ['from', 'import',',','.']

importables = []

for file in pyfiles:
    with open(file, encoding="ISO-8859-1") as f:
        content = f.readlines()

        for line in content:
            if "import" in line:
                for sw in stopWords:
                    line = ' '.join(line.split(sw))

                importables.append(line.strip().split(' ')[0])

importables = set(importables)

subprocess.call(f"pip freeze > {path}/requirements.txt", shell=True)

with open(path+'/requirements.txt', encoding="ISO-8859-1") as req:
    modules = req.readlines()
    modules = {m.split('=')[0].lower() : m for m in modules}


notList = [''.join(i.split('_')) for i in sys.builtin_module_names]+['os']

new_requirements = []
for req_module in importables:
    try :
        new_requirements.append(modules[req_module])

    except KeyError:
        for k,v in modules.items():
            if len(req_module)>1 and req_module not in notList:
                if fuzz.partial_ratio(req_module,k) > 90:
                    new_requirements.append(modules[k])

new_requirements = [i for i in set(new_requirements)]

new_requirements

with open(path+'/requirements.txt','w', encoding="ISO-8859-1") as req:
    req.write(''.join(new_requirements))