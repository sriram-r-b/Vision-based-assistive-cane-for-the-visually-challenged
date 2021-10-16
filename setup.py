import os
from setuptools import setup

arch = os.uname().machine

if arch == 'armv7l':
	tensorflow = 'tensorflow @ https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl'
	
elif arch == 'aarch64':
	tensorflow = 'https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_aarch64.whl'	

elif arch == 'x86_64':
	tensorflow = "tensorflow==2.4.0rc2"
else:
	raise Exception(f'Could not find TensorFlow binary for target {arch}. Please open a Github issue.')
    
requirements = [
    tensorflow,
    # specify additional package requirements here
]
 
setup(
    install_requires=requirements,
    # specify additional setup parameters here
)
