import os

projects = ['Java', 'JFace', 'commons.collections']
labeltypes = ['train.label', 'test.label']
for project in projects:
	for fold in range(10):
		for cent in [20, 40, 60, 70]:
			for labeltype in labeltypes:
				with open(os.path.join('Data/exp2', project, str(fold), str(cent), labeltype)) as f:
					labels = f.readlines()
				true = []
				for label in labels:
					true.append(int(float(label.strip())))
				with open(os.path.join('Data/exp2', project, str(fold), str(cent), labeltype), 'w') as f:
					for label in true:
						f.write(str(label))
						f.write('\n')