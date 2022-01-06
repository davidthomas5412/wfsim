import numpy as np
from wfTel import LSSTFactory

factory = LSSTFactory('r')
dof = np.zeros(50)
visit = factory.make_visit_telescope(dof=dof)
print(visit.dz())