import sys
import numpy as np

try:
    mat_a = np.loadtxt(sys.argv[1], delimiter=',')
    mat_b = np.loadtxt(sys.argv[2], delimiter=',')
except Exception as e:
    print('Error:', str(e))
    sys.exit(-1)

mat_c = np.matmul(mat_a, mat_b)
unsort_list = list(mat_c.flatten())

with open('ans_one.txt', 'w') as out_f:
    for v in sorted(unsort_list):
        out_f.write('%d\n' % int(v))

