import numpy as np
import time
def test_dot():
    start_time = time.time()
    for i in range(10000):
        a = np.random.rand(1,376)
        b = np.random.rand(376,25)
        c = np.dot(a,b)
    print("--- %s seconds ---" % (time.time() - start_time))
    
def test_matmul():
    start_time = time.time()
    for i in range(10000):
        a = np.random.rand(1,376)
        b = np.random.rand(376,25)
        c = np.matmul(a,b)
    print("--- %s seconds ---" % (time.time() - start_time))
    
test_dot()
test_matmul()
