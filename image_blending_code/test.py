import math
num_t = 20
lent = 144
for t in range(num_t*2):
    start = math.floor(lent/num_t*(t%num_t))
    end = min(lent, math.floor(lent/num_t*(t%num_t+1)))
    
    print(start)
    print(end)
