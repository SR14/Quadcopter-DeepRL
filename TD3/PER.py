import numpy as np

class PrioritizedReplayBuffer: 
     
        
    def __init__(self, buffer_size, batch_size, db, a=0.6, b=0.4, min_error=0.0001):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.calc = self.BinaryHeap(buffer_size)
        self.a = a
        self.b = b
        self.db = db
        self.min_error = min_error
        
        
    def add(self, experience):
        max_p = np.max(self.calc.priorities[-self.buffer_size:])
        if max_p == 0:
            max_p = 1.
        self.calc.add(max_p, experience)
        
        
    def sample(self):
        experiences = []
        priorities_idx = np.empty((self.batch_size, 1), dtype=np.int32)
        weights = np.empty((self.batch_size, 1), dtype=np.float32)
        p_segment = self.calc.total_priority / self.batch_size        
        self.b = np.min([1., self.b + self.db])         
        p_min = np.min(self.calc.priorities[-self.buffer_size:]) / self.calc.total_priority
        max_w = (p_min * self.batch_size) ** (-self.b)     
        for idx in range(self.batch_size):
            idx_start = p_segment * idx
            idx_end = p_segment * (idx + 1)
            p_idx = np.random.uniform(idx_start, idx_end)
            p_idx, priority, exp = self.calc.priority(p_idx)
            prob = priority / self.calc.total_priority            
            weights[idx, 0] = np.power(self.batch_size * prob, -self.b) / max_w
            priorities_idx[idx, 0] = p_idx
            experiences.append(exp)    
        return priorities_idx, weights, experiences
    
    
    def update_weights(self, priorities_idx, errors):
        errors += self.min_error
        errors = np.minimum(errors, self.min_error)
        priorities = np.power(errors, self.a)
        for idx, p in zip(priorities_idx, priorities):
            self.calc.update(idx, p)
 

    class BinaryHeap:
        def __init__(self, buffer_size):
            self.buffer_size = buffer_size
            self.priorities = np.zeros(2 * buffer_size - 1)
            self.experiences = np.zeros(buffer_size, dtype=object)
            self.idx = 0
        def add(self, max_p, exp):
            self.experiences[self.idx] = exp
            idx_p = self.idx + self.buffer_size - 1
            self.update(idx_p, max_p)
            self.idx += 1
            if self.idx >= self.buffer_size:
                self.idx = 0    
        def update(self, idx_p, p):
            change = p - self.priorities[idx_p]
            self.priorities[idx_p] = p        
            while idx_p != 0:    
                idx_p = (idx_p - 1) // 2
                self.priorities[idx_p] += change         
        def priority(self, p):
            main_idx = 0
            while True:
                left_idx = 2 * main_idx + 1
                right_idx = left_idx + 1            
                if left_idx >= len(self.priorities):
                    break           
                else:                 
                    if p <= self.priorities[left_idx]:
                        main_idx = left_idx                    
                    else:
                        p -= self.priorities[left_idx]
                        main_idx = right_idx            
            exp_idx = main_idx - self.buffer_size + 1
            return main_idx, self.priorities[main_idx], self.experiences[exp_idx] 
        @property
        def total_priority(self):
            return self.priorities[0]  
            