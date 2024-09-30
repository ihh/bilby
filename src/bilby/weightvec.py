import numpy as np
from copy import deepcopy

# Time-efficient, memory-greedy data structure for storing a vector of weights whose length N is a power of 2
# Uses 2N memory to store partial sums with pool sizes at all powers of 2, from 0 (individual elements) up to log2(length) (full sum).
# Computing total weight is an O(1) operation
# Get, modify, and retrieve element by cumulative weight (e.g. for sampling) are O(log(N)) operations
class WeightVec:
    def __init__ (self, n, dtype=np.float16):
        self.dtype = dtype
        self.n = n
        self.log2n = int(n).bit_length() - 1
        if 2**self.log2n != n:
            raise ValueError ("WeightVec length is not a power of 2: " + str(n))
        # partialSums[k] contains 2^(log2n-k) partial sums of size-2^k contiguous blocks
        self.partialSums = [np.zeros(1 << (self.log2n - k), dtype=dtype) for k in range(self.log2n + 1)]

    def rebuildPartialSums (self):
        for k in range(1, self.log2n + 1):
            current = self.partialSums[k]
            prev = self.partialSums[k-1]
            for i in range(1 << (self.log2n - k)):
                current[i] = prev[2*i] + prev[2*i + 1]

    def __setitem__ (self, key, weight):
        assert 0 <= key < self.n
        oldWeight = self.partialSums[0][key]
        delta = weight - oldWeight
        for k in range(self.log2n + 1):
            self.partialSums[k][key >> k] += delta

    def __getitem__ (self, key):
        assert 0 <= key < self.n
        return self.partialSums[0][key]    

    def __len__  (self):
        return self.n

    def __str__ (self):
        return str(self.partialSums[0])

    def __add__ (self, other):
        if isinstance(other, WeightVec):
            if self.n != other.n:
                raise ValueError ("WeightVecs are different lengths: " + str(self.n) + " vs " + str(other.n))
            result = deepcopy(self)
            for k in range(self.log2n + 1):
                result.partialSums[k] += other.partialSums[k]
            return result
        elif isinstance(other, np.ndarray):
            if other.shape != (self.n,):
                raise ValueError ("WeightVec and array are incompatible shapes: " + str(self.n) + " vs " + str(other.shape[0]))
            result = deepcopy(self)
            result.partialSums[0] += other
            result.rebuildPartialSums()
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for k in range(self.log2n + 1):
                result.partialSums[k] += other * (2 ** k)
            return result
        else:
            return NotImplemented

    def __radd__ (self, other):
        return self.__add__(other)
    
    def __sub__ (self, other):
        if isinstance(other, WeightVec):
            if self.n != other.n:
                raise ValueError ("WeightVecs are different lengths: " + str(self.n) + " vs " + str(other.n))
            result = deepcopy(self)
            for k in range(self.log2n + 1):
                result.partialSums[k] -= other.partialSums[k]
            return result
        elif isinstance(other, np.ndarray):
            if other.shape != (self.n,):
                raise ValueError ("WeightVec and array are incompatible shapes: " + str(self.n) + " vs " + str(other.shape[0]))
            result = deepcopy(self)
            result.partialSums[0] -= other
            result.rebuildPartialSums()
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for k in range(self.log2n + 1):
                result.partialSums[k] -= other * (2 ** k)
            return result
        else:
            return NotImplemented
    
    def __rsub__ (self, other):
        return self.__sub__(other)
    
    def __mul__ (self, other):
        if isinstance(other, WeightVec):
            if self.n != other.n:
                raise ValueError ("WeightVecs are different lengths: " + str(self.n) + " vs " + str(other.n))
            return self * other.partialSums[0]
        elif isinstance(other, np.ndarray):
            if other.shape != (self.n,):
                raise ValueError ("WeightVec and array are incompatible shapes: " + str(self.n) + " vs " + str(other.shape[0]))
            result = deepcopy(self)
            result.partialSums[0] *= other
            result.rebuildPartialSums()
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for k in range(self.log2n + 1):
                result.partialSums[k] *= other
            return result
        else:
            return NotImplemented

    def __rmul__ (self, other):
        return self.__mul__(other)

    def __truediv__ (self, other):
        if isinstance(other, WeightVec):
            if self.n != other.n:
                raise ValueError ("WeightVecs are different lengths: " + str(self.n) + " vs " + str(other.n))
            return self / other.partialSums[0]
        elif isinstance(other, np.ndarray):
            if other.shape != (self.n,):
                raise ValueError ("WeightVec and array are incompatible shapes: " + str(self.n) + " vs " + str(other.shape[0]))
            result = deepcopy(self)
            result.partialSums[0] /= other
            result.rebuildPartialSums()
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for k in range(self.log2n + 1):
                result.partialSums[k] /= other
            return result
        else:
            return NotImplemented
    
    def __rtruediv__ (self, other):
        return self.__truediv__(other)

    def sum (self):
        return self.partialSums[self.log2n][0]

    def invCDF (self, p):
        if p < 0 or p >= 1:
            raise ValueError ("Invalid probability: " + str(p))
        totalWeight = self.partialSums[self.log2n][0]
        if totalWeight <= 0:
            return None
        weight = p * totalWeight
        index = 0
        for k in range(self.log2n - 1, -1, -1):
            index <<= 1
            if weight >= self.partialSums[k][index]:
                weight -= self.partialSums[k][index]
                index += 1
        return index

    def sample (self):
        return self.invCDF(np.random.uniform())

    def sampleWithoutReplacement (self):
        if self.sum() <= 0:
            return None
        index = self.vec.sample()
        self[index] = 0
        return index

# IntervalWeightVec is a WeightVec wrapper that is indexed by intervals over some range [0,L)
# Stores weights for all tuples (i,j) with 0 <= i <= j < L
class IntervalWeightVec:
    def __init__(self, L, initVal=None, dtype=np.float16):
        self.L = L
        self.vec = WeightVec(L*L, dtype=dtype)
        if initVal is not None:
            for ij in iter(self):
                self.vec.partialSums[0][self.intervalIndex(*ij)] = initVal
            self.vec.rebuildPartialSums()
    
    def intervalIndex (self, start, end):
        assert 0 <= start <= end < self.L
        return start * self.L + end
    
    def indexInterval (self, index):
        return divmod(index, self.L)

    def __iter__ (self):
        for i in range(self.L):
            for j in range(i, self.L):
                yield (i,j)

    def __setitem__ (self, key, weight):
        start, end = key
        self.vec[self.intervalIndex(start, end)] = weight
    
    def __getitem__ (self, key):
        start, end = key
        return self.vec[self.intervalIndex(start, end)]
    
    def __len__ (self):
        return self.L

    def __str__ (self):
        return str([self[ij] for ij in iter(self)])

    def sum (self):
        return self.vec.sum()

    def invCDF (self, p):
        return self.indexInterval(self.vec.invCDF(p))

    def sample (self):
        return self.invCDF(np.random.uniform())

    def sampleWithoutReplacement (self):
        if self.sum() <= 0:
            return None
        interval = self.vec.sample()
        self[interval] = 0
        return interval

    def __add__ (self, other):
        if isinstance(other, IntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("IntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec += other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] += other
            return result
        else:
            return NotImplemented
        
    def __radd__ (self, other):
        return self.__add__(other)

    def __sub__ (self, other):
        if isinstance(other, IntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("IntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec -= other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] -= other
            return result
        else:
            return NotImplemented
    
    def __rsub__ (self, other):
        return self.__sub__(other)

    def __mul__ (self, other):
        if isinstance(other, IntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("IntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec *= other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] *= other
            return result
        else:
            return NotImplemented
    
    def __rmul__ (self, other):
        return self.__mul__(other)
    
    def __truediv__ (self, other):
        if isinstance(other, IntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("IntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec /= other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] /= other
            return result
        else:
            return NotImplemented
    
    def __rtruediv__ (self, other):
        return self.__truediv__(other)


# FastIntervalWeightVec has the same interface as IntervalWeightVec, except for invCDF and sample methods,
# It does not store the partial sums, so is suitable as a scratch structure
# Stores weights for all tuples (i,j) with 0 <= i <= j < L
class FastIntervalWeightVec:
    def __init__(self, L, initVal=None, dtype=np.float16):
        self.L = L
        self.vec = np.zeros (L*L, dtype=dtype)
        if initVal is not None:
            for ij in iter(self):
                self.vec[self.intervalIndex(*ij)] = initVal
    
    def intervalIndex (self, start, end):
        assert 0 <= start <= end < self.L
        return start * self.L + end
    
    def indexInterval (self, index):
        return divmod(index, self.L)

    def __iter__ (self):
        for i in range(self.L):
            for j in range(i, self.L):
                yield (i,j)

    def __setitem__ (self, key, weight):
        start, end = key
        self.vec[self.intervalIndex(start, end)] = weight
    
    def __getitem__ (self, key):
        start, end = key
        return self.vec[self.intervalIndex(start, end)]
    
    def __len__ (self):
        return self.L

    def __str__ (self):
        return str([self[ij] for ij in iter(self)])

    def sum (self):
        return np.sum(self.vec)

    def intervalWeightVec (self):
        result = IntervalWeightVec(self.L)
        result.vec.partialSums[0] = deepcopy(self.vec)
        result.rebuildPartialSums()
        return result

    def __add__ (self, other):
        if isinstance(other, FastIntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("FastIntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec += other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] += other
            return result
        else:
            return NotImplemented
        
    def __radd__ (self, other):
        return self.__add__(other)

    def __sub__ (self, other):
        if isinstance(other, FastIntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("FastIntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec -= other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] -= other
            return result
        else:
            return NotImplemented
    
    def __rsub__ (self, other):
        return self.__sub__(other)

    def __mul__ (self, other):
        if isinstance(other, FastIntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("FastIntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec *= other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] *= other
            return result
        else:
            return NotImplemented
    
    def __rmul__ (self, other):
        return self.__mul__(other)
    
    def __truediv__ (self, other):
        if isinstance(other, FastIntervalWeightVec):
            if self.L != other.L:
                raise ValueError ("FastIntervalWeightVecs are different lengths: " + str(self.L) + " vs " + str(other.L))
            result = deepcopy(self)
            result.vec /= other.vec
            return result
        elif isinstance(other, (int, float, np.number)):
            result = deepcopy(self)
            for ij in iter(result):
                result[ij] /= other
            return result
        else:
            return NotImplemented
    
    def __rtruediv__ (self, other):
        return self.__truediv__(other)
