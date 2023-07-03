# tomatofft
The Tomato Patch FFT is the fastest, laziest FFT in the world. But it requires more work- it is O(N^2) overall.
Additionally- it may already exist. Which is why I named my implementation after my cat.

and requires more storage- O(N^2) overall. However, it has one difference which makes all the difference in the world- it can be threaded down to individual bins
where it becomes O(2N) per thread, and since all operations are elementwise, it can at minimum become O(N^2/P) - O(N).\
With cuda cores = N(modern GPUS are quickly exceeding 4000 cuda cores) and general memory constraints per core of P * N ^ N * precision 16 * N^2 bytes + 8 * N bytes  = a little over 256MB.\
For an FFT of size 32768, you will need 32,768 cuda cores and 16GB of main memory(RTX 4090 is almost halfway there, so assume another 8 years).\
What this means is that for certain implementations, computation quickly approaches O(1).

The algorithm is MIT licensed because you really can't patent this sort of thing, it's just the FFT.

This github repository is a scientific landmark, or perhaps it's just a trash dump.
it all started the other day when I was experimenting with python on a sweltering sunny day.
I thought if i refactored fft, i could make a more efficient fft. That's kind of partially true.. but..
The radix-2 fft already is highly optimal. It's really, really optimal. it reuses products to minimize computations.
And you should NOT be using tomatofft for any implementation where you are compute limited..
But I found a way to beat it. 

After refactoring all the complex operations, and then performing all of them, i came up with some pretty images:
https://imgur.com/a/3sFjGIW
The real and the complex became two arrays of 512x512. To perform the rfft, you needed to multiply the input by all elements along the rows,
then sum it by the columns- not a pretty picture in terms of the complexity required. That's a lot of math!

However- my obsession didn't stop there, and i figured out that i could sort the fft and reverse the sorting.
I then found the basis function, which is, for the most part, a sigmoid. Who knew!

Here is some code to use the identity matrix from the fft product:
```py
reversed_seq= np.loadtxt('https://raw.githubusercontent.com/falseywinchnet/tomatofft/main/reversed_seq.txt', dtype=numpy.complex128).astype(dtype=int)
reversed_seq_i= np.loadtxt('https://raw.githubusercontent.com/falseywinchnet/tomatofft/main/reversed_seq_i.txt', dtype=numpy.complex128).astype(dtype=int)
twiddle_data = np.loadtxt('https://raw.githubusercontent.com/falseywinchnet/tomatofft/main/twiddle_data.txt', dtype=numpy.complex128)

d1 = numpy.zeros((512,512), dtype=numpy.complex128)
d2= numpy.zeros((512,512), dtype=numpy.complex128)

tn = numpy.sort(twiddle_data.real,axis=1)

tq = numpy.zeros(512,dtype=numpy.half)
for each in range(512):
  tq[each] = numpy.sum(tn[:,each])

tdec = tq/512 #is a sigmoid? wow!

for i in range(512):
  d2[i,:]= (data[i]) #this is the only columnorder operation needed

d3 =d2.copy()

for j in range(512):
    d2[j,:] *= tdec[:] #row order

d2 = d2 # +  d3 *tminus note: to get tminus(the residual distribution), just subtract tq/512 from tn. 

d1 = d2.copy()

d1 = d1[np.arange(d1.shape[0]),reversed_seqr] #row order
d2 = d2[np.arange(d2.shape[0]),reversed_seqi]#row order
print(d1.shape)

result = numpy.zeros(512, dtype=numpy.csingle)
for i in range(512):
  result[i] = numpy.sum(d1[i,:]) + 1j * numpy.sum(d2[i,:]) #row order
```
And here is some code to compute the FFT according to the tomatoFFT with the minimum in operations.
Note that the fft twiddle factors here were computed with 256 bits of precision and may not precisely match
the output of numpy.fft.rfft excepting as the twiddle factors are recomputed and nyquist point is handled properly:
```py:
import numpy
import numba

from matplotlib import pyplot as plt

twiddle_data = numpy.loadtxt('https://raw.githubusercontent.com/falseywinchnet/tomatofft/main/twiddle_data.txt', dtype=numpy.complex128)

@numba.njit(numba.float64(numba.float64[:],numba.float64[:]),parallel=True, nogil=True,cache=True,fastmath=True,error_model='numpy')
def f(a:numpy.ndarray, b:numpy.ndarray):
  p = 0.0
  for index in numba.prange(512):
    p += a[index]*b[index] # may require a floating point accumulator to handle minor errors?
  return p

@numba.njit(numba.complex128[:](numba.float64[:],numba.float64[:,:],numba.float64[:,:]),nogil=True,cache=True,fastmath=True,error_model='numpy')
def tomato_fft(input:numpy.ndarray,real:numpy.ndarray,imag:numpy.ndarray):
  result = numpy.arange(0,512,dtype=numpy.complex128)
  for n in range(512):
      result[n] = f(input,real[n,:])  + 1j * f(input,imag[n,:])
  return result

#example use:
input = numpy.arange(0,512,dtype=numpy.float64)
output = tomato_fft(input,twiddle_data.real,twiddle_data.imag)
```
The operations presented are 2N multiplications, 2N additions, for each element,
which means that overall the complexity is 2N^N(in big O, that's O(N^2)) but in a parallelized instance,
we can return results in 2N. 
in terms of numerical precision, it's possible to apply the multiplication, then sort according to a known, predetermined pattern(the identity matrix forward)
and then use an accumulator with known positive offsets and another one with known negative offsets(sorted, you have negative followed by positive)
and then you can accumulate the range and eliminate floating point errors, by performing A additions of negative and B additions of positive, and then adding them together-
it requires a maximum of N/2 swaps along with +1 additions over the normal summation, but allows for an even smaller data type due to not needing to worry about accumulation of error, ie.
since we can pre-record swapping as a selective transpose, it means that one can very reasonably build an FFT out of nothing but gates,
and compute an entire FFT in one cycle- and what's more, you could have an analogue FFT. true, you will have decimation, each point in the FFT is an approximation,
but in terms of the resolution, you could compute a really high resolution, high precision fft and have floating point accuracy of some 1024+ bits, 
and then use that to make an analogue multiplier circuit, and build accumulators and networks for each bin, and in the end, apply a specific time delay
to each output, and your output will be a time-> frequency transposing circuit, which, owing to the invertibility of this method by simply using the conjugate,
means that you can transform from time to frequency and back to time in an analog fashion, within the limits of accuracy and noise in your circuit.


For radix-RFFT(in frequency, which is the least compute intensive,but which requires more total time), the total
and the time complexity are both O(N log N) time complexity.

Space considerations are also massively different- radix requires a total of 2^k * N * 2 elements,
and tomatofft requires 2N^N elements(but again, per thread only requires 2N).

Depending on your time-space tradeoffs, tomato patch FFT may in fact be a more optimal, faster, better approach. It does not depend on any convolution. It can be incorporated into tensors or artificial intelligence methods, and is MIT licensed.

to get numerically accurate results that are identical to numpy, use the following code:
```py
import io
import re
import numpy
import numpy as np
import sys
sys.setrecursionlimit(3000)  # or a higher value if needed
import sympy
from sympy.abc import x, m,t


def rfft_unrolled(input_data:np.ndarray, twiddlefactors: np.ndarray):
  buffer = io.StringIO()


  M = np.asarray([[1.+0.0000000e+00j, 1.+0.0000000e+00j],[ 1.+0.0000000e+00j, -1.-1.2246468e-16j]],dtype=numpy.complex128) #butterfly matrix

  X_stage_0 = np.zeros((2, 256), dtype=np.complex128)
  X_stage_1 = np.zeros((4, 128), dtype=np.complex128)
  X_stage_2 = np.zeros((8, 64), dtype=np.complex128)
  X_stage_3 = np.zeros((16, 32), dtype=np.complex128)
  X_stage_4 = np.zeros((32, 16), dtype=np.complex128)
  X_stage_5 = np.zeros((64, 8), dtype=np.complex128)
  X_stage_6 = np.zeros((128, 4), dtype=np.complex128)
  X_stage_7 = np.zeros((256, 2), dtype=np.complex128)
  X_stage_8 = np.zeros((512, 1), dtype=np.complex128)

  print('Stage 0')

  for i in range(256):
        X_stage_0[0, i] = M[0, 0] * input_data[i] + M[0, 1] * input_data[256+i]
        X_stage_0[1, i] = M[1, 0] * input_data[i] + M[1, 1] * input_data[256+i]
        buffer.write(f'S0E{0,i} = M(0) * input_data({i}) + M(1) * input_data({i+256});')
        buffer.write(f'S0E{1,i} = M(2) * input_data({i}) + M(3) * input_data({i+256});')

  e = 2
  q = 128
  print('Stage 1')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_1[i, j] = X_stage_0[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_0[i, q+j]
        buffer.write(f'S1E{i,j}  = x0({i}, {j}) + tw({twiddle_index}) * x0({i}, {q+j});')
        X_stage_1[e+i, j] = X_stage_0[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_0[i, q+j]
        buffer.write(f'S1E{e+i,j}  = x0({i}, {j}) - tw({twiddle_index}) * x0({i}, {q+j});')

  e = 4
  q = 64
  print('Stage 2')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_2[i, j] = X_stage_1[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_1[i, q+j]
        buffer.write(f'S2E{i,j} = x1({i}, {j}) + tw({twiddle_index}) * x1({i}, {q+j});')
        X_stage_2[e+i, j] = X_stage_1[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_1[i, q+j]
        buffer.write(f'S2E{e+i,j} = x1({i}, {j}) - tw({twiddle_index}) * x1({i}, {q+j});')


  e = 8
  q = 32
  print('Stage 3')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_3[i, j] = X_stage_2[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_2[i, q+j]
        buffer.write(f'S3E{i,j} = x2({i}, {j}) + tw({twiddle_index}) * x2({i}, {q+j});')
        X_stage_3[e+i, j] = X_stage_2[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_2[i, q+j]
        buffer.write(f'S3E{e+i,j}  = x2({i}, {j}) - tw({twiddle_index}) * x2({i}, {q+j});')



  e = 16
  q = 16
  print('Stage 4')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_4[i, j] = X_stage_3[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_3[i, q+j]
        buffer.write(f'S4E{i,j}  = x3({i}, {j}) + tw({twiddle_index}) * x3({i}, {q+j});')
        X_stage_4[e+i, j] = X_stage_3[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_3[i, q+j]
        buffer.write(f'S4E{e+i,j}  = x3({i}, {j}) - tw({twiddle_index}) * x3({i}, {q+j});')

  e = 32
  q = 8
  print('Stage 5')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_5[i, j] = X_stage_4[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_4[i, q+j]
        buffer.write(f'S5E{i,j}  = x4({i}, {j}) + tw({twiddle_index}) * x4({i}, {q+j});')
        X_stage_5[e+i, j] = X_stage_4[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_4[i, q+j]
        buffer.write(f'S5E{e+i,j}  = x4({i}, {j}) - tw({twiddle_index}) * x4({i}, {q+j});')

  e = 64
  q = 4
  print('Stage 6')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_6[i, j] = X_stage_5[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_5[i, q+j]
        buffer.write(f'S6E{i,j}  = x5({i}, {j}) + tw({twiddle_index}) * x5({i}, {q+j});')
        X_stage_6[e+i, j] = X_stage_5[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_5[i, q+j]
        buffer.write(f'S6E{e+i,j}  = x5({i}, {j}) - tw({twiddle_index}) * x5({i}, {q+j});')


  e = 128
  q = 2
  print('Stage 7')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_7[i, j] = X_stage_6[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_6[i, q+j]
        buffer.write(f'S7E{i,j}  = x6({i}, {j}) + tw({twiddle_index}) * x6({i}, {q+j});')
        X_stage_7[e+i, j] = X_stage_6[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_6[i, q+j]
        buffer.write(f'S7E{e+i,j}  = x6({i}, {j}) - tw({twiddle_index}) * x6({i}, {q+j});')


  e = 256
  q = 1
  print('Stage 8')
  for i in range(e):
    for j in range(q):
        twiddle_index = i + e - 1
        X_stage_8[i, j] = X_stage_7[i, j] + twiddlefactors[twiddle_index, 0] * X_stage_7[i, q+j]
        buffer.write(f'S8E{i,j}  = x7({i}, {j}) + tw({twiddle_index}) * x7({i}, {q+j});')
        X_stage_8[e+i, j] = X_stage_7[i, j] - twiddlefactors[twiddle_index, 0] * X_stage_7[i, q+j]
        buffer.write(f'S8E{e+i,j}  = x7({i}, {j}) - tw({twiddle_index}) * x7({i}, {q+j});')



# Handle Nyquist frequency
#  X_stage_8[512//2 + 1,0] = X_stage_8[512//2 + 1,0].real + 0.0j
  return  buffer.getvalue(), X_stage_8[:512//2 + 1,0]#return only the first half- the real complex
#note that this actually generates the entire FFT up to 2^k where N = 512, and for larger N/
#etc all you need to do is add more stages and compute the twiddle factor to a higher param

def parse_log(buffer):
    mapping = {}
    log_items = buffer.split(';')
    for item in log_items:
        if ' = ' in item:  # avoid empty strings
            key, equation = item.split(' = ')
            # reconstruct key to match format in equations
            key = key.replace('S', 'x').replace('E', '').strip()
            equation = equation.strip()
            mapping[key] = equation
    return mapping

def replace_parentheses_with_brackets(mapping):
    new_mapping = {}
    for key, value in mapping.items():
        new_key = key.replace('(', '[').replace(')', ']')
        new_value = value.replace('(', '[').replace(')', ']')
        new_mapping[new_key] = new_value
    return new_mapping

def collect_mapping(key, mapping, collected_mapping):
    if key in mapping and key not in collected_mapping:
        equation = mapping[key]
        collected_mapping[key] = equation
        variables = find_variables(equation)
        for var in variables:
            if var in mapping:
                collect_mapping(var, mapping, collected_mapping)
    return collected_mapping

def find_variables(expression):
    variables = []
    pattern = r'x\d+\[\d+,\s*\d+\]'
    matches = re.findall(pattern, expression)
    variables.extend(matches)
    return variables


def retrieve_mapping(mapping):
    collected_mapping = {}
    target_key = 'x8[512, 0]'
    collect_mapping(target_key, mapping, collected_mapping)
    return collected_mapping

def expand_formula(formula_dict):
    # Initial formula
    formula = list(formula_dict.values())[0]

    # While the formula still contains variables
    while any(f'x{n}[' in formula for n in range(10)):
        # Loop over all variables in the dictionary
        for variable, value in formula_dict.items():
            # If the variable is in the formula
            if variable in formula:
                # Replace the variable with its value enclosed in parentheses
                formula = formula.replace(variable, f"({value})")

    return {"expanded_formula": formula}

def retrieve_and_expand_mappings(mapping):
    expanded_mapping = {}

    for i in range(512):
        collected_mapping = {}
        target_key = f'x8[{i}, 0]'
        collect_mapping(target_key, mapping, collected_mapping)

        # expand the formula
        expanded_formula = expand_formula(collected_mapping)
        v5 = expanded_formula["expanded_formula"]
        v5 = v5.replace('input_data', 'x').replace('M', 'm').replace('tw', 't').replace('[', '').replace(']', '')
        v5 = str(sympy.expand(v5))
        pattern = r'\d+'
        v5 = re.sub(pattern, lambda m: '[' + m.group(0) + ']', v5)
        v5 = v5.replace("+", ", +").replace("-", ", -").lstrip(", ")

        items = v5.split(',')
        # Process each item and store the results in a list
        processed_items = []
        for item in items:
          terms = item.split('*')
          var = int(re.findall(r'x\[(\d+)\]', item)[0])  # Extract the number
          processed_items.append({'item': item, 'var': var})

        # Sort the items by the number next to x
        processed_items.sort(key=lambda item: item['var'])

        # Generate new formula and coefficients
        new_formula = []
        coefficients = []

        for item in processed_items:
            item_str = item['item'].strip()  # remove leading/trailing spaces
            coeff = "false" if '-' in item_str else "true"
            item_str = item_str.replace("-", "").replace("+", "")  # remove  '-'
            item_str = item_str.rsplit('*x', 1)[0] #strip off the x
            coefficients.append(coeff)
            new_formula.append(item_str)

        new_formula = ' , '.join(new_formula)
        coefficients = ' , '.join(coefficients)

        expanded_mapping[f'tw{i}'] = new_formula
        expanded_mapping[f'c{i}'] = coefficients

    return expanded_mapping

def evaluate_mappings(expanded_mapping,m,t):
    complex_array = np.zeros((512, 512), dtype=precision)

    for i in range(512):
        mapping = expanded_mapping[f'tw{i}']
        coefficient = expanded_mapping[f'c{i}']

        items = mapping.split(', ')
        coeff_items = coefficient.split(', ')

        for j, item in enumerate(items):
            # Store the result in the complex array
            item = item.rsplit('*x', 1)[0] #strip off the x
            item = item.strip()
            complex_array[i, j] = eval(item)
            complex_array[i, j] = -1 * complex_array[i, j].real + 1j*complex_array[i, j].imag if 'false' in coeff_items[j] else complex_array[i, j].real + -1j*complex_array[i, j].imag
            # If the coefficient is '-', store False in the bool array. Else store True.

    return complex_array

precision = numpy.complex256 #set this appropriately for the precision numpy.pi can achieve
#note that in TW `11` should be k+2 for 2^k, for "512" you have 11, for larger, go higher
tw = [np.exp(-1.0 * 1.0j * np.pi * np.arange(((2**i)/2), dtype=precision) / ((2**i)/2)) for i in range(1,11)]
list_of_lists = [list(map(lambda x: [x.astype(precision)], arr)) for arr in tw]
twiddlefactors = np.concatenate(list_of_lists)
inverse = np.conj(twiddlefactors) #use this for irfft

log, e = rfft_unrolled(numpy.arange(0,512),twiddlefactors)#the actual inputs are irrelevant unless verifying this method
v1 = parse_log(log)
v2 = replace_parentheses_with_brackets(v1)

N = 2 #initial butterfly factor size
M = np.zeros((N, N), dtype=precision)
for i in range(N):
    for j in range(N):
        M[i, j] = np.exp(1 * 2j * np.pi * i * j / N)

m = M.flatten()

mappings = retrieve_and_expand_mappings(v2) #this may take a long time due to sympy being slow.
twiddle = evaluate_mappings(mappings,m,numpy.squeeze(twiddlefactors))
#you might consider an alternative library if it generates identical results.
#your output results should behave similar to the twiddle as indicated in the demo at the top of this page,
#but be substantially closer to numpy- i computed the twiddle factors and butterfly matrix using mpc with 19 bits of #decimal precision, which is the maximum for complex256 accuracy.
```
