# tomatofft
The Tomato Patch FFT is the fastest FFT in the world- but it may not be the most accurate.
This github repository is a scientific landmark, or perhaps it's just a trash dump.
it all started one day when I was experimenting with python.
I thought if i refactored fft, i could make a more efficient fft. That's kind of partially true.. but..
The radix-2 fft already is highly optimal. 
But I found a way to beat it. 

After refactoring all the complex operations, and then performing all of them, i came up with some pretty images:
https://imgur.com/a/3sFjGIW
The real and the complex became two arrays of 512x512. To perform the rfft, you needed to multiply the input by all elements along the rows,
then sum it by the columns- not a pretty picture in terms of the complexity required. That's a lot of math!

However- my obsession didn't stop there, and i figured out that i could sort the fft and reverse the sorting.
I then found the basis function, which is, for the most part, a sigmoid. Who knew!

```py

d1 = numpy.zeros((512,512), dtype=numpy.half)
d2= numpy.zeros((512,512), dtype=numpy.half)

tn = numpy.sort(twiddle.real,axis=1)
t44 = numpy.sort(twiddle.imag,axis=1)


tz = numpy.argsort(numpy.arctan(twiddle.real),axis=1)
reversed_seq = np.argsort(tz,axis=1).T
tzi = numpy.argsort(numpy.arctan(twiddle.imag),axis=1)
reversed_seq_i = np.argsort(tzi,axis=1).T


tq = numpy.zeros(512,dtype=numpy.half)
for each in range(512):
  tq[each] = numpy.sum(tn[:,each])

tdec = tq/512

for i in range(512):
  d2[i,:]= (data[i])

d3 =d2.copy()

for j in range(512):
    d2[:,j] *= tdec[j]

d2 = d3 *tminus + d2

d1 = d2.copy()

d1 = d1[np.arange(d1.shape[1]),reversed_seqr].T
d2 = d2[np.arange(d2.shape[1]),reversed_seqi].T


result = numpy.zeros(512, dtype=numpy.csingle)
for i in range(512):
  result[i] = numpy.sum(d1[:,i]) + 1j * numpy.sum(d2[:,i])
```

Note that in the code, we are simply filling the array with one set of multiplications(ordinary multiplications, N in number), and then there is N * N additions,
and then we need to do N swaps, and then there's a pair of transposes- 2 * N*(N-1)/2 swaps - , and then we need to do N*N additions(in sequence!), and then we're done.
Total : 2 * N * N additions, N multiplications, N^2 swaps.
The overall time complexity is - O(N^2). it might be a little bit higher if there's more tuning needed.

For radix-RFFT(in frequency, which is the least compute intensive,but which requires more space), you need N * 2^k * 4  multiplications, 
which become O(N log N) real multiplications and N * 2^k * 4 additions(we'll count subtractions as additions for simplicity),
which refactor to O(N log N) time complexity.

The difference is  N / log N. At less than 512 elements or equal to it, it's faster than radix, requiring less operations overall.
Additionally, further optimizations may be possible that I just havn't figured out.

All of tomatopatch FFT is elementwise. Most is highly vectorizable. None requires complex math, which means it is highly acceleratable on any platform and invariant to precision issues.
However, at this time, because the basis function is not truly a sigmoid(see the tminus image), the results are not perfectly precise, but they are CLOSE.
For the first half of the real fourier transform- they are identical.

Now, here's the complexity- you need more storage space. You need more space to do the computations. A lot more. You need a total of N * N * 5 bytes, fully unrolled.
Given the above, it's possible you could just accumulate it, but you still need space for the transforms and the residual(not all rows need to be stored, i think). I estimate it's on the order of 3* N * N bytes total. 
For radix-FFT, you only need a little over 2^k * N * 2 bytes.

So, depending on your time-space tradeoffs, tomato patch FFT may in fact be a more optimal, faster, better approach. It does not depend on any convolution.
It can be incorporated into tensors or artificial intelligence methods, especially so, because of the higher optimal behavior.
Without the residual, the precision suffers somewhat, but it's still probably useful for AI purposes(invertible to within 99% accuracy),
and requires less additions and a lot less space- you need only N * N * 2, plus a few lines for the accumulators.


