SIZE=(10 100 1000 10000 100000 1000000 10000000 100000000)
nvcc -o timed_vector_add.out timed_vector_add.cu
TH=256

echo "problem_size, blocksPerGrid, threadsPerBlock, device_time, host_time">>output.txt
for s in ${SIZE[@]}; do
    ./timed_vector_add.out $s $TH>>output.txt
done
