SIZE=(10 100 1000 10000 100000 1000000 10000000 100000000)
nvcc -o timed_vector_add.out timed_vector_add.cu

echo "problem_size, device_time, host_time">>output.txt
for s in ${SIZE[@]}; do
    ./timed_vector_add.out $s >>output.txt
done
