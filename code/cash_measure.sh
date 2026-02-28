INPUT=code/inputs/timeinput/abundant_wires.txt
MODE=A
BATCH=4
SA_ITERS=5
SA_PROB=0.1

for N in 1 2 4 8 16; do
  perf stat -x, -o total_N${N}.csv -e cache-misses -- \
    ./wireroute -f $INPUT -n $N -m $MODE -b $BATCH -i $SA_ITERS -p $SA_PROB

  perf stat --per-thread -x, -o threads_N${N}.csv -e cache-misses -- \
    ./wireroute -f $INPUT -n $N -m $MODE -b $BATCH -i $SA_ITERS -p $SA_PROB

  mean=$(awk -F, '$3=="cache-misses" && $1 ~ /^[0-9]/ {sum+=$1; cnt+=1} END {if(cnt) print sum/cnt; else print "NA"}' threads_N${N}.csv)
  echo "$N,$mean" >> mean_per_thread.csv
done