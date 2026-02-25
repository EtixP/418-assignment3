#To run, type "bash bench_across.sh" in bash
export OMP_PROC_BIND=true
export OMP_PLACES=cores

out=outputs/across_wires_bench.csv
echo "input,threads,run,init_sec,compute_sec,max_occ,total_cost,validate" > "$out"

for f in inputs/timeinput/*.txt; do
  for t in 1 2 4 8; do
    for r in 1; do
      log=$(./wireroute -f "$f" -n "$t" -m A -b 1 -i 5 -p 0.1)
      init=$(echo "$log" | awk -F': ' '/Initialization time/{print $2}')
      comp=$(echo "$log" | awk -F': ' '/Computation time/{print $2}')
      maxo=$(echo "$log" | awk -F': ' '/Max occupancy/{print $2}')
      cost=$(echo "$log" | awk -F': ' '/Total cost/{print $2}')
      val=$(echo "$log" | awk '/Validate Passed/{print "pass"} /Validate:/{print "fail"}' | tail -n1)
      echo "$(basename "$f"),$t,$r,$init,$comp,$maxo,$cost,$val" >> "$out"
    done
  done
done