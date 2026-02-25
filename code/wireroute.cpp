/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <climits>
#include <atomic>

#include <omp.h>
#include <unistd.h>
using namespace std;


void print_stats(const std::vector<std::vector<int>> &occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto &row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

/* This function write the output into 2 files
(1) It write occupancy grids into a file
(2) It convert wires from Wire to validate_wire_t by to_validate_format
(2) It write wires into another file
*/

struct Candidate{
  long long cost;
  Wire route;    
};

void write_output(
    const std::vector<Wire> &wires, const int num_wires,
    const std::vector<std::vector<int>> &occupancy, const int dim_x,
    const int dim_y,
    std::string wires_output_file_path = "outputs/wire_output.txt",
    std::string occupancy_output_file_path = "outputs/occ_output.txt") {

  std::ofstream out_occupancy(occupancy_output_file_path, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }
  out_occupancy << dim_x << ' ' << dim_y << '\n';

  for (const auto &row : occupancy) {
    for (size_t i = 0; i < row.size(); ++i)
      out_occupancy << row[i] << (i == row.size() - 1 ? "" : " ");
    out_occupancy << '\n';
  }
  out_occupancy.close();

  std::ofstream out_wires(wires_output_file_path, std::fstream::out);
  if (!out_wires) {
    std::cerr << "Unable to open file: " << wires_output_file_path << '\n';
    exit(EXIT_FAILURE);
  }

  out_wires << dim_x << ' ' << dim_y << '\n';
  out_wires << num_wires << '\n';

  for (const auto &wire : wires) {
    // NOTICE: we convert to keypoint representation here, using
    // to_validate_format which need to be defined in the bottom of this file
    validate_wire_t keypoints = wire.to_validate_format();
    for (int i = 0; i < keypoints.num_pts; ++i) {
      out_wires << keypoints.p[i].x << ' ' << keypoints.p[i].y;
      if (i < keypoints.num_pts - 1)
        out_wires << ' ';
    }
    out_wires << '\n';
  }

  out_wires.close();
}

// Helper function to either subtract or add wire in occupancy
void apply_wire(const Wire &w, std::vector<std::vector<int>> &occupancy, int change){
  validate_wire_t vw = w.to_validate_format().cleanup();

  for(int i=0; i<vw.num_pts-1; i++){
    int x=vw.p[i].x;
    int y=vw.p[i].y;
    int xn=vw.p[i+1].x;
    int yn=vw.p[i+1].y;
    int sx=(xn>x)?1:(xn<x?-1:0);
    int sy=(yn>y)?1:(yn<y?-1:0);
    
    while(x!=xn || y!=yn){
      occupancy[y][x] += change;
      x+=sx;
      y+=sy;
    }
    if(i==vw.num_pts-2) occupancy[y][x]+=change; //Account for last point
  }
}

int sgn(int v){
  return (v>0)-(v<0);
}

int count_candidates(const Wire &w){
  int dx = std::abs(w.end_x-w.start_x);
  int dy = std::abs(w.end_y-w.start_y);

  if(dx==0||dy==0) return 1;
  return 2+(dx-1)+(dy-1)+2*(dx-1)*(dy-1);
}

Wire candidate_from_id(const Wire &base, int cid){
  int x0 = base.start_x, y0 = base.start_y;
  int x1 = base.end_x, y1 = base.end_y;
  int dx = std::abs(x1-x0), dy = std::abs(y1-y0);
  int sx = sgn(x1-x0), sy = sgn(y1-y0);

  Wire w = base;

  //Straight line
  if (dx == 0 || dy == 0) {
    w.move_x_start = true;
    w.move_x_end = false;
    w.mid_x = x1;
    w.mid_y = y0;
    return w;
  }

  //1 bend
  if(cid == 0) { // horizontal first
    w.move_x_start = true;
    w.move_x_end = false;
    w.mid_x = x1;
    w.mid_y = y0;
    return w;
  }
  if(cid == 1) { // vertical first
    w.move_x_start = false;
    w.move_x_end = true;
    w.mid_x = x0;
    w.mid_y = y1;
    return w;
  }
  cid -= 2;

  //2 bend
  if(cid<dx-1){ 
    int xm = x0+sx*(cid+1);
    w.move_x_start = true;
    w.move_x_end = true;
    w.mid_x = xm;
    w.mid_y = y1;
    return w;
  }
  cid -= (dx-1);

  if(cid < dy - 1) {
    int ym = y0 + sy * (cid + 1); // interior y
    w.move_x_start = false;
    w.move_x_end = false;
    w.mid_x = x1;
    w.mid_y = ym;
    return w;
  }
  cid -= (dy-1);

  //3 bend
  int interior = (dx - 1)*(dy - 1);
  int orient = cid / interior;      // 0: HF, 1: VF
  int idx = cid % interior;

  int k = idx % (dx - 1) + 1;       // x interior index
  int l = idx / (dx - 1) + 1;       // y interior index
  int xm = x0 + sx * k;
  int ym = y0 + sy * l;

  w.mid_x = xm;
  w.mid_y = ym;
  if (orient == 0) {                 // Horizontal: x->y then x->y
    w.move_x_start = true;
    w.move_x_end = true;
  } else {                           // Vertical: y->x then y->x
    w.move_x_start = false;
    w.move_x_end = false;
  }
  return w;

}

long long route_add_cost(const Wire &candi, const std::vector<std::vector<int>> &occupancy){
  validate_wire_t vw = candi.to_validate_format().cleanup();
  long long tot = 0;
  for (int i = 0; i < vw.num_pts - 1; ++i) {
    int x = vw.p[i].x,     y = vw.p[i].y;
    int xn = vw.p[i + 1].x, yn = vw.p[i + 1].y;
    int sx = (xn > x) ? 1 : (xn < x ? -1 : 0);
    int sy = (yn > y) ? 1 : (yn < y ? -1 : 0);

    while (x != xn || y != yn) {
      int n = occupancy[y][x];
      tot += 2LL * n + 1;            // (n+1)^2 - n^2
      x += sx;
      y += sy;
    }
    if (i == vw.num_pts - 2) {  // final endpoint exactly once
      int n = occupancy[y][x];
      tot += 2LL * n + 1;
    }
  }
  return tot;
}

Candidate find_best_serial(const Wire &wire, const vector<vector<int>> &occupancy, double SA_prob, mt19937 &seed){
  int total = count_candidates(wire);
  uniform_real_distribution<double> rand0to1(0.0,1.0); //Random numb from 0.0 to 1.0
  if(rand0to1(seed)<SA_prob){
    uniform_int_distribution<int> pick(0, total - 1); //Pick random possible route
    Wire w = candidate_from_id(wire, pick(seed));
    return {route_add_cost(w,occupancy),w};
  }
  Candidate best{LLONG_MAX,wire};
  for(int cid=0; cid<total; cid++){
    Wire candi = candidate_from_id(wire,cid);
    long long cost = route_add_cost(candi,occupancy);
    if(cost<best.cost) best = {cost,candi};
  }
  return best;
}

int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();

  std::string input_filename;
  int num_threads = 0;
  double SA_prob = 0.1;
  int SA_iters = 5;
  char parallel_mode = '\0';
  int batch_size = 1;

  int opt;
  while ((opt = getopt(argc, argv, "f:n:p:i:m:b:")) != -1) {
    switch (opt) {
    case 'f':
      input_filename = optarg;
      break;
    case 'n':
      num_threads = atoi(optarg);
      break;
    case 'p':
      SA_prob = atof(optarg);
      break;
    case 'i':
      SA_iters = atoi(optarg);
      break;
    case 'm':
      parallel_mode = *optarg;
      break;
    case 'b':
      batch_size = atoi(optarg);
      break;
    default:
      std::cerr << "Usage: " << argv[0]
                << " -f input_filename -n num_threads [-p SA_prob] [-i "
                   "SA_iters] -m parallel_mode -b batch_size\n";
      exit(EXIT_FAILURE);
    }
  }

  // Check if required options are provided
  if (empty(input_filename) || num_threads <= 0 || SA_iters <= 0 ||
      (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
    std::cerr << "Usage: " << argv[0]
              << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] "
                 "-m parallel_mode -b batch_size\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Number of threads: " << num_threads << '\n';
  std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
  std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
  std::cout << "Input file: " << input_filename << '\n';
  std::cout << "Parallel mode: " << parallel_mode << '\n';
  std::cout << "Batch size: " << batch_size << '\n';

  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  int dim_x, dim_y;
  int num_wires;

  /* Read the grid dimension and wire information from file */
  fin >> dim_x >> dim_y >> num_wires;

  std::vector<Wire> wires(num_wires);
  std::vector occupancy(dim_y, std::vector<int>(dim_x));
  std::cout << "Question Spec: dim_x=" << dim_x << ", dim_y=" << dim_y
            << ", number of wires=" << num_wires << '\n';

  // TODO (student code start): Read the wire information from file, 
  // you may need to change this if you define the wire structure differently.
  for (auto &wire : wires) {
    fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
    wire.move_x_start = true;
    wire.move_x_end = false;
    wire.mid_x = wire.end_x;
    wire.mid_y = wire.start_y;
  }

  /* Initialize any additional data structures needed in the algorithm */
  for (const auto &wire : wires) {
    apply_wire(wire, occupancy, 1);
  }


  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  // Student code end
  const double init_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - init_start)
          .count();
  std::cout << "Initialization time (sec): " << std::fixed
            << std::setprecision(10) << init_time << '\n';

  const auto compute_start = std::chrono::steady_clock::now();

  /* TODO (student code start): Implement the wire routing algorithm here and
    feel free to structure the algorithm into different functions.
    Don't use global variables.
    Use OpenMP to parallelize the algorithm.
  */
  
  if (parallel_mode == 'W') {
    // within wires
    Wire wire;
    int total=0;
    Candidate global_best;

    #pragma omp parallel shared(wires, occupancy, wire, total, global_best)
    {
      for(int w = 0; w < (int)wires.size(); w++) {
        #pragma omp single
        {
          wire = wires[w];
          apply_wire(wire, occupancy, -1);
          total = count_candidates(wire);
          global_best = {LLONG_MAX, wire};
        }

        Candidate local_best{LLONG_MAX, wire};
        #pragma omp for schedule(static)
        for (int cid = 0; cid < total; cid++) {
          Wire candi = candidate_from_id(wire, cid);
          long long c = route_add_cost(candi, occupancy);

          if (c < local_best.cost) {
            local_best.cost = c;
            local_best.route = candi;
          }
        }

        #pragma omp critical
        {
          if (local_best.cost < global_best.cost) {
            global_best = local_best;
          }
        }

        #pragma omp barrier
        #pragma omp single
        {
          wires[w] = global_best.route;
          apply_wire(wires[w], occupancy, 1);
        }
      }
    }
    
  } else {
    // across wires
    for(int SA_i=0; SA_i<SA_iters; SA_i++){
      int total = (int)wires.size();
      std::atomic<int> batch_index(0);
      #pragma omp parallel shared(batch_index, wires, occupancy)
      {
        std::mt19937 seed(omp_get_thread_num());
        std::vector<int> batch_ids;
        std::vector<Wire> old_wires;
        std::vector<Wire> new_wires;

        while(true){
          int start = batch_index.fetch_add(batch_size, std::memory_order_relaxed);
          if (start>=total) break;
          int end = std::min(start+batch_size,total);

          batch_ids.clear();
          old_wires.clear();
          new_wires.clear();
          batch_ids.reserve(end - start);
          old_wires.reserve(end - start);
          new_wires.reserve(end - start);

          for(int i=start; i<end; i++){
            Wire old_wire = wires[i];
            Candidate best = find_best_serial(old_wire, occupancy, SA_prob, seed);
            batch_ids.push_back(i);
            old_wires.push_back(old_wire);
            new_wires.push_back(best.route);
          }

          #pragma omp critical
          {
            for (size_t k = 0; k < batch_ids.size(); k++) {
              apply_wire(old_wires[k], occupancy, -1);
              wires[batch_ids[k]] = new_wires[k];
              apply_wire(wires[batch_ids[k]], occupancy, 1);
            }
          }
        }
      }
    }
  }

  // Student code end
  // DON'T CHANGE THE FOLLOWING CODE
  const double compute_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - compute_start)
          .count();
  std::cout << "Computation time (sec): " << compute_time << '\n';

  /* wire to run check on wires and occupancy */
  wr_checker checker(wires, occupancy);
  checker.validate();

  /* Write wires and occupancy matrix to files */
  print_stats(occupancy);
  write_output(wires, num_wires, occupancy, dim_x, dim_y);
}

/* TODO (student): implement to_validate_format to convert Wire to
  validate_wire_t keypoint representation in order to run checker and
  write output
  
*/
struct Point {
    int x;
    int y;
};

// Helper to skip duplicates with the previous point
static void add_point(std::vector<Point>& pts, int x, int y) {
    if (!pts.empty() && pts.back().x == x && pts.back().y == y) return;
    pts.push_back({x, y});
}

validate_wire_t Wire::to_validate_format(void) const {
    validate_wire_t w;

    std::vector<Point> pt_arr;
    pt_arr.reserve(MAX_PTS_PER_WIRE);

    add_point(pt_arr, start_x, start_y);
    if(move_x_start) add_point(pt_arr, mid_x, start_y);
    else add_point(pt_arr, start_x, mid_y);
    add_point(pt_arr, mid_x, mid_y);

    if(move_x_end) add_point(pt_arr, end_x, mid_y);
    else add_point(pt_arr, mid_x, end_y);
    add_point(pt_arr, end_x, end_y);

    for (size_t i = 1; i + 1 < pt_arr.size(); ) {
        bool same_x = (pt_arr[i-1].x == pt_arr[i].x) && (pt_arr[i].x == pt_arr[i+1].x);
        bool same_y = (pt_arr[i-1].y == pt_arr[i].y) && (pt_arr[i].y == pt_arr[i+1].y);
        if (same_x || same_y) pt_arr.erase(pt_arr.begin() + i);
        else ++i;
    }

    w.num_pts = static_cast<uint8_t>(pt_arr.size());
    for (size_t i = 0; i < pt_arr.size(); ++i) {
        w.p[i].x = static_cast<uint16_t>(pt_arr[i].x);
        w.p[i].y = static_cast<uint16_t>(pt_arr[i].y);
    }

    return w;
}
