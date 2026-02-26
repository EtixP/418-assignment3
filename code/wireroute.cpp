/**
 * Parallel VLSI Wire Routing via OpenMP
 * Jongsun Park (jongsunp), Dylan Sun (bdsun)
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

struct IntPoint {
  int x;
  int y;
};

static void add_point_compact(IntPoint pts[], int &n, int x, int y) {
  if (n > 0 && pts[n-1].x == x && pts[n-1].y == y) return;
  pts[n++] = {x, y};
}

static int build_wire_points(const Wire &w, IntPoint pts[5]) {
  int n = 0;
  add_point_compact(pts, n, w.start_x, w.start_y);
  if (w.move_x_start) add_point_compact(pts, n, w.mid_x, w.start_y);
  else add_point_compact(pts, n, w.start_x, w.mid_y);
  add_point_compact(pts, n, w.mid_x, w.mid_y);
  if (w.move_x_end) add_point_compact(pts, n, w.end_x, w.mid_y);
  else add_point_compact(pts, n, w.mid_x, w.end_y);
  add_point_compact(pts, n, w.end_x, w.end_y);

  // Remove overlaping points
  int i = 1;
  while (i + 1 < n) {
    bool same_x = (pts[i - 1].x == pts[i].x) && (pts[i].x == pts[i + 1].x);
    bool same_y = (pts[i - 1].y == pts[i].y) && (pts[i].y == pts[i + 1].y);
    if (same_x || same_y) {
      for (int j = i; j + 1 < n; j++) pts[j] = pts[j + 1];
      n--;
    } else {
      i++;
    }
  }
  return n;
}

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
  IntPoint pts[5];
  int n = build_wire_points(w, pts);

  for (int i = 0; i < n - 1; i++) {
    int x = pts[i].x;
    int y = pts[i].y;
    int xn = pts[i + 1].x;
    int yn = pts[i + 1].y;
    int sx = (xn > x) ? 1 : (xn < x ? -1 : 0);
    int sy = (yn > y) ? 1 : (yn < y ? -1 : 0);

    while (x != xn || y != yn) {
      occupancy[y][x] += change;
      x += sx;
      y += sy;
    }
    if (i == n - 2) occupancy[y][x] += change;
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

static long long incr_cost(int n) {
  return 2LL*n + 1;  // (n+1)^2 - n^2
}

static void add_segment_cost(const vector<vector<int>> &occupancy,
                             int x0, int y0, int x1, int y1,
                             long long &tot) {
  if (x0 == x1 && y0 == y1) return;

  if (y0 == y1) {
    int x = x0;
    int sx = (x1 > x0) ? 1 : -1;
    while (x != x1) {
      int n = occupancy[y0][x];
      tot += 2LL*n + 1;
      x += sx;
    }
  } else {
    int y = y0;
    int sy = (y1 > y0) ? 1 : -1;
    while (y != y1) {
      int n = occupancy[y][x0];
      tot += 2LL*n + 1;
      y += sy;
    }
  }
}

static bool add_segment_cost_bounded(const vector<vector<int>> &occupancy,
                                     int x0, int y0, int x1, int y1,
                                     long long &tot, long long bound) {
  if (x0 == x1 && y0 == y1) return false;

  if (y0 == y1) {
    int x = x0;
    int sx = (x1 > x0) ? 1 : -1;
    while (x != x1) {
      int n = occupancy[y0][x];
      tot += 2LL*n + 1;
      if (tot >= bound) return true;
      x += sx;
    }
  } else {
    int y = y0;
    int sy = (y1 > y0) ? 1 : -1;
    while (y != y1) {
      int n = occupancy[y][x0];
      tot += 2LL*n + 1;
      if (tot >= bound) return true;
      y += sy;
    }
  }

  return false;
}

long long route_add_cost(const Wire &candi,
                         const vector<vector<int>> &occupancy) {
  long long tot = 0;

  int p0x = candi.start_x, p0y = candi.start_y;
  int p1x = candi.move_x_start ? candi.mid_x : candi.start_x;
  int p1y = candi.move_x_start ? candi.start_y : candi.mid_y;
  int p2x = candi.mid_x, p2y = candi.mid_y;
  int p3x = candi.move_x_end ? candi.end_x : candi.mid_x;
  int p3y = candi.move_x_end ? candi.mid_y : candi.end_y;
  int p4x = candi.end_x, p4y = candi.end_y;

  add_segment_cost(occupancy, p0x, p0y, p1x, p1y, tot);
  add_segment_cost(occupancy, p1x, p1y, p2x, p2y, tot);
  add_segment_cost(occupancy, p2x, p2y, p3x, p3y, tot);
  add_segment_cost(occupancy, p3x, p3y, p4x, p4y, tot);

  tot += incr_cost(occupancy[p4y][p4x]);
  return tot;
}

long long route_add_cost_bounded(const Wire &candi,
                                 const vector<vector<int>> &occupancy,
                                 long long bound) {
  long long tot = 0;

  int p0x = candi.start_x, p0y = candi.start_y;
  int p1x = candi.move_x_start ? candi.mid_x : candi.start_x;
  int p1y = candi.move_x_start ? candi.start_y : candi.mid_y;
  int p2x = candi.mid_x, p2y = candi.mid_y;
  int p3x = candi.move_x_end ? candi.end_x : candi.mid_x;
  int p3y = candi.move_x_end ? candi.mid_y : candi.end_y;
  int p4x = candi.end_x, p4y = candi.end_y;

  if (add_segment_cost_bounded(occupancy, p0x, p0y, p1x, p1y, tot, bound)) return tot;
  if (add_segment_cost_bounded(occupancy, p1x, p1y, p2x, p2y, tot, bound)) return tot;
  if (add_segment_cost_bounded(occupancy, p2x, p2y, p3x, p3y, tot, bound)) return tot;
  if (add_segment_cost_bounded(occupancy, p3x, p3y, p4x, p4y, tot, bound)) return tot;

  tot += incr_cost(occupancy[p4y][p4x]);
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

  long long current_cost = route_add_cost(wire, occupancy);
  Candidate best{current_cost,wire};

  int probe_n = (total < 24) ? total : 24;
  for (int p = 0; p < probe_n; p++) {
    int cid = (probe_n == 1) ? 0 : (p * (total - 1)) / (probe_n - 1);
    Wire candi = candidate_from_id(wire, cid);
    long long cost = route_add_cost_bounded(candi, occupancy, best.cost);
    if(cost<best.cost) best = {cost,candi};
  }

  for(int cid=0; cid<total; cid++){
    Wire candi = candidate_from_id(wire,cid);
    long long cost = route_add_cost_bounded(candi, occupancy, best.cost);
    if(cost<best.cost) best = {cost,candi};
  }
  return best;
}


static void collect_wire_tiles(const Wire &w, int tile_w, int tile_h, int tiles_x,
                               vector<int>&tile_ids){
  IntPoint pts[5];
  int n = build_wire_points(w, pts);

  for (int i = 0; i < n - 1; i++) {
    int x = pts[i].x;
    int y = pts[i].y;
    int xn = pts[i + 1].x;
    int yn = pts[i + 1].y;

    // Each segment is axis-aligned collect touched tiles by tile range
    int min_x = (x < xn) ? x : xn;
    int max_x = (x > xn) ? x : xn;
    int min_y = (y < yn) ? y : yn;
    int max_y = (y > yn) ? y : yn;

    int tx0 = min_x / tile_w;
    int tx1 = max_x / tile_w;
    int ty0 = min_y / tile_h;
    int ty1 = max_y / tile_h;

    for (int ty = ty0; ty <= ty1; ty++) {
      for (int tx = tx0; tx <= tx1; tx++) {
        tile_ids.push_back(ty * tiles_x + tx);
      }
    }
  }
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
    for (int SA_i = 0; SA_i < SA_iters; SA_i++) {
      Wire wire;
      int total=0;
      Candidate global_best;
      bool choose_random = false;
      long long current_cost = 0;
      mt19937 sa_seed(SA_i);
      atomic<long long> best_bound(LLONG_MAX);

      #pragma omp parallel shared(wires, occupancy, wire, total, global_best, choose_random, current_cost, sa_seed, best_bound)
      {
        for(int w = 0; w < (int)wires.size(); w++) {
          #pragma omp single
          {
            wire = wires[w];
            apply_wire(wire, occupancy, -1);
            total = count_candidates(wire);

            uniform_real_distribution<double> rand0to1(0.0, 1.0);
            choose_random = (rand0to1(sa_seed) < SA_prob);
            if (choose_random) {
              uniform_int_distribution<int> pick(0, total - 1);
              Wire random_route = candidate_from_id(wire, pick(sa_seed));
              global_best = {route_add_cost(random_route, occupancy), random_route};
            } else {
              current_cost = route_add_cost(wire, occupancy);
              global_best = {current_cost, wire};

              int probe_n = (total < 24) ? total : 24;
              for (int p = 0; p < probe_n; p++) {
                int cid = (probe_n == 1) ? 0 : (p * (total - 1)) / (probe_n - 1);
                Wire probe = candidate_from_id(wire, cid);
                long long probe_cost = route_add_cost_bounded(probe, occupancy, global_best.cost);
                if (probe_cost < global_best.cost) {
                  global_best = {probe_cost, probe};
                }
              }

              current_cost = global_best.cost;
              best_bound.store(current_cost, memory_order_relaxed);
            }
          }

          Candidate local_best{current_cost, wire};
          if (!choose_random) {
            #pragma omp for schedule(static)
            for (int cid = 0; cid < total; cid++) {
              Wire candi = candidate_from_id(wire, cid);

              long long bound = local_best.cost;
              long long gbound = best_bound.load(memory_order_relaxed);
              if (gbound < bound) bound = gbound;

              long long c = route_add_cost_bounded(candi, occupancy, bound);

              if (c < local_best.cost) {
                local_best.cost = c;
                local_best.route = candi;

                long long old_bound = best_bound.load(memory_order_relaxed);
                while (c < old_bound &&
                       !best_bound.compare_exchange_weak(old_bound, c, memory_order_relaxed)) {
                }
              }
            }
          }

          if (!choose_random) {
            #pragma omp critical
            {
              if (local_best.cost < global_best.cost) {
                global_best = local_best;
              }
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
    }

  } else {
    // across wires
    const int tile_w = 8; //tile width
    const int tile_h = 8; //tile height
    const int tiles_x = (dim_x + tile_w - 1) / tile_w; //# of tiles in a row
    const int tiles_y = (dim_y + tile_h - 1) / tile_h; //# of tiles in a col
    const int num_tile_locks = tiles_x * tiles_y;

    vector<omp_lock_t> tile_locks(num_tile_locks);
    for (int i = 0; i < num_tile_locks; i++) {
      omp_init_lock(&tile_locks[i]);
    }

    for(int SA_i=0; SA_i<SA_iters; SA_i++){
      int total = (int)wires.size();
      std::atomic<int> batch_index(0);
      #pragma omp parallel shared(batch_index, wires, occupancy)
      {
        mt19937 seed(omp_get_thread_num());
        vector<int> batch_ids;
        vector<Wire> old_wires;
        vector<Wire> new_wires;
        vector<int> touched_tiles;

        while(true){
          int start = batch_index.fetch_add(batch_size, memory_order_relaxed);
          if (start>=total) break;
          int end = min(start+batch_size,total);

          batch_ids.clear(); //No need to realloc
          old_wires.clear();
          new_wires.clear();
          batch_ids.reserve(end - start);
          old_wires.reserve(end - start); //Store old routes
          new_wires.reserve(end - start); //Store new routes

          for(int i=start; i<end; i++){
            Wire old_wire = wires[i];
            Candidate best = find_best_serial(old_wire, occupancy, SA_prob, seed);
            batch_ids.push_back(i);
            old_wires.push_back(old_wire);
            new_wires.push_back(best.route);
          }

          for(size_t k = 0; k<batch_ids.size(); k++){
            touched_tiles.clear();
            collect_wire_tiles(old_wires[k], tile_w, tile_h, tiles_x, touched_tiles);
            collect_wire_tiles(new_wires[k], tile_w, tile_h, tiles_x, touched_tiles);
            sort(touched_tiles.begin(), touched_tiles.end());
            touched_tiles.erase(unique(touched_tiles.begin(), touched_tiles.end()),
                                touched_tiles.end()); // Erase Duplicates

            //Lock all tiles that are touched by old/new routes
            for (int tile_id : touched_tiles) { 
              omp_set_lock(&tile_locks[tile_id]);
            }

            apply_wire(old_wires[k], occupancy, -1);
            wires[batch_ids[k]] = new_wires[k];
            apply_wire(wires[batch_ids[k]], occupancy, 1);

            for (int ti = (int)touched_tiles.size() - 1; ti >= 0; ti--) {
              omp_unset_lock(&tile_locks[touched_tiles[ti]]);
            }
          }
        }
      }
    }

    for (int i = 0; i < num_tile_locks; i++) {
      omp_destroy_lock(&tile_locks[i]);
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
