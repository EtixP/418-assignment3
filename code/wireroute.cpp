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


#include <omp.h>
#include <unistd.h>

struct Candidate{
  long long cost;
  Wire route;    
};

struct Point {
    int x;
    int y;
};

// representation of a given route of a wire path
struct Route {
  uint8_t num_pts;
  Point pts[5];
};

// struct for selecting best candidate in within_wire parallel
struct BestRoute_local {
  long long cost;
  int cid;
};

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

// ! helper to initialize all necessary components before stepping into T = 1
void initialize() {}

static void add_point_compact(Point pts[], int &n, int x, int y) {
  if (n > 0 && pts[n-1].x == x && pts[n-1].y == y) return;
  pts[n++] = {x, y};
}

static int build_wire_points(const Wire &w, Point pts[5]) {
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

static inline int sign_step(int v) {
  return (v > 0) - (v < 0);
}

static inline void apply_segment(int x0, int y0, int x1, int y1,
                                 std::vector<std::vector<int>> &occupancy,
                                 int delta, bool include_endpoint) {
  int x = x0;
  int y = y0;
  int sx = sign_step(x1 - x0);
  int sy = sign_step(y1 - y0);

  // Walk from start to end (excluding end by default).
  while (x != x1 || y != y1) {
    occupancy[y][x] += delta;
    x += sx;
    y += sy;
  }

  // Include final endpoint once for the last segment only.
  if (include_endpoint) {
    occupancy[y][x] += delta;
  }
}

void apply_wire(const Wire &w, std::vector<std::vector<int>> &occupancy, int delta) {
  Point pts[5];
  int n = build_wire_points(w, pts);

  for (int i = 0; i + 1 < n; ++i) {
    bool include_endpoint = (i + 2 == n); // only last segment includes its end
    apply_segment(pts[i].x, pts[i].y,
                  pts[i + 1].x, pts[i + 1].y,
                  occupancy, delta, include_endpoint);
  }
}


// ! helper for greedy search of each wire's best route
void routing(Wire w) {}

static inline int sign_dir(int v) {
  return (v > 0) - (v < 0);
}

// output the current Route 
static inline Route route_from_wire(const Wire &w) {
  Route r{};
  Point pts[5];
  int n = build_wire_points(w, pts);
  r.num_pts = static_cast<uint8_t>(n);
  for (int i = 0; i < n; ++i) r.pts[i] = pts[i];
  return r;
}

static inline void push_candidate_route(std::vector<Route> &out,
                                        const Wire &base,
                                        bool move_x_start,
                                        bool move_x_end,
                                        int mid_x,
                                        int mid_y) {
  Wire c = base;
  c.move_x_start = move_x_start;
  c.move_x_end = move_x_end;
  c.mid_x = mid_x;
  c.mid_y = mid_y;
  out.push_back(route_from_wire(c));
}

// generate a set of all possible routes for a given wires[wire_index]
std::vector<Route> generate_candidates(const std::vector<Wire> &wires, size_t wire_index) {
  std::vector<Route> out;
  if (wire_index >= wires.size()) return out;

  const Wire &base = wires[wire_index];
  const int x0 = base.start_x, y0 = base.start_y;
  const int x1 = base.end_x,   y1 = base.end_y;

  const int dx = std::abs(x1 - x0);
  const int dy = std::abs(y1 - y0);
  const int sx = sign_dir(x1 - x0);
  const int sy = sign_dir(y1 - y0);

  // Straight line: only one legal route.
  if (dx == 0 || dy == 0) {
    push_candidate_route(out, base, true, false, x1, y0);
    return out;
  }

  // total = dx + dy + 2*(dx-1)*(dy-1)
  out.reserve(dx + dy + 2 * (dx - 1) * (dy - 1));

  // 1-bend
  push_candidate_route(out, base, true,  false, x1, y0); // horizontal-first
  push_candidate_route(out, base, false, true,  x0, y1); // vertical-first

  // 2-bend (HF family)
  for (int k = 1; k <= dx - 1; ++k) {
    int xm = x0 + sx * k;
    push_candidate_route(out, base, true, true, xm, y1);
  }

  // 2-bend (VF family)
  for (int l = 1; l <= dy - 1; ++l) {
    int ym = y0 + sy * l;
    push_candidate_route(out, base, false, false, x1, ym);
  }

  // 3-bend: two orientations per interior point
  for (int l = 1; l <= dy - 1; ++l) {
    int ym = y0 + sy * l;
    for (int k = 1; k <= dx - 1; ++k) {
      int xm = x0 + sx * k;
      push_candidate_route(out, base, true,  true,  xm, ym); // HF overall
      push_candidate_route(out, base, false, false, xm, ym); // VF overall
    }
  }

  return out;
}

// convert a given Route object and rewrite the input Wire object
static inline void wire_from_route(Wire &wire, const Route &route) {
  const int first = 0;
  const int last = static_cast<int>(route.num_pts) - 1;

  wire.start_x = route.pts[first].x;
  wire.start_y = route.pts[first].y;
  wire.end_x   = route.pts[last].x;
  wire.end_y   = route.pts[last].y;

  const Point &start_point = route.pts[first];
  const Point &second_point = route.pts[first + 1];
  const Point &before_end_point = route.pts[last - 1];
  const Point &end_point = route.pts[last];

  // True if first segment is horizontal.
  wire.move_x_start = (second_point.y == start_point.y);

  // True if last segment is horizontal.
  wire.move_x_end = (before_end_point.y == end_point.y);

  if (route.num_pts == 2) {
    // Straight line canonical form.
    wire.mid_x = wire.end_x;
    wire.mid_y = wire.start_y;
    wire.move_x_start = true;
    wire.move_x_end = false;
    return;
  }

  if (route.num_pts == 3) {
    wire.mid_x = route.pts[1].x;
    wire.mid_y = route.pts[1].y;
    return;
  }

  if (route.num_pts == 4) {
    wire.mid_x = route.pts[2].x;
    wire.mid_y = route.pts[2].y;
    return;
  }

  // route.num_pts == 5
  wire.mid_x = route.pts[2].x;
  wire.mid_y = route.pts[2].y;
}

static inline long long incremental_add_cost(int occ_value) {
  // (n+1)^2 - n^2
  return 2LL * occ_value + 1LL;
}

// ! function to evaluate the cost of choosing a given candidate route for a wire
long long eval_cost(const std::vector<std::vector<int>> &occupancy,
                    const std::vector<Route> &candidates,
                    size_t candidate_index) {
  assert(candidate_index < candidates.size());
  const Route &route = candidates[candidate_index];

  long long total = 0;

  for (int seg = 0; seg + 1 < route.num_pts; ++seg) {
    int x0 = route.pts[seg].x;
    int y0 = route.pts[seg].y;
    int x1 = route.pts[seg + 1].x;
    int y1 = route.pts[seg + 1].y;

    int sx = sign_dir(x1 - x0);
    int sy = sign_dir(y1 - y0);

    int x = x0, y = y0;
    while (x != x1 || y != y1) {
      total += incremental_add_cost(occupancy[y][x]);
      x += sx;
      y += sy;
    }

    // Include final endpoint only for last segment.
    if (seg + 2 == route.num_pts) {
      total += incremental_add_cost(occupancy[y][x]);
    }
  }

  return total;
}

// End of Helper functions
// ========================================================================//

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

  // 2D array for all wires possible candidate routes
  std::vector<std::vector<Route>> wire_routes(num_wires);
  wire_routes.reserve(num_wires);

  // TODO (student code start): Read the wire information from file,
  // you may need to change this if you define the wire structure differently.

   // ! 1. Randomize the initial routing of every wire by setting them to horizontal L shape first (i.e. go right to mid and then go down to end)
  for (auto &wire : wires) {
    fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
    wire.move_x_start = true;
    wire.move_x_end = false;
    wire.mid_x = wire.end_x;
    wire.mid_y = wire.start_y;
  }

  /* Initialize any additional data structures needed in the algorithm */

  /*
    TODO (student code start): Implement the wire routing algorithm here and
    feel free to structure the algorithm into different functions.
    Don't use global variables.
    Use OpenMP to parallelize the algorithm.
  */
  std::mt19937 rng(std::random_device{}()); // seeding for a random number for probablistic on SA_prob

  for(Wire &wire: wires){
    // ! 2. Place the initialized wire routing cost into occupancy matrix
    apply_wire(wire, occupancy, 1);
  }

  // Student code end
  const double init_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - init_start)
          .count();
  std::cout << "Initialization time (sec): " << std::fixed
            << std::setprecision(10) << init_time << '\n';

  const auto compute_start = std::chrono::steady_clock::now();
  
  // generate all route candidates for all wires
  for (size_t w = 0; w < (size_t) num_wires; ++w) {
    wire_routes[w] = generate_candidates(wires, w);
  }


  if (parallel_mode == 'W') {
    // within wires

    // thread-wise shared data
    std::vector<long long> thread_best_cost(num_threads, LLONG_MAX);
    std::vector<size_t> thread_best_idx(num_threads, 0);

    bool skip_wire = false;
    bool do_random = false;
    size_t random_index = 0;

    #pragma omp parallel num_threads(num_threads) shared(wires, occupancy, wire_routes, rng, SA_prob, SA_iters, num_wires, thread_best_cost, thread_best_idx, skip_wire, do_random, random_index)
    {
      int tid = omp_get_thread_num();
      for (size_t iter = 0; iter < (size_t) SA_iters; ++iter) {
        for (size_t wire_index = 0; wire_index < (size_t) num_wires; ++wire_index) {
          // serial decisions/state mutation for this wire
          #pragma omp single
          {
            skip_wire = false;
            do_random = false;
            random_index = 0;

              // If there is only one candidate route from this wire, we skip this
            if (wire_routes[wire_index].size() == 1){
              skip_wire = true;
            }
            // ! if P is hit choose a route randomly from the set of all candidates
            else if (std::bernoulli_distribution(SA_prob)(rng)) {
              do_random = true;
              random_index =
                std::uniform_int_distribution<size_t>(0, wire_routes[wire_index].size() - 1)(rng);
              apply_wire(wires[wire_index], occupancy, -1);
              wire_from_route(wires[wire_index], wire_routes[wire_index][random_index]);
              apply_wire(wires[wire_index], occupancy, +1);
            }
            else{
              // ! non-random path: remove old route once before candidate eval
              apply_wire(wires[wire_index], occupancy, -1);
            }
          }

          if (skip_wire || do_random) {
            continue;
          }

          // local best for current thread
          long long local_best{LLONG_MAX};
          size_t local_best_index = 0;

          // parallelize the next for loop via static assignment
          #pragma omp for schedule(static)
          for (size_t can_index = 0; can_index < wire_routes[wire_index].size(); ++can_index){
            long long route_cost = eval_cost(occupancy, wire_routes[wire_index], can_index);
            
            // update thread best cost and route if found cheaper
            if(route_cost < local_best){
              local_best = route_cost;
              local_best_index = can_index;
            }
          }

          // publish per-thread result
          thread_best_cost[tid] = local_best;
          thread_best_idx[tid] = local_best_index;
          
          //#pragma omp barrier
          
          // serial merge + commit chosen route
          #pragma omp single
          {
            long long global_best = LLONG_MAX;
            size_t best_index = 0;

            for (int t = 0; t < num_threads; ++t) {
              if (thread_best_cost[t] < global_best) {
                global_best = thread_best_cost[t];
                best_index = thread_best_idx[t];
              }
            }
            // ! update wire formation and occupancy matrix
            wire_from_route(wires[wire_index], wire_routes[wire_index][best_index]);
            apply_wire(wires[wire_index], occupancy, +1);
          }
          //#pragma omp barrier
        }
      }
    }
  }
  else {
    // across wires
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

/*
  TODO (student): implement to_validate_format to convert Wire to
  validate_wire_t keypoint representation in order to run checker and
  write output
*/

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