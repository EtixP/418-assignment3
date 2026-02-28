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

// ======================================================================= //
// Start of Helper functions for within-wire
// ======================================================================= //

struct Point {
    int x;
    int y;
};

// representation of a given route of a wire path
struct Route {
  struct Seg {
    int x0, y0, x1, y1;
    int sx, sy;
  };  // segments are stored individually here such that we can quickly compute eval_cost_bounded function
  uint8_t num_pts;
  Point pts[5];
  Wire wire_repr;
  uint8_t num_segs;
  Seg segs[4];
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

// append (x,y) to pts[], skip if identical to last point in pts[]
static void add_point_compact(Point pts[], int &n, int x, int y) {
  if (n > 0 && pts[n-1].x == x && pts[n-1].y == y) return;
  pts[n++] = {x, y};
}

// Fill pts on the Points which Wire w touches and return the number of pts element filled
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

// Determine direction of the segment given two endpoint
static inline int sign_dir(int v) {
  return (v > 0) - (v < 0);
}

// Using two endpoints of a segment, apply delta change to all occupancy entries which is covered by this segment
static inline void apply_segment(int x0, int y0, int x1, int y1,
                                 std::vector<std::vector<int>> &occupancy,
                                 int delta, bool include_endpoint) {
  int x = x0;
  int y = y0;
  int sx = sign_dir(x1 - x0);
  int sy = sign_dir(y1 - y0);

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

// Given a Wire object, add / remove the wire from the occupancy matrix based on delta
void apply_wire(const Wire &w, std::vector<std::vector<int>> &occupancy, int delta) {
  Point pts[5];
  int n = build_wire_points(w, pts);

  for (int i = 0; i + 1 < n; ++i) {
    bool include_endpoint = (i + 2 == n); // only last segment includes its end

    // 
    apply_segment(pts[i].x, pts[i].y,
                  pts[i + 1].x, pts[i + 1].y,
                  occupancy, delta, include_endpoint);
  }
}


// output a Route object absed on current Wire object data
static inline Route route_from_wire(const Wire &w) {
  Route r{};
  Point pts[5];
  int n = build_wire_points(w, pts);
  r.num_pts = static_cast<uint8_t>(n);
  for (int i = 0; i < n; ++i) r.pts[i] = pts[i];
  r.wire_repr = w;
  r.num_segs = static_cast<uint8_t>(n > 0 ? n - 1 : 0);
  for (int i = 0; i + 1 < n; ++i) {
    r.segs[i].x0 = r.pts[i].x;
    r.segs[i].y0 = r.pts[i].y;
    r.segs[i].x1 = r.pts[i + 1].x;
    r.segs[i].y1 = r.pts[i + 1].y;
    r.segs[i].sx = sign_dir(r.segs[i].x1 - r.segs[i].x0);
    r.segs[i].sy = sign_dir(r.segs[i].y1 - r.segs[i].y0);
  }
  return r;
}

// Return num of potential candidates using a given Wire object via formula (dx + dy + 2 * (dx - 1) * (dy - 1))
static inline size_t count_candidates_for_wire(const Wire &base) {
  const int dx = std::abs(base.end_x - base.start_x);
  const int dy = std::abs(base.end_y - base.start_y);
  if (dx == 0 || dy == 0) return 1;
  return static_cast<size_t>(dx + dy + 2 * (dx - 1) * (dy - 1));
}

// Using cid, decode its underlying Route for a given wire
static inline Route decode_candidate_from_cid(const Wire &base, size_t cid) {
  const int x0 = base.start_x;
  const int y0 = base.start_y;
  const int x1 = base.end_x;
  const int y1 = base.end_y;

  const int dx = std::abs(x1 - x0);
  const int dy = std::abs(y1 - y0);
  const int sx = sign_dir(x1 - x0);
  const int sy = sign_dir(y1 - y0);
  const size_t total_candidates = count_candidates_for_wire(base);
  assert(cid < total_candidates);

  Wire c = base;

  // Straight line has only one legal candidate route.
  if (dx == 0 || dy == 0) {
    c.move_x_start = true;
    c.move_x_end = false;
    c.mid_x = x1;
    c.mid_y = y0;
    return route_from_wire(c);
  }

  // 1-bend horizontal-first
  if (cid == 0) {
    c.move_x_start = true;
    c.move_x_end = false;
    c.mid_x = x1;
    c.mid_y = y0;
    return route_from_wire(c);
  }

  // 1-bend vertical-first
  if (cid == 1) {
    c.move_x_start = false;
    c.move_x_end = true;
    c.mid_x = x0;
    c.mid_y = y1;
    return route_from_wire(c);
  }

  size_t rem = cid - 2;

  // 2-bend HF family: k in [1, dx-1]
  if (rem < static_cast<size_t>(dx - 1)) {
    const int k = static_cast<int>(rem) + 1;
    c.move_x_start = true;
    c.move_x_end = true;
    c.mid_x = x0 + sx * k;
    c.mid_y = y1;
    return route_from_wire(c);
  }
  rem -= static_cast<size_t>(dx - 1);

  // 2-bend VF family: l in [1, dy-1]
  if (rem < static_cast<size_t>(dy - 1)) {
    const int l = static_cast<int>(rem) + 1;
    c.move_x_start = false;
    c.move_x_end = false;
    c.mid_x = x1;
    c.mid_y = y0 + sy * l;
    return route_from_wire(c);
  }
  rem -= static_cast<size_t>(dy - 1);

  // 3-bend family: for each (l, k), two candidates: HF then VF.
  const int k_count = dx - 1;
  const size_t pair_index = rem / 2;
  const int orient = static_cast<int>(rem % 2); // 0 => HF, 1 => VF
  const int l = static_cast<int>(pair_index / static_cast<size_t>(k_count)) + 1;
  const int k = static_cast<int>(pair_index % static_cast<size_t>(k_count)) + 1;

  c.mid_x = x0 + sx * k;
  c.mid_y = y0 + sy * l;
  if (orient == 0) {
    c.move_x_start = true;
    c.move_x_end = true;
  } else {
    c.move_x_start = false;
    c.move_x_end = false;
  }

  return route_from_wire(c);
}

// function to evaluate the cost of choosing a given candidate Route for a wire
// Main optimization target for Version 3, where we added:
// We realize the assumption that: all Routes decoded via CID will get their Segment representation
// stored inside route_from_wire call (from decode function). This step happens before eval function,
// hence applicable for us to directly evaluate the cost of a Route using its Seg without reconstructing
long long eval_cost_bounded(const std::vector<std::vector<int>> &occupancy,
                            const Route &route,
                            long long bound) {
  long long total = 0;

  for (int seg = 0; seg < route.num_segs; ++seg) {
    const Route::Seg &s = route.segs[seg];
    int x = s.x0;
    int y = s.y0;

    if (s.sx != 0) {
      const int *row = occupancy[y].data();
      while (x != s.x1) {
        total += 2LL * row[x] + 1LL;
        x += s.sx;
      }
    } else {
      while (y != s.y1) {
        total += 2LL * occupancy[y][x] + 1LL;
        y += s.sy;
      }
    }

    if (seg + 1 == route.num_segs) {
      total += 2LL * occupancy[y][x] + 1LL;
    }
  }

  return total;
}

// ======================================================================= //
// End of Helper functions for within-wire
// ======================================================================= //


// ======================================================================= //
// Start of Helper functions for across-wire
// ======================================================================= //

struct BestRoute {
  long long cost;
  Route route;
};

struct Candidate{
  long long cost;
  Wire route;    
};

Wire candidate_from_id(const Wire &base, int cid){
  int x0 = base.start_x, y0 = base.start_y;
  int x1 = base.end_x, y1 = base.end_y;
  int dx = std::abs(x1-x0), dy = std::abs(y1-y0);
  int sx = sign_dir(x1-x0), sy = sign_dir(y1-y0);

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

static void add_segment_cost(const std::vector<std::vector<int>> &occupancy,
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

static bool add_segment_cost_bounded(const std::vector<std::vector<int>> &occupancy,
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
                         const std::vector<std::vector<int>> &occupancy) {
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
                                 const std::vector<std::vector<int>> &occupancy,
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

// If random SA branch triggers: picks a random candidate route.
// Else: enumerates all candidate cids, decodes each to Route, computes cost with eval_cost_bounded, keeps the minimum.
// Output: {best_cost, best_route} for that wire.
// It is “serial” because one thread evaluates that wire's candidates sequentially (used in across mode).
Candidate find_best_serial(const Wire &wire, const std::vector<std::vector<int>> &occupancy, double SA_prob, std::mt19937 &seed){
  int total = count_candidates_for_wire(wire);
  std::uniform_real_distribution<double> rand0to1(0.0,1.0); //Random numb from 0.0 to 1.0
  if(rand0to1(seed)<SA_prob){
    std::uniform_int_distribution<int> pick(0, total - 1); //Pick random possible route
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

// Builds that wire’s keypoints, iterates each segment, computes tile-range touched by segment bbox.
// Pushes all tile IDs touched by the wire into tile_ids.
// Caller then sorts/uniques this list and locks those tiles before occupancy update.
static void collect_wire_tiles(const Wire &w, int tile_w, int tile_h, int tiles_x,
                               std::vector<int> &tile_ids) {
  Point pts[5];
  int n = build_wire_points(w, pts);

  for (int i = 0; i + 1 < n; i++) {
    const int x = pts[i].x;
    const int y = pts[i].y;
    const int xn = pts[i + 1].x;
    const int yn = pts[i + 1].y;

    // Segment is axis-aligned: touched tiles are the tile-range of its bbox.
    const int min_x = std::min(x, xn);
    const int max_x = std::max(x, xn);
    const int min_y = std::min(y, yn);
    const int max_y = std::max(y, yn);

    const int tx0 = min_x / tile_w;
    const int tx1 = max_x / tile_w;
    const int ty0 = min_y / tile_h;
    const int ty1 = max_y / tile_h;

    for (int ty = ty0; ty <= ty1; ty++) {
      for (int tx = tx0; tx <= tx1; tx++) {
        tile_ids.push_back(ty * tiles_x + tx);
      }
    }
  }
}


// ======================================================================= //
// End of Helper functions for across-wire
// ======================================================================= //

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

  // Read the wire information from file (unchanged),
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

  std::mt19937 rng(std::random_device{}()); // seeding for a random number for probablistic on SA_prob

  for(const Wire &wire: wires){
    // ! 2. Place the initialized wire routing cost into occupancy matrix, notice this is the default intialization wires same as original code
    // A very simple logic, according to Piazza post can be placed inside init_time
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

    /* Implement the wire routing algorithm here and
    feel free to structure the algorithm into different functions.
    Don't use global variables.
    Use OpenMP to parallelize the algorithm.
  */


  if (parallel_mode == 'W') {
    // within wires

    // thread-wise shared data
    std::vector<long long> thread_best_cost(num_threads, LLONG_MAX);
    std::vector<size_t> thread_best_idx(num_threads, 0);
    // double t_random_update = 0.0;
    // double t_candidate_eval = 0.0;
    // double t_commit_merge_apply = 0.0;
    // double eval_phase_start = 0.0;

    bool skip_wire = false;
    bool do_random = false;
    size_t random_index = 0;
    size_t current_route_count = 0;

    #pragma omp parallel num_threads(num_threads) shared(wires, occupancy, rng, SA_prob, SA_iters, num_wires, thread_best_cost, thread_best_idx, skip_wire, do_random, random_index, current_route_count)
    {
      int tid = omp_get_thread_num();

      for (size_t iter = 0; iter < static_cast<size_t> (SA_iters); ++iter) {
        for (size_t wire_index = 0; wire_index < static_cast<size_t> (num_wires); ++wire_index) {
          // serial decisions/state mutation for this wire
          #pragma omp single
          {
            skip_wire = false;
            do_random = false;
            random_index = 0;
            current_route_count = count_candidates_for_wire(wires[wire_index]);

            // If there is only one candidate route from this wire, we skip the rest of the steps
            if (current_route_count == 1){
              skip_wire = true;
            }
            // if P is hit: choose a route randomly from the set of all candidates
            else if (std::bernoulli_distribution(SA_prob)(rng)) {
              // const double random_start = omp_get_wtime();
              do_random = true;
              random_index = std::uniform_int_distribution<size_t>(0, current_route_count - 1)(rng);

              apply_wire(wires[wire_index], occupancy, -1);
              const Route random_route = decode_candidate_from_cid(wires[wire_index], random_index);
              wires[wire_index] = random_route.wire_repr;
              apply_wire(wires[wire_index], occupancy, +1);
              // t_random_update += (omp_get_wtime() - random_start);
            }
            else{
              // ! non-random path: remove old route once before candidate eval
              apply_wire(wires[wire_index], occupancy, -1);
            }
          }

          bool skip_local = skip_wire;
          bool random_local = do_random;

          #pragma omp barrier // ensures all threads captured same decision
          
          if (skip_local || random_local) {
            continue;
          }

          // local best for current thread
          long long local_best{LLONG_MAX};
          size_t local_best_index = 0;

          // #pragma omp single
          // {
          //   eval_phase_start = omp_get_wtime();
          // }

          // parallelize the next for loop via static assignment
          #pragma omp for schedule(static)
          for (size_t can_index = 0; can_index < current_route_count; ++can_index){
            const Route candidate_route = decode_candidate_from_cid(wires[wire_index], can_index);
            long long route_cost = eval_cost_bounded(occupancy, candidate_route, local_best);
            
            // update thread best cost and route if found cheaper
            if(route_cost < local_best){
              local_best = route_cost;
              local_best_index = can_index;
            }
          }

          // #pragma omp single
          // {
          //   t_candidate_eval += (omp_get_wtime() - eval_phase_start);
          // }

          // publish per-thread result
          thread_best_cost[tid] = local_best;
          thread_best_idx[tid] = local_best_index;

          // Ensure all thread_best_* writes are visible before merge.
          #pragma omp barrier
          
          // serial merge + commit chosen route
          #pragma omp single
          {
            // const double commit_start = omp_get_wtime();
            long long global_best = LLONG_MAX;
            size_t best_index = 0;

            for (int t = 0; t < num_threads; ++t) {
              if (thread_best_cost[t] < global_best) {
                global_best = thread_best_cost[t];
                best_index = thread_best_idx[t];
              }
            }
            // update wire formation and occupancy matrix
            const Route best_route = decode_candidate_from_cid(wires[wire_index], best_index);
            wires[wire_index] = best_route.wire_repr;
            apply_wire(wires[wire_index], occupancy, +1);
          }
        }
      }
    }

    // std::cout << "Profile random_update (sec): " << t_random_update << '\n';
    // std::cout << "Profile candidate_eval (sec): " << t_candidate_eval << '\n';
  }
  else {
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    
    // across wires
    const int tile_w = 8; //tile width
    const int tile_h = 8; //tile height
    const int tiles_x = (dim_x + tile_w - 1) / tile_w; //# of tiles in a row
    const int tiles_y = (dim_y + tile_h - 1) / tile_h; //# of tiles in a col
    const int num_tile_locks = tiles_x * tiles_y;

    std::vector<omp_lock_t> tile_locks(num_tile_locks);
    for (int i = 0; i < num_tile_locks; i++) {
      omp_init_lock(&tile_locks[i]);
    }

    for(int SA_i=0; SA_i<SA_iters; SA_i++){
      int total = (int)wires.size();
      std::atomic<int> batch_index(0);
      #pragma omp parallel shared(batch_index, wires, occupancy)
      {
        std::mt19937 seed(omp_get_thread_num());
        std::vector<int> batch_ids;
        std::vector<Wire> old_wires;
        std::vector<Wire> new_wires;
        std::vector<int> touched_tiles;

        while(true){
          int start = batch_index.fetch_add(batch_size, std::memory_order_relaxed);
          if (start>=total) break;
          int end = std::min(start+batch_size,total);

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

/*
  implemented to_validate_format to convert Wire to
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
