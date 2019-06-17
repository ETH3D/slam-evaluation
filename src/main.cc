// Copyright 2019 ETH Zürich, Thomas Schöps
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;


// Globals
bool is_little_endian;


// Return codes of the program:
// 0: Success.
// 1: System failure (e.g., due to an internal error).
// 2: Bad user input (e.g., input file does not follow the expected format; "input" excludes the program arguments and the ground truth).
enum class ReturnCodes {
  Success = 0,
  SystemFailure = 1,
  BadInput = 2
};


enum class Mode {
  SE3 = 0,
  Sim3 = 1
};


struct TimestampedPose {
  inline TimestampedPose(
      double timestamp,
      Quaterniond rotation,
      Vector3d translation)
      : timestamp(timestamp),
        rotation(rotation),
        translation(translation) {}
  
  double timestamp;
  Quaterniond rotation;
  Vector3d translation;
};


struct Pose {
  inline Pose() {}
  
  inline Pose(
      Quaterniond rotation,
      Vector3d translation)
      : rotation(rotation),
        translation(translation) {}
  
  Quaterniond rotation;
  Vector3d translation;
  
  Pose inverse() const {
    // y = R * x + t
    // =>  x = R^(-1) * y - R^(-1) * t
    
    Quaterniond inv_rotation = rotation.inverse();
    return Pose(
        inv_rotation,
        -1 * (inv_rotation * translation));
  }
  
  Pose operator* (const Pose& other) const {
    // y = R1 * (R2 * x + t2) + t1
    // =>  y = (R1 * R2) * x + (R1 * t2 + t1)
    
    return Pose(
        rotation * other.rotation,
        rotation * other.translation + translation);
  }
};


bool IsLittleEndian() {
  constexpr int32_t num = 1;
  return (*reinterpret_cast<const int8_t*>(&num) == 1);
}


template <typename T>
inline T SwapEndianness(T val) {
  T result;
  char* ptr = reinterpret_cast<char*>(&val);
  char* result_ptr = reinterpret_cast<char*>(&result);
  constexpr int size = sizeof(T);
  for (int i = 0; i < size; ++ i) {
    result_ptr[size - 1 - i] = ptr[i];
  }
  return result;
}


float FloatToLittleEndian(float value) {
  if (is_little_endian) {
    return value;
  } else {
    return SwapEndianness(value);
  }
}


bool ReadTrajectory(
    const string& path,
    vector<TimestampedPose>* poses_global_tr_frame) {
  ifstream trajectory_file(path);
  if (!trajectory_file) {
    cout << "Could not open trajectory file: " << path << endl;
    return false;
  }
  
  string line;
  getline(trajectory_file, line);
  while (! line.empty()) {
    char time_string[128];
    Vector3d translation;
    Quaterniond rotation;
    
    if (line[0] == '#') {
      getline(trajectory_file, line);
      continue;
    }
    if (sscanf(line.c_str(), "%s %lf %lf %lf %lf %lf %lf %lf",
        time_string,
        &translation[0],
        &translation[1],
        &translation[2],
        &rotation.x(),
        &rotation.y(),
        &rotation.z(),
        &rotation.w()) != 8) {
      cout << "Cannot read poses from file: " << path << endl << "Line:" << endl << line << endl;
      return false;
    }
    
    poses_global_tr_frame->push_back(
        TimestampedPose(atof(time_string),
                        rotation,
                        translation));
    
    getline(trajectory_file, line);
  }
  
  return true;
}


bool ReadImageListFile(
    const string& path,
    vector<double>* image_timestamps) {
  ifstream list_file(path);
  if (!list_file) {
    cout << "Could not open image list file: " << path << endl;
    return false;
  }
  
  string line;
  getline(list_file, line);
  while (! line.empty()) {
    char time_string[128];
    
    if (line[0] == '#') {
      getline(list_file, line);
      continue;
    }
    if (sscanf(line.c_str(), "%s",
        time_string) != 1) {
      cout << "Cannot read line in file: " << path << endl << "Line:" << endl << line << endl;
      return false;
    }
    
    image_timestamps->push_back(atof(time_string));
    
    getline(list_file, line);
  }
  
  return true;
}


// Assumes that the input timestamps and poses are sorted by increasing timestamp.
// Performs simple linear interpolation for translations and slerp for rotations.
// For extrapolation, the outermost poses are simply repeated.
// (An alternative would be to also extrapolate linearly, but then some small
//  noise in the outer poses might be inflated as a result.)
bool ComputePosesAtTimestamps(
    vector<double>* timestamps,
    const vector<TimestampedPose>& input_global_tr_frame,
    vector<Pose>* output_global_tr_frame,
    bool delete_non_interpolated,
    double max_interpolation_timespan) {
  if (input_global_tr_frame.empty()) {
    std::cout << "WARNING: ComputePosesAtTimestamps(): Input poses are empty!" << endl;
    return false;
  }
  
  output_global_tr_frame->resize(timestamps->size());
  
  vector<bool> timestamps_to_delete;
  if (delete_non_interpolated) {
    timestamps_to_delete.resize(timestamps->size(), false);
  }
  
  size_t input_poses_index = 0;
  for (size_t i = 0; i < timestamps->size(); ++ i) {
    while (input_poses_index < input_global_tr_frame.size() &&
           input_global_tr_frame[input_poses_index].timestamp <= timestamps->at(i)) {
      ++ input_poses_index;
    }
    
    if (input_poses_index >= input_global_tr_frame.size()) {
      // Cannot interpolate a pose for this timestamp anymore since the last pose timestamp
      // we have is smaller than it. Extrapolate the remaining poses.
      if (i == 0) {
        // We do not have any pose.
        std::cout << "WARNING: Failed to interpolate any pose, maybe the timestamps are incorrect?" << endl;
        return false;
      }
      
      do {
        if (delete_non_interpolated) {
          timestamps_to_delete[i] = true;
        } else {
          output_global_tr_frame->at(i).rotation = input_global_tr_frame.back().rotation;
          output_global_tr_frame->at(i).translation = input_global_tr_frame.back().translation;
        }
        ++ i;
      } while (i < timestamps->size());
      break;
    }
    
    if (input_poses_index == 0) {
      // Extrapolation at the start.
      if (delete_non_interpolated) {
        timestamps_to_delete[i] = true;
      } else {
        output_global_tr_frame->at(i).rotation = input_global_tr_frame[0].rotation;
        output_global_tr_frame->at(i).translation = input_global_tr_frame[0].translation;
      }
    } else {
      // Interpolate pose.
      const TimestampedPose& next_pose = input_global_tr_frame[input_poses_index];
      const TimestampedPose& prev_pose = input_global_tr_frame[input_poses_index - 1];
      
      double interpolation_factor = (timestamps->at(i) - prev_pose.timestamp) / (next_pose.timestamp - prev_pose.timestamp);
      if (interpolation_factor < -1e-6 ||
          interpolation_factor > 1. + 1e-6) {
        std::cout << "WARNING: Internal error: incorrect interpolation factor." << endl;
        return false;
      }
      interpolation_factor = std::max<double>(0, std::min<double>(1, interpolation_factor));
      
      if (delete_non_interpolated &&
          next_pose.timestamp - prev_pose.timestamp > max_interpolation_timespan) {
        timestamps_to_delete[i] = true;
      } else {
        output_global_tr_frame->at(i).rotation =
            prev_pose.rotation.slerp(interpolation_factor, next_pose.rotation);
        output_global_tr_frame->at(i).translation =
            (1 - interpolation_factor) * prev_pose.translation + interpolation_factor * next_pose.translation;
      }
    }
  }
  
  // Perform deletion in timestamps and output_global_tr_frame?
  if (delete_non_interpolated) {
    std::size_t output_index = 0;
    for (std::size_t i = 0; i < timestamps->size(); ++ i) {
      if (!timestamps_to_delete[i]) {
        if (output_index != i) {
          timestamps->at(output_index) = timestamps->at(i);
          output_global_tr_frame->at(output_index) = output_global_tr_frame->at(i);
        }
        ++ output_index;
      }
    }
    timestamps->resize(output_index);
    output_global_tr_frame->resize(output_index);
  }
  
  return true;
}


void WriteTrajectorySVG(
    std::ofstream& stream,
    int plot_size_in_pixels,
    const Vector3d& min_vec,
    const Vector3d& max_vec,
    const vector<double>& timestamps,
    const vector<Pose>& global_tr_image,
    const string& color,
    const float stroke_width,
    int dimension1,
    int dimension2) {
  constexpr double kTimestampHoleThreshold = 0.07;
  
  ostringstream stroke_width_stream;
  stroke_width_stream << stroke_width;
  string stroke_width_string = stroke_width_stream.str();
  
  ostringstream half_stroke_width_stream;
  half_stroke_width_stream << (0.5 * stroke_width);
  string half_stroke_width_string = half_stroke_width_stream.str();
  
  bool within_polyline = false;
  
  for (std::size_t i = 0; i < global_tr_image.size() - 1; ++ i) {
    const Vector3d& point = global_tr_image[i].translation;
    Vector3d plot_point = plot_size_in_pixels * (point - min_vec).cwiseQuotient(max_vec - min_vec);
    
    // Is the segment [i, i + 1] valid (or is it a hole)?
    bool segment_valid = (timestamps[i + 1] - timestamps[i] <= kTimestampHoleThreshold);
    
    if (!segment_valid && !within_polyline) {
      stream << "<circle cx=\"" << plot_point.coeff(dimension1) << "\" cy=\"" << plot_point.coeff(dimension2) << "\" r=\"" << half_stroke_width_string << "\" fill=\"" << color << "\"/>\n";
      continue;
    }
    
    if (segment_valid && !within_polyline) {
      // Start new polyline
      stream << "<polyline points=\"";
      within_polyline = true;
    } else {
      // Write the space between two points
      stream << " ";
    }
    
    stream << plot_point.coeff(dimension1) << "," << plot_point.coeff(dimension2);
    
    if (!segment_valid && within_polyline) {
      // End polyline
      stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
      within_polyline = false;
    }
  }
  
  if (within_polyline) {
    // End polyline
    stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
    // within_polyline = false;
  }
}

void PlotTrajectories(
    const string& path,
    int plot_size_in_pixels,
    const vector<double>& timestamps,
    const vector<Pose>& ground_truth_global_tr_image,
    const vector<Pose>& aligned_global_tr_image,
    int dimension1,
    int dimension2) {
  std::ofstream stream(path, std::ios::out);
  
  stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  stream << "<svg width=\"" << plot_size_in_pixels << "\" height=\"" << plot_size_in_pixels
                          << "\" viewBox=\"0 0 " << plot_size_in_pixels << " " << plot_size_in_pixels
                          << "\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n";
  
  // Determine plot extent based on the ground truth trajectory
  Vector3d min_vec = Vector3d::Constant(std::numeric_limits<double>::infinity());
  Vector3d max_vec = Vector3d::Constant(-1 * std::numeric_limits<double>::infinity());
  for (std::size_t i = 0; i < ground_truth_global_tr_image.size(); ++ i) {
    min_vec = min_vec.cwiseMin(ground_truth_global_tr_image[i].translation);
    max_vec = max_vec.cwiseMax(ground_truth_global_tr_image[i].translation);
  }
  
  double largest_size = (max_vec - min_vec).maxCoeff();
  Vector3d center = 0.5 * (min_vec + max_vec);
  constexpr double kSizeExtensionFactor = 1.1;
  min_vec = center - 0.5 * Vector3d::Constant(kSizeExtensionFactor * largest_size);
  max_vec = center + 0.5 * Vector3d::Constant(kSizeExtensionFactor * largest_size);
  
  // Plot ground truth trajectory
  WriteTrajectorySVG(stream, plot_size_in_pixels, min_vec, max_vec, timestamps, ground_truth_global_tr_image, "green", 1, dimension1, dimension2);
  
  // Plot estimated trajectory
  WriteTrajectorySVG(stream, plot_size_in_pixels, min_vec, max_vec, timestamps, aligned_global_tr_image, "red", 1, dimension1, dimension2);
  
  stream << "</svg>\n";
  
  stream.close();
}


void PlotRelativeError(
    const string& path,
    int plot_width,
    int plot_height,
    const vector<double>& relative_error_timestamps,
    const vector<double>& relative_error_array,
    double max_displayed_error) {
  std::ofstream stream(path, std::ios::out);
  
  stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  stream << "<svg width=\"" << plot_width << "\" height=\"" << plot_height
                          << "\" viewBox=\"0 0 " << plot_width << " " << plot_height
                          << "\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n";
  
  constexpr double kTimestampHoleThreshold = 0.07;
  constexpr double stroke_width = 1;
  const string color = "red";
  
  ostringstream stroke_width_stream;
  stroke_width_stream << stroke_width;
  string stroke_width_string = stroke_width_stream.str();
  
  ostringstream half_stroke_width_stream;
  half_stroke_width_stream << (0.5 * stroke_width);
  string half_stroke_width_string = half_stroke_width_stream.str();
  
  bool within_polyline = false;
  
  for (std::size_t i = 0; i < relative_error_timestamps.size() - 1; ++ i) {
    double x = plot_width * (relative_error_timestamps[i] - relative_error_timestamps.front()) / (relative_error_timestamps.back() - relative_error_timestamps.front());
    double y = plot_height * (1 - (relative_error_array[i] / max_displayed_error));
    
    // Is the segment [i, i + 1] valid (or is it a hole)?
    bool segment_valid = (relative_error_timestamps[i + 1] - relative_error_timestamps[i] <= kTimestampHoleThreshold);
    
    if (!segment_valid && !within_polyline) {
      stream << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"" << half_stroke_width_string << "\" fill=\"" << color << "\"/>\n";
      continue;
    }
    
    if (segment_valid && !within_polyline) {
      // Start new polyline
      stream << "<polyline points=\"";
      within_polyline = true;
    } else {
      // Write the space between two points
      stream << " ";
    }
    
    stream << x << "," << y;
    
    if (!segment_valid && within_polyline) {
      // End polyline
      stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
      within_polyline = false;
    }
  }
  
  if (within_polyline) {
    // End polyline
    stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
    // within_polyline = false;
  }
  
  stream << "</svg>\n";
  
  stream.close();
}


int main(int argc, char** argv) {
  is_little_endian = IsLittleEndian();
  
  // Parse arguments
  if (argc < 4) {
    cout << "Usage:\nETH3DSLAMEvaluation ground_truth.txt estimated_trajectory.txt <rgb.txt or depth.txt> [--sim3] [--max_interpolation_timespan t] [--all_visualizations] [--write_estimated_trajectory_ply]" << endl;
    return static_cast<int>(ReturnCodes::SystemFailure);
  }
  
  string ground_truth_path = argv[1];
  string estimated_trajectory_path = argv[2];
  string rgb_or_depth_file_path = argv[3];
  bool sim3_only = false;
  bool write_visualizations = false;
  bool write_estimated_trajectory_ply = false;
  // Maximum timespan (in seconds) between two ground truth measurements for
  // interpolating between them.
  double max_interpolation_timespan = 1 / 75.;
  
  for (int i = 4; i < argc; ++ i) {
    if (argv[i] == string("--sim3")) {
      sim3_only = true;
    } else if (argv[i] == string("--max_interpolation_timespan")) {
      if (i == argc - 1) {
        cout << "--max_interpolation_timespan requires a parameter." << endl;
        return static_cast<int>(ReturnCodes::SystemFailure);
      }
      
      max_interpolation_timespan = atof(argv[i + 1]);
      i += 1;
    } else if (argv[i] == string("--all_visualizations")) {
      write_visualizations = true;
    } else if (argv[i] == string("--write_estimated_trajectory_ply")) {
      write_estimated_trajectory_ply = true;
    } else {
      cout << "Could not parse argument: " << argv[i] << endl;
      return static_cast<int>(ReturnCodes::SystemFailure);
    }
  }
  
  if (write_visualizations) {
    // Enable all visualizations
    write_estimated_trajectory_ply = true;
  }
  
  
  // Read estimated trajectory
  vector<TimestampedPose> estimated_trajectory_global_tr_frame;
  if (!ReadTrajectory(estimated_trajectory_path, &estimated_trajectory_global_tr_frame)) {
    return static_cast<int>(ReturnCodes::BadInput);
  }
  
  if (estimated_trajectory_global_tr_frame.empty()) {
    std::cout << "WARNING: Input poses are empty. Not outputting any evaluation results.";
    return static_cast<int>(ReturnCodes::Success);
  }
  
  
  // Read all image timestamps from rgb.txt or depth.txt. At these timestamps, the poses will be compared.
  vector<double> image_timestamps;  // in seconds
  if (!ReadImageListFile(
      rgb_or_depth_file_path,
      &image_timestamps)) {
    return static_cast<int>(ReturnCodes::SystemFailure);
  }
  
  if (image_timestamps.size() == 0) {
    cout << "Read 0 image timestamps." << endl;
    return static_cast<int>(ReturnCodes::SystemFailure);
  }
  
  
  // Read and interpolate ground truth poses at the image timestamps to get matching correspondences.
  // Delete timestamps where no ground truth can be interpolated.
  vector<TimestampedPose> ground_truth_global_tr_frame;
  if (!ReadTrajectory(ground_truth_path, &ground_truth_global_tr_frame)) {
    return static_cast<int>(ReturnCodes::SystemFailure);
  }
  
  vector<Pose> ground_truth_global_tr_image(image_timestamps.size());
  if (!ComputePosesAtTimestamps(
      &image_timestamps,
      ground_truth_global_tr_frame,
      &ground_truth_global_tr_image,
      /*delete_non_interpolated*/ true,
      max_interpolation_timespan)) {
    return static_cast<int>(ReturnCodes::SystemFailure);
  }
  
  if (image_timestamps.size() == 0) {
    cout << "No images left after deleting those without interpolated ground truth." << endl;
    return static_cast<int>(ReturnCodes::SystemFailure);
  }
  
  
  // To prepare for the relative error evaluation, compute the cumulative
  // distance of each ground truth pose from the start of the trajectory.
  // NOTE: This is just to get pose pairs that are roughly separated by some
  //       distance. It does not need to be very accurate. Therefore, we only
  //       consider the distances between the poses that are interpolated at
  //       the image timestamps (and disregard the ground truth measurements
  //       in-between).
  vector<double> gt_distance_from_start(image_timestamps.size());
  
  gt_distance_from_start[0] = 0;
  for (std::size_t i = 1; i < image_timestamps.size(); ++ i) {
    double distance = (ground_truth_global_tr_image[i].translation - ground_truth_global_tr_image[i - 1].translation).norm();
    gt_distance_from_start[i] = gt_distance_from_start[i - 1] + distance;
  }
  
  
  // Inter- or extrapolate poses for images with missing pose, if needed
  vector<Pose> estimated_global_tr_image(image_timestamps.size());
  if (!ComputePosesAtTimestamps(
      &image_timestamps,
      estimated_trajectory_global_tr_frame,
      &estimated_global_tr_image,
      /*delete_non_interpolated*/ false,
      numeric_limits<double>::infinity())) {
    std::cout << "WARNING: Failed to interpolate the estimated poses. Not outputting any evaluation results.";
    return static_cast<int>(ReturnCodes::Success);
  }
  
  
  // For Sim3 and / or SE3 ...
  vector<Mode> modes;
  if (!sim3_only) {
    modes.push_back(Mode::SE3);
  }
  modes.push_back(Mode::Sim3);
  
  unordered_map<int, string> mode_names;
  mode_names[static_cast<int>(Mode::Sim3)] = "SIM3_";
  mode_names[static_cast<int>(Mode::SE3)] = "SE3_";
  
  unordered_map<int, string> mode_filename_prefixes;
  mode_filename_prefixes[static_cast<int>(Mode::Sim3)] = "sim3_";
  mode_filename_prefixes[static_cast<int>(Mode::SE3)] = "se3_";
  
  for (Mode mode : modes) {
    // Align the trajectory to the ground truth in the given mode
    Matrix<double, 3, Dynamic> estimated_points;
    estimated_points.resize(NoChange, estimated_global_tr_image.size());
    for (std::size_t i = 0; i < estimated_global_tr_image.size(); ++ i) {
      estimated_points.col(i) = estimated_global_tr_image[i].translation;
    }
    
    Matrix<double, 3, Dynamic> ground_truth_points;
    ground_truth_points.resize(NoChange, ground_truth_global_tr_image.size());
    for (std::size_t i = 0; i < ground_truth_global_tr_image.size(); ++ i) {
      ground_truth_points.col(i) = ground_truth_global_tr_image[i].translation;
    }
    
    // Estimate transform such that: ground_truth_point =approx= transform * estimated_point
    Matrix<double, 4, 4> transform = Eigen::umeyama(estimated_points, ground_truth_points, mode == Mode::Sim3);
    // TODO: In case of mode == Mode::Sim3, check whether the line below handles the scaled rotation correctly, or whether we need to remove the scale
    Quaterniond transform_rotation = Quaterniond(transform.topLeftCorner<3, 3>());
    
    vector<Pose> aligned_global_tr_image(estimated_global_tr_image.size());
    for (std::size_t i = 0; i < estimated_global_tr_image.size(); ++ i) {
      aligned_global_tr_image[i].translation = transform.topLeftCorner<3, 3>() * estimated_global_tr_image[i].translation + transform.topRightCorner<3, 1>();
      aligned_global_tr_image[i].rotation = transform_rotation * estimated_global_tr_image[i].rotation;
    }
    
    
    // Compute ATE RMSE
    double squared_sum = 0;
    int count = 0;
    for (std::size_t i = 0; i < ground_truth_global_tr_image.size(); ++ i) {
      squared_sum += (ground_truth_global_tr_image[i].translation - aligned_global_tr_image[i].translation).squaredNorm();
      count += 1;
    }
    double ate_rmse = std::sqrt(squared_sum / count);
    
    std::cout << mode_names.at(static_cast<int>(mode)) << "ATE_RMSE[cm]: " << std::setprecision(14) << (100 * ate_rmse) << endl;
    
    
    // Compute relative metric(s)
//     constexpr double kTimestampHoleThreshold = 0.07;
    constexpr int kNumEvaluationDistances = 4;
    double evaluation_distances[kNumEvaluationDistances] = {0.5, 1.0, 1.5, 2.0};
    constexpr double kDistanceDeviationThreshold = 0.025;
    
    for (int evaluation_distance_index = 0; evaluation_distance_index < kNumEvaluationDistances; ++ evaluation_distance_index) {
      double evaluation_distance = evaluation_distances[evaluation_distance_index];
      
      double relative_translation_error_sum = 0;  // in percent
      double relative_rotation_error_sum = 0;  // in deg per meter
      int relative_error_count = 0;
      
      vector<double> relative_error_timestamps;
      vector<double> relative_translation_error_array;
      vector<double> relative_rotation_error_array;
      
      // Iterate over all pose pairs in the trajectory which are (approximately) separated by the evaluation distance
      std::size_t second_pose = 0;
//       int latest_hole_end_index = -1;
      for (std::size_t first_pose = 0; first_pose < ground_truth_global_tr_image.size(); ++ first_pose) {
        // Find the next pose which is separated from the first by evaluation_distance
        double target_distance = gt_distance_from_start[first_pose] + evaluation_distance;
        double best_difference = fabs(target_distance - gt_distance_from_start[second_pose]);
        bool finish = false;
        
        while (true) {
          double next_difference = fabs(target_distance - gt_distance_from_start[second_pose + 1]);
          if (next_difference < best_difference) {
            ++ second_pose;
            
            if (second_pose >= ground_truth_global_tr_image.size() - 1) {
              // Reached the last pose, finish.
              finish = true;
              break;
            }
            
//             // Check for a hole
//             if (image_timestamps[second_pose] - image_timestamps[second_pose - 1] > kTimestampHoleThreshold) {
//               latest_hole_end_index = second_pose;
//             }
          } else {
            break;
          }
        }
        
        if (finish) {
          break;
        }
        
//         if (latest_hole_end_index > first_pose &&
//             latest_hole_end_index <= second_pose) {
//           // The segment [first_pose, second_pose] contains at least one trajectory hole.
//           // Thus, do not use it for relative pose evaluation.
//           continue;
//         }
        
        if (first_pose == second_pose) {
          // There is a large hole. We cannot evaluate here.
          continue;
        }
        
        double pose_distance = (gt_distance_from_start[second_pose] - gt_distance_from_start[first_pose]);
        if (fabs(pose_distance - evaluation_distance) > kDistanceDeviationThreshold) {
          // The actual distance between the two picked poses deviates too much from the
          // intended distance for evaluation (possibly due to trajectory holes or very
          // fast movement). Skip this pair.
          continue;
        }
        
        // Evaluate the pose pair with indices (first_pose, second_pose).
        Pose ground_truth_second_tr_first = ground_truth_global_tr_image[second_pose].inverse() * ground_truth_global_tr_image[first_pose];
        Pose estimated_second_tr_first = aligned_global_tr_image[second_pose].inverse() * aligned_global_tr_image[first_pose];
        
        double translation_error = (ground_truth_second_tr_first.translation - estimated_second_tr_first.translation).norm();
        double relative_translation_error = 100. * translation_error / pose_distance;
        
        Quaterniond rotation_error = ground_truth_second_tr_first.rotation.inverse() * estimated_second_tr_first.rotation;
        double angle_error = fabs(AngleAxisd(rotation_error).angle());
        if (angle_error > M_PI) {
          angle_error = 2 * M_PI - angle_error;
        }
        double relative_angle_error = (180. / M_PI) * angle_error / pose_distance;
        
        // Accumulate averaged errors
        relative_translation_error_sum += relative_translation_error;
        relative_rotation_error_sum += relative_angle_error;
        relative_error_count += 1;
        
        // Save individual errors for plots
        relative_error_timestamps.push_back(image_timestamps[first_pose]);
        relative_translation_error_array.push_back(relative_translation_error);
        relative_rotation_error_array.push_back(relative_angle_error);
      }
      
      if (relative_error_count == 0) {
        // No suitable subsequences found to evaluate.
        continue;
      } else if (relative_error_count < 50) {
        std::cout << "WARNING: Found less than 50 subsequences to evaluate relative errors for evaluation_distance " << evaluation_distance << ". The results may be too noisy." << endl;
      }
      
      // TODO: There were some NaN results, check where they came from
      std::cout << mode_names.at(static_cast<int>(mode)) << "REL_TRANSLATION_" << std::setprecision(2) << evaluation_distance << "M[%]: "
                << std::setprecision(14) << (relative_translation_error_sum / relative_error_count) << endl;
      std::cout << mode_names.at(static_cast<int>(mode)) << "REL_ROTATION_" << std::setprecision(2) << evaluation_distance << "M[deg/m]: "
                << std::setprecision(14) << (relative_rotation_error_sum / relative_error_count) << endl;
      
      // Plot relative trajectory errors
      if (write_visualizations) {
        constexpr int kRelativePlotWidth = 800;
        constexpr int kRelativePlotHeight = 400;
        
        ostringstream relative_translation_error_path;
        relative_translation_error_path << mode_filename_prefixes.at(static_cast<int>(mode)) << "relative_translation_error_" << evaluation_distance << "M.svg";
        PlotRelativeError(relative_translation_error_path.str(), kRelativePlotWidth, kRelativePlotHeight, relative_error_timestamps, relative_translation_error_array, 15);
        
        ostringstream relative_rotation_error_path;
        relative_rotation_error_path << mode_filename_prefixes.at(static_cast<int>(mode)) << "relative_rotation_error_" << evaluation_distance << "M.svg";
        PlotRelativeError(relative_rotation_error_path.str(), kRelativePlotWidth, kRelativePlotHeight, relative_error_timestamps, relative_rotation_error_array, 5);
      }
    }
    
    
    // Save trajectories in 3D
    if (write_visualizations) {
      string ply_path = mode_filename_prefixes.at(static_cast<int>(mode)) + "trajectories.ply";
      std::ofstream ply_stream(ply_path, std::ios::out);
      ply_stream << "ply\n";
      ply_stream << "format binary_little_endian 1.0\n";
      ply_stream << "element vertex " << (aligned_global_tr_image.size() + ground_truth_global_tr_image.size()) << "\n";
      ply_stream << "property float x\n";
      ply_stream << "property float y\n";
      ply_stream << "property float z\n";
      ply_stream << "property uint8 red\n";
      ply_stream << "property uint8 green\n";
      ply_stream << "property uint8 blue\n";
      ply_stream << "end_header\n";
      
      // Write ground truth points in green
      for (std::size_t i = 0; i < ground_truth_global_tr_image.size(); ++ i) {
        Vector3f point_float = ground_truth_global_tr_image[i].translation.cast<float>();
        Vector3f little_endian_data;
        little_endian_data[0] = FloatToLittleEndian(point_float[0]);
        little_endian_data[1] = FloatToLittleEndian(point_float[1]);
        little_endian_data[2] = FloatToLittleEndian(point_float[2]);
        ply_stream.write(reinterpret_cast<const char*>(little_endian_data.data()), 3 * sizeof(float));
        
        const unsigned char color[3] = {0, 200, 0};
        ply_stream.write(reinterpret_cast<const char*>(color), 3 * sizeof(unsigned char));
      }
      
      // Write estimated trajectory points in red
      for (std::size_t i = 0; i < aligned_global_tr_image.size(); ++ i) {
        Vector3f point_float = aligned_global_tr_image[i].translation.cast<float>();
        Vector3f little_endian_data;
        little_endian_data[0] = FloatToLittleEndian(point_float[0]);
        little_endian_data[1] = FloatToLittleEndian(point_float[1]);
        little_endian_data[2] = FloatToLittleEndian(point_float[2]);
        ply_stream.write(reinterpret_cast<const char*>(little_endian_data.data()), 3 * sizeof(float));
        
        const unsigned char color[3] = {200, 0, 0};
        ply_stream.write(reinterpret_cast<const char*>(color), 3 * sizeof(unsigned char));
      }
      
      ply_stream.close();
    }
    
    
    // Save estimated trajectory in 3D (without the ground truth trajectory)
    if (write_estimated_trajectory_ply) {
      string ply_path = mode_filename_prefixes.at(static_cast<int>(mode)) + "estimated_trajectory.ply";
      std::ofstream ply_stream(ply_path, std::ios::out);
      ply_stream << "ply\n";
      ply_stream << "format binary_little_endian 1.0\n";
      ply_stream << "element vertex " << aligned_global_tr_image.size() << "\n";
      ply_stream << "property float x\n";
      ply_stream << "property float y\n";
      ply_stream << "property float z\n";
      ply_stream << "property uint8 red\n";
      ply_stream << "property uint8 green\n";
      ply_stream << "property uint8 blue\n";
      ply_stream << "end_header\n";
      
      // Write estimated trajectory points in red
      for (std::size_t i = 0; i < aligned_global_tr_image.size(); ++ i) {
        Vector3f point_float = aligned_global_tr_image[i].translation.cast<float>();
        Vector3f little_endian_data;
        little_endian_data[0] = FloatToLittleEndian(point_float[0]);
        little_endian_data[1] = FloatToLittleEndian(point_float[1]);
        little_endian_data[2] = FloatToLittleEndian(point_float[2]);
        ply_stream.write(reinterpret_cast<const char*>(little_endian_data.data()), 3 * sizeof(float));
        
        const unsigned char color[3] = {200, 0, 0};
        ply_stream.write(reinterpret_cast<const char*>(color), 3 * sizeof(unsigned char));
      }
      
      ply_stream.close();
    }
    
    
    // Plot trajectories in 2D
    if (write_visualizations) {
      constexpr int kPlotSize = 600;  // in pixels; this is the default display size of the SVGs
      
      PlotTrajectories(
          mode_filename_prefixes.at(static_cast<int>(mode)) + "trajectories_top.svg",
          kPlotSize,
          image_timestamps,
          ground_truth_global_tr_image,
          aligned_global_tr_image,
          0,
          1);
      PlotTrajectories(
          mode_filename_prefixes.at(static_cast<int>(mode)) + "trajectories_front.svg",
          kPlotSize,
          image_timestamps,
          ground_truth_global_tr_image,
          aligned_global_tr_image,
          0,
          2);
      PlotTrajectories(
          mode_filename_prefixes.at(static_cast<int>(mode)) + "trajectories_side.svg",
          kPlotSize,
          image_timestamps,
          ground_truth_global_tr_image,
          aligned_global_tr_image,
          1,
          2);
    }
    
    std::cout << endl;
  }
  
  return static_cast<int>(ReturnCodes::Success);
}
