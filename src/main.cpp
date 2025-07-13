// main.cpp - Complete implementation with performance benchmarking
#include "ultratrack.hpp"
#include "version.hpp"
#include <iostream>
#include <chrono>
#include <numeric>
#include <fstream>
#include <filesystem>

void print_usage() {
    std::cout << "UltraTracker " << ultratrack::Version::get_version_string() << " - High-Performance Object Tracking\n";
    std::cout << "Usage: ultratrack [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input <path>     Input video file or camera index (default: 0)\n";
    std::cout << "  -o, --output <path>    Output video file (optional)\n";
    std::cout << "  -m, --model <path>     YOLOv11 model path (default: models/yolov11n.onnx)\n";
    std::cout << "  -f, --features <path>  Feature model path (default: models/resnet50_features.onnx)\n";
    std::cout << "  -c, --confidence <val> Confidence threshold (default: 0.3)\n";
    std::cout << "  -l, --learning <val>   Learning rate (default: 0.02)\n";
    std::cout << "  -b, --benchmark        Run benchmark mode\n";
    std::cout << "  -v, --version          Show version information\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  ultratrack                              # Track from camera 0\n";
    std::cout << "  ultratrack -i video.mp4 -o output.mp4  # Process video file\n";
    std::cout << "  ultratrack -b -c 0.5                   # Benchmark with higher confidence\n";
}

struct Config {
    std::string input = "0";
    std::string output = "";
    std::string model_path = "models/yolov11n.onnx";
    std::string features_path = "models/resnet50_features.onnx";
    float confidence = 0.3f;
    float learning_rate = 0.02f;
    bool benchmark = false;
    bool help = false;
    bool version = false;
};

Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            config.help = true;
        } else if (arg == "-v" || arg == "--version") {
            config.version = true;
        } else if (arg == "-b" || arg == "--benchmark") {
            config.benchmark = true;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            config.input = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.output = argv[++i];
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if ((arg == "-f" || arg == "--features") && i + 1 < argc) {
            config.features_path = argv[++i];
        } else if ((arg == "-c" || arg == "--confidence") && i + 1 < argc) {
            config.confidence = std::stof(argv[++i]);
        } else if ((arg == "-l" || arg == "--learning") && i + 1 < argc) {
            config.learning_rate = std::stof(argv[++i]);
        }
    }
    
    return config;
}

bool check_model_files(const Config& config) {
    if (!std::filesystem::exists(config.model_path)) {
        std::cerr << "Error: Model file not found: " << config.model_path << std::endl;
        std::cerr << "Please download YOLOv11 model or adjust the path." << std::endl;
        return false;
    }
    
    if (!config.features_path.empty() && !std::filesystem::exists(config.features_path)) {
        std::cout << "Warning: Feature model not found: " << config.features_path << std::endl;
        std::cout << "Continuing without deep feature extraction." << std::endl;
    }
    
    return true;
}

void save_performance_report(const std::vector<double>& processing_times, 
                           const std::vector<int>& track_counts,
                           int total_frames) {
    std::ofstream report("performance_report.txt");
    
    double avg_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) 
                     / processing_times.size();
    double fps = 1000.0 / avg_time;
    
    double avg_tracks = std::accumulate(track_counts.begin(), track_counts.end(), 0.0) 
                       / track_counts.size();
    
    auto minmax_time = std::minmax_element(processing_times.begin(), processing_times.end());
    auto minmax_tracks = std::minmax_element(track_counts.begin(), track_counts.end());
    
    report << "UltraTracker Performance Report\n";
    report << "==============================\n\n";
    report << "Processing Statistics:\n";
    report << "  Total frames processed: " << total_frames << "\n";
    report << "  Average processing time: " << avg_time << " ms\n";
    report << "  Average FPS: " << fps << "\n";
    report << "  Min processing time: " << *minmax_time.first << " ms\n";
    report << "  Max processing time: " << *minmax_time.second << " ms\n\n";
    report << "Tracking Statistics:\n";
    report << "  Average active tracks: " << avg_tracks << "\n";
    report << "  Min active tracks: " << *minmax_tracks.first << "\n";
    report << "  Max active tracks: " << *minmax_tracks.second << "\n\n";
    report << "Performance Targets:\n";
    report << "  Target (Jetson Orin NX): < 1.5ms (vs CvTracker's 2.1ms)\n";
    report << "  Target (Raspberry Pi 5): < 1.0ms (vs CvTracker's 2.3ms)\n";
    
    if (avg_time < 1.5) {
        report << "  ✓ Exceeds Jetson Orin NX target!\n";
    }
    if (avg_time < 1.0) {
        report << "  ✓ Exceeds Raspberry Pi 5 target!\n";
    }
    
    report.close();
    std::cout << "Performance report saved to performance_report.txt\n";
}

int main(int argc, char* argv[]) {
    Config config = parse_args(argc, argv);
    
    if (config.help) {
        print_usage();
        return 0;
    }
    
    if (config.version) {
        std::cout << ultratrack::Version::get_build_info() << std::endl;
        return 0;
    }
    
    // Check if model files exist
    if (!check_model_files(config)) {
        return -1;
    }
    
    try {
        // Initialize UltraTracker
        std::cout << "Initializing UltraTracker..." << std::endl;
        ultratrack::UltraTracker tracker(config.model_path, config.features_path);
        tracker.set_confidence_threshold(config.confidence);
        tracker.set_learning_rate(config.learning_rate);
        
        std::cout << "UltraTracker initialized successfully." << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Model: " << config.model_path << std::endl;
        std::cout << "  Features: " << (config.features_path.empty() ? "None" : config.features_path) << std::endl;
        std::cout << "  Confidence threshold: " << config.confidence << std::endl;
        std::cout << "  Learning rate: " << config.learning_rate << std::endl;
        
        // Open input source
        cv::VideoCapture cap;
        bool is_camera = false;
        
        // Try to parse as camera index
        try {
            int camera_index = std::stoi(config.input);
            cap.open(camera_index);
            is_camera = true;
            std::cout << "Opened camera " << camera_index << std::endl;
        } catch (const std::exception&) {
            // Not a number, treat as file path
            cap.open(config.input);
            std::cout << "Opened video file: " << config.input << std::endl;
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open input source: " << config.input << std::endl;
            std::cerr << "Please check that the camera/file exists and is accessible." << std::endl;
            return -1;
        }
        
        // Get video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        if (fps <= 0) fps = 30.0; // Default for cameras
        
        std::cout << "Video properties: " << frame_width << "x" << frame_height 
                  << " @ " << fps << " FPS" << std::endl;
        
        // Setup output video writer if specified
        cv::VideoWriter writer;
        if (!config.output.empty()) {
            cv::Size frame_size(frame_width, frame_height);
            
            writer.open(config.output, cv::VideoWriter::fourcc('M','P','4','V'), 
                       fps, frame_size, true);
            
            if (!writer.isOpened()) {
                std::cerr << "Warning: Cannot open output video file: " << config.output << std::endl;
                std::cerr << "Make sure the output directory exists and is writable." << std::endl;
            } else {
                std::cout << "Recording to: " << config.output << std::endl;
            }
        }
        
        cv::Mat frame;
        std::vector<double> processing_times;
        std::vector<int> track_counts;
        int frame_count = 0;
        int error_count = 0;
        const int max_errors = 10;
        
        std::cout << "Starting tracking... Press ESC to exit, SPACE to pause." << std::endl;
        
        bool paused = false;
        
        while (cap.read(frame)) {
            if (frame.empty()) {
                std::cerr << "Warning: Empty frame encountered at frame " << frame_count << std::endl;
                error_count++;
                if (error_count > max_errors) {
                    std::cerr << "Too many errors, stopping." << std::endl;
                    break;
                }
                continue;
            }
            
            if (paused) {
                cv::imshow("UltraTracker", frame);
                int key = cv::waitKey(30);
                if (key == 27) break; // ESC
                if (key == 32) paused = false; // SPACE
                continue;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            try {
                // Process frame
                std::vector<ultratrack::Detection> detections;
                tracker.update(frame, detections);
                auto tracks = tracker.get_active_tracks();
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                double processing_time = duration.count() / 1000.0; // Convert to milliseconds
                processing_times.push_back(processing_time);
                track_counts.push_back(static_cast<int>(tracks.size()));
                
                // Visualize results
                for (const auto& track : tracks) {
                    // Draw bounding box
                    cv::Scalar color = cv::Scalar(0, 255, 0); // Green for confirmed tracks
                    if (track.age < 5) color = cv::Scalar(0, 255, 255); // Yellow for new tracks
                    
                    cv::rectangle(frame, track.bbox, color, 2);
                    
                    // Draw track ID and confidence
                    std::string label = "ID:" + std::to_string(track.id) + 
                                      " (" + std::to_string(static_cast<int>(track.confidence * 100)) + "%)";
                    
                    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
                    cv::Point label_pos(static_cast<int>(track.bbox.x), 
                                       static_cast<int>(track.bbox.y) - 10);
                    
                    // Ensure label is within frame bounds
                    if (label_pos.y < 0) label_pos.y = static_cast<int>(track.bbox.y + track.bbox.height + 20);
                    
                    // Background for text
                    cv::rectangle(frame, 
                                cv::Point(label_pos.x, label_pos.y - label_size.height - 5),
                                cv::Point(label_pos.x + label_size.width + 5, label_pos.y + 5),
                                color, -1);
                    
                    cv::putText(frame, label, label_pos,
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                }
                
                // Add performance info
                if (config.benchmark || is_camera) {
                    std::string perf_info = "FPS: " + std::to_string(static_cast<int>(1000.0 / processing_time)) +
                                           " | Tracks: " + std::to_string(tracks.size()) +
                                           " | Time: " + std::to_string(processing_time) + "ms";
                    
                    cv::putText(frame, perf_info, cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                }
                
                // Add pause indicator
                if (paused) {
                    cv::putText(frame, "PAUSED - Press SPACE to continue", cv::Point(10, frame.rows - 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing frame " << frame_count << ": " << e.what() << std::endl;
                error_count++;
                if (error_count > max_errors) {
                    std::cerr << "Too many processing errors, stopping." << std::endl;
                    break;
                }
            }
            
            // Show frame
            cv::imshow("UltraTracker", frame);
            
            // Write to output video if specified
            if (writer.isOpened()) {
                try {
                    writer.write(frame);
                } catch (const cv::Exception& e) {
                    std::cerr << "Warning: Failed to write frame to output video: " << e.what() << std::endl;
                }
            }
            
            // Check for exit
            int key = cv::waitKey(1);
            if (key == 27) break; // ESC
            if (key == 32) paused = !paused; // SPACE
            
            frame_count++;
            
            // Limit benchmark frames if not from camera
            if (config.benchmark && !is_camera && frame_count >= 1000) {
                break;
            }
            
            // Print progress for long videos
            if (frame_count % 100 == 0) {
                if (!processing_times.empty()) {
                    double avg_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) 
                                    / processing_times.size();
                    std::cout << "Processed " << frame_count << " frames, avg: " 
                             << avg_time << "ms (" << static_cast<int>(1000.0 / avg_time) << " FPS)" << std::endl;
                }
            }
        }
        
        // Cleanup
        cap.release();
        if (writer.isOpened()) writer.release();
        cv::destroyAllWindows();
        
        // Performance statistics
        if (!processing_times.empty()) {
            double avg_time = std::accumulate(processing_times.begin(), processing_times.end(), 0.0) 
                             / processing_times.size();
            double fps = 1000.0 / avg_time;
            double avg_tracks = std::accumulate(track_counts.begin(), track_counts.end(), 0.0) 
                               / track_counts.size();
            
            std::cout << "\n=== Performance Summary ===" << std::endl;
            std::cout << "Frames processed: " << frame_count << std::endl;
            std::cout << "Processing errors: " << error_count << std::endl;
            std::cout << "Average processing time: " << avg_time << " ms" << std::endl;
            std::cout << "Average FPS: " << fps << std::endl;
            std::cout << "Average active tracks: " << avg_tracks << std::endl;
            
            // Save detailed report
            if (config.benchmark) {
                save_performance_report(processing_times, track_counts, frame_count);
            }
            
            // Performance comparison
            std::cout << "\n=== Performance Targets ===" << std::endl;
            std::cout << "Target (Jetson Orin NX): < 1.5ms (vs CvTracker's 2.1ms)" << std::endl;
            std::cout << "Target (Raspberry Pi 5): < 1.0ms (vs CvTracker's 2.3ms)" << std::endl;
            
            if (avg_time < 1.5) {
                std::cout << "✓ Exceeds Jetson Orin NX target!" << std::endl;
            }
            if (avg_time < 1.0) {
                std::cout << "✓ Exceeds Raspberry Pi 5 target!" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
