// Configuration for SharpMLX library
// Allows users to control logging and memory management behavior

import Foundation
import MLX

/// Global configuration for SharpMLX library
public struct SharpMLXConfiguration {
    /// Enable memory logging during inference (default: false)
    public var enableLogging: Bool
    
    /// Enable aggressive memory management (default: false)
    /// When enabled, calls eval() and GPU.clearCache() frequently to minimize peak memory
    /// Recommended for memory-constrained devices (e.g., Vision Pro, iPhone, iPad)
    public var aggressiveMemoryManagement: Bool
    
    /// Default configuration with all features disabled
    public static let `default` = SharpMLXConfiguration(
        enableLogging: false,
        aggressiveMemoryManagement: false
    )
    
    /// Configuration optimized for memory-constrained devices
    public static let memoryConstrained = SharpMLXConfiguration(
        enableLogging: false,
        aggressiveMemoryManagement: true
    )
    
    /// Configuration for debugging with full logging and memory management
    public static let debug = SharpMLXConfiguration(
        enableLogging: true,
        aggressiveMemoryManagement: true
    )
    
    public init(enableLogging: Bool = false, aggressiveMemoryManagement: Bool = false) {
        self.enableLogging = enableLogging
        self.aggressiveMemoryManagement = aggressiveMemoryManagement
    }
}

/// Shared configuration instance
/// Set this before running inference to control library behavior
public var sharpMLXConfig = SharpMLXConfiguration.default

// MARK: - Helper Functions

/// Log memory usage if logging is enabled
@inlinable
public func logMemoryIfEnabled(_ label: String, prefix: String = "") {
    guard sharpMLXConfig.enableLogging else { return }
    let snapshot = GPU.snapshot()
    let activeMB = Double(snapshot.activeMemory) / (1024 * 1024)
    let peakMB = Double(snapshot.peakMemory) / (1024 * 1024)
    print("\(prefix)[\(label)] Active: \(String(format: "%.1f", activeMB))MB, Peak: \(String(format: "%.1f", peakMB))MB")
}

/// Synchronize evaluation and clear GPU cache if aggressive memory management is enabled
@inlinable
public func syncAndClearIfEnabled() {
    guard sharpMLXConfig.aggressiveMemoryManagement else { return }
    GPU.clearCache()
}

/// Evaluate arrays and optionally clear cache if aggressive memory management is enabled
@inlinable
public func evalAndClearIfEnabled(_ arrays: MLXArray...) {
    guard sharpMLXConfig.aggressiveMemoryManagement else { return }
    eval(arrays)
    GPU.clearCache()
}

/// Evaluate arrays (variadic helper that accepts array)
@inlinable
public func evalAndClearIfEnabled(_ arrays: [MLXArray]) {
    guard sharpMLXConfig.aggressiveMemoryManagement else { return }
    eval(arrays)
    GPU.clearCache()
}
