// Multi-resolution convolutional decoder for Sharp MLX Swift
// Port of sharp_mlx/decoder.py

import Foundation
import MLX
import MLXNN

// MARK: - MultiresConvDecoder

/// Decoder for multi-resolution encodings
/// Progressively fuses features from low resolution to high resolution
public class MultiresConvDecoder: Module {
    let dims_encoder: [Int]
    let dims_decoder: [Int]
    public let dim_out: Int
    
    @ModuleInfo var convs: [Module]
    @ModuleInfo var fusions: [FeatureFusionBlock2d]
    
    public init(
        dimsEncoder: [Int],
        dimsDecoder: [Int],
        upsamplingMode: UpsamplingMode = "transposed_conv"
    ) {
        self.dims_encoder = dimsEncoder
        
        // Expand dimsDecoder if single value
        if dimsDecoder.count == 1 {
            self.dims_decoder = Array(repeating: dimsDecoder[0], count: dimsEncoder.count)
        } else {
            self.dims_decoder = dimsDecoder
        }
        
        precondition(self.dims_decoder.count == dimsEncoder.count,
                     "Received dims_encoder and dims_decoder of different sizes.")
        
        self.dim_out = self.dims_decoder[0]
        let numEncoders = dimsEncoder.count
        
        // Projection convolutions (bias=False to match PyTorch checkpoint)
        var convsList: [Module] = []
        for i in 0..<numEncoders {
            let conv: Module
            if i == 0 {
                // At highest resolution, use 1x1 conv if dimensions differ
                if dimsEncoder[i] != self.dims_decoder[i] {
                    conv = Conv2d(inputChannels: dimsEncoder[i], outputChannels: self.dims_decoder[i], kernelSize: IntOrPair(1), bias: false)
                } else {
                    conv = IdentityModule()
                }
            } else {
                conv = Conv2d(inputChannels: dimsEncoder[i], outputChannels: self.dims_decoder[i], kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1), bias: false)
            }
            convsList.append(conv)
        }
        self.convs = convsList
        
        // Fusion blocks
        var fusionsList: [FeatureFusionBlock2d] = []
        for i in 0..<numEncoders {
            let dimOutput = i != 0 ? self.dims_decoder[i - 1] : self.dim_out
            let mode: UpsamplingMode? = i != 0 ? upsamplingMode : nil
            fusionsList.append(
                FeatureFusionBlock2d(
                    dimIn: self.dims_decoder[i],
                    dimOut: dimOutput,
                    upsamplingMode: mode,
                    batchNorm: false
                )
            )
        }
        self.fusions = fusionsList
        
        super.init()
    }
    
    public func callAsFunction(_ encodings: [MLXArray]) -> MLXArray {
        let numLevels = encodings.count
        let numEncoders = dims_encoder.count
        
        precondition(numLevels == numEncoders,
                     "Encoder output levels=\(numLevels) mismatch with expected levels=\(numEncoders).")
        
        // Project features and fuse from lowest resolution to highest
        var features = applyConv(convs[numLevels - 1], encodings[numLevels - 1])
        features = fusions[numLevels - 1](features, nil)
        
        for i in stride(from: numLevels - 2, through: 0, by: -1) {
            let featuresI = applyConv(convs[i], encodings[i])
            features = fusions[i](features, featuresI)
        }
        
        return features
    }
    
    // Helper to apply conv module
    private func applyConv(_ conv: Module, _ x: MLXArray) -> MLXArray {
        if let c = conv as? Conv2d {
            return c(x)
        } else if let id = conv as? Identity {
            return id(x)
        }
        return x
    }
}

// MARK: - BaseDecoder

/// Base class for decoders
public class BaseDecoder: Module {
    public var dim_out: Int = 0
    
    public override init() {
        super.init()
    }
}
