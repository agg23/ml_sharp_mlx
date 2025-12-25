// Reusable building blocks for Sharp MLX Swift
// Port of sharp_mlx/blocks.py

import Foundation
import MLX
import MLXNN

// MARK: - Type Aliases

public typealias NormLayerName = String  // "noop", "batch_norm", "group_norm", "instance_norm"
public typealias UpsamplingMode = String  // "transposed_conv", "nearest", "bilinear"

// MARK: - PyTorchGroupNorm

/// Wrapper for MLX GroupNorm with pytorch_compatible flag for NHWC format
public class PyTorchGroupNorm: Module {
    let numGroups: Int
    let dims: Int
    let affine: Bool
    @ModuleInfo var norm: GroupNorm
    
    public init(numGroups: Int, numChannels: Int, eps: Float = 1e-5, affine: Bool = true) {
        self.numGroups = numGroups
        self.dims = numChannels
        self.affine = affine
        self.norm = GroupNorm(groupCount: numGroups, dimensions: numChannels, eps: eps, affine: affine, pytorchCompatible: true)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return norm(x)
    }
}

// MARK: - Normalization Layer Factory

/// Create normalization layer for 2D features (NHWC format)
public func normLayer2d(numFeatures: Int, normType: NormLayerName, numGroups: Int = 8) -> Module {
    switch normType {
    case "noop":
        return IdentityModule()
    case "batch_norm":
        return BatchNorm(featureCount: numFeatures)
    case "group_norm":
        return PyTorchGroupNorm(numGroups: numGroups, numChannels: numFeatures)
    case "instance_norm":
        return PyTorchGroupNorm(numGroups: numFeatures, numChannels: numFeatures)
    default:
        fatalError("Invalid normalization layer type: \(normType)")
    }
}

// MARK: - ConvTranspose2d

/// ConvTranspose2d for MLX with NHWC format
/// Note: MLX Swift doesn't have a native ConvTranspose2d, so we implement using ConvolutionTransposed
public class ConvTranspose2d: Module {
    let kernelSize: (Int, Int)
    let stride: (Int, Int)
    let padding: (Int, Int)
    let inChannels: Int
    let outChannels: Int
    
    var weight: MLXArray
    var bias: MLXArray?
    
    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = true
    ) {
        self.kernelSize = (kernelSize, kernelSize)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.inChannels = inChannels
        self.outChannels = outChannels
        
        // Weight shape: (out_channels, kernel_h, kernel_w, in_channels) for MLX conv_transpose2d
        let scale = 1.0 / sqrt(Float(inChannels * kernelSize * kernelSize))
        self.weight = MLXRandom.normal([outChannels, kernelSize, kernelSize, inChannels]) * scale
        
        if bias {
            self.bias = MLXArray.zeros([outChannels])
        } else {
            self.bias = nil
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, H, W, C) - NHWC format
        // MLX expects weight shape: (out_channels, kernel_h, kernel_w, in_channels / groups) 
        var y = MLX.convTransposed2d(x, weight, stride: [stride.0, stride.1], padding: [padding.0, padding.1])
        if let b = bias {
            y = y + b
        }
        return y
    }
}

// MARK: - Upsample

/// Upsampling using interpolation
public class Upsample: Module {
    let scaleFactor: Int
    let mode: String
    
    public init(scaleFactor: Int, mode: String = "nearest") {
        self.scaleFactor = scaleFactor
        self.mode = mode
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, H, W, C) - NHWC format
        if mode == "nearest" || mode == "bilinear" {
            // Nearest neighbor upsampling
            var result = repeated(x, count: scaleFactor, axis: 1)
            result = repeated(result, count: scaleFactor, axis: 2)
            return result
        } else {
            fatalError("Unknown interpolation mode: \(mode)")
        }
    }
}

// MARK: - Upsampling Layer Factory

/// Create upsampling layer
public func upsamplingLayer(mode: UpsamplingMode, scaleFactor: Int, dimIn: Int) -> Module {
    switch mode {
    case "transposed_conv":
        return ConvTranspose2d(
            inChannels: dimIn,
            outChannels: dimIn,
            kernelSize: scaleFactor,
            stride: scaleFactor,
            padding: 0,
            bias: false
        )
    case "nearest", "bilinear":
        return Upsample(scaleFactor: scaleFactor, mode: mode)
    default:
        fatalError("Invalid upsampling mode: \(mode)")
    }
}

// MARK: - ResidualBlock

/// Generic implementation of residual blocks
/// He et al. - Identity Mappings in Deep Residual Networks (2016)
public class ResidualBlock: Module {
    @ModuleInfo var residual: SequentialModule
    @ModuleInfo var shortcut: Conv2d?
    
    public init(residual: SequentialModule, shortcut: Conv2d? = nil) {
        self.residual = residual
        self.shortcut = shortcut
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let deltaX = residual(x)
        var out = x
        if let sc = shortcut {
            out = sc(x)
        }
        return out + deltaX
    }
}

// MARK: - Sequential

/// Sequential container for modules
public class SequentialModule: Module {
    @ModuleInfo var layers: [Module]
    
    public init(_ layers: [Module]) {
        self.layers = layers
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        for layer in layers {
            result = applyLayer(layer, result)
        }
        return result
    }
    
    // Helper to call various module types
    private func applyLayer(_ m: Module, _ x: MLXArray) -> MLXArray {
        if let conv = m as? Conv2d { return conv(x) }
        if let lin = m as? Linear { return lin(x) }
        if let norm = m as? PyTorchGroupNorm { return norm(x) }
        if let relu = m as? ReLUModule { return relu(x) }
        if let bn = m as? BatchNorm { return bn(x) }
        if let id = m as? IdentityModule { return id(x) }
        if let seq = m as? SequentialModule { return seq(x) }
        if let res = m as? ResidualBlock { return res(x) }
        if let ct = m as? ConvTranspose2d { return ct(x) }
        if let up = m as? Upsample { return up(x) }
        if let ffb = m as? FeatureFusionBlock2d { return ffb(x, nil) }
        // Fallback - just return input
        return x
    }
}

// MARK: - ReLU

/// ReLU activation module
public class ReLUModule: Module {
    public override init() {
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return maximum(x, MLXArray(0))
    }
}

// MARK: - Identity

/// Identity module (passthrough)
public class IdentityModule: Module {
    public override init() {
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x
    }
}

// MARK: - Residual Block 2D Factory

/// Create a simple 2D residual block
public func residualBlock2d(
    dimIn: Int,
    dimOut: Int,
    dimHidden: Int? = nil,
    normType: NormLayerName = "noop",
    normNumGroups: Int = 8,
    dilation: Int = 1,
    kernelSize: Int = 3
) -> ResidualBlock {
    let hiddenDim = dimHidden ?? (dimOut / 2)
    
    // Padding to maintain output size
    let padding = (dilation * (kernelSize - 1)) / 2
    
    func createBlock(dIn: Int, dOut: Int) -> [Module] {
        return [
            normLayer2d(numFeatures: dIn, normType: normType, numGroups: normNumGroups),
            ReLUModule(),
            Conv2d(inputChannels: dIn, outputChannels: dOut, kernelSize: IntOrPair(kernelSize), stride: IntOrPair(1), padding: IntOrPair(padding))
        ]
    }
    
    let residualLayers = createBlock(dIn: dimIn, dOut: hiddenDim) + createBlock(dIn: hiddenDim, dOut: dimOut)
    let residual = SequentialModule(residualLayers)
    
    var shortcut: Conv2d? = nil
    if dimIn != dimOut {
        shortcut = Conv2d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: IntOrPair(1))
    }
    
    return ResidualBlock(residual: residual, shortcut: shortcut)
}

// MARK: - FeatureFusionBlock2d

/// Feature fusion block for DPT-style decoders
/// Fuses features at different resolutions with optional upsampling
public class FeatureFusionBlock2d: Module {
    @ModuleInfo var resnet1: ResidualBlock
    @ModuleInfo var resnet2: ResidualBlock
    @ModuleInfo var deconv: Module
    @ModuleInfo var out_conv: Conv2d
    
    public init(
        dimIn: Int,
        dimOut: Int? = nil,
        upsamplingMode: UpsamplingMode? = nil,
        batchNorm: Bool = false
    ) {
        let outputDim = dimOut ?? dimIn
        
        self.resnet1 = FeatureFusionBlock2d.createResidualBlock(numFeatures: dimIn, batchNorm: batchNorm)
        self.resnet2 = FeatureFusionBlock2d.createResidualBlock(numFeatures: dimIn, batchNorm: batchNorm)
        
        if let mode = upsamplingMode {
            self.deconv = upsamplingLayer(mode: mode, scaleFactor: 2, dimIn: dimIn)
        } else {
            self.deconv = IdentityModule()
        }
        
        self.out_conv = Conv2d(inputChannels: dimIn, outputChannels: outputDim, kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0))
        
        super.init()
    }
    
    public func callAsFunction(_ x0: MLXArray, _ x1: MLXArray?) -> MLXArray {
        var x = x0
        
        if let skip = x1 {
            let res = resnet1(skip)
            evalAndClearIfEnabled(res)
            x = x + res
        }
        
        x = resnet2(x)
        evalAndClearIfEnabled(x)
        
        // Apply deconv
        if let ct = deconv as? ConvTranspose2d {
            x = ct(x)
        } else if let up = deconv as? Upsample {
            x = up(x)
        }
        evalAndClearIfEnabled(x)
        
        x = out_conv(x)
        
        return x
    }
    
    /// Create a residual block for fusion
    private static func createResidualBlock(numFeatures: Int, batchNorm: Bool) -> ResidualBlock {
        var layers: [Module] = [
            ReLUModule(),
            Conv2d(inputChannels: numFeatures, outputChannels: numFeatures, kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1))
        ]
        
        if batchNorm {
            layers.append(BatchNorm(featureCount: numFeatures))
        }
        
        layers.append(ReLUModule())
        layers.append(Conv2d(inputChannels: numFeatures, outputChannels: numFeatures, kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)))
        
        if batchNorm {
            layers.append(BatchNorm(featureCount: numFeatures))
        }
        
        let residual = SequentialModule(layers)
        return ResidualBlock(residual: residual, shortcut: nil)
    }
}
