// Gaussian prediction modules for Sharp MLX Swift
// Port of sharp_mlx/gaussian.py

import Foundation
import MLX
import MLXNN

// MARK: - Output Structs

/// 3D Gaussian splat parameters
public struct Gaussians3D {
    public var means: MLXArray       // (B, N, 3) or (B, H, W, 3)
    public var scales: MLXArray      // (B, N, 3) or (B, H, W, 3)
    public var quaternions: MLXArray // (B, N, 4) or (B, H, W, 4)
    public var colors: MLXArray      // (B, N, 3) or (B, H, W, 3)
    public var opacities: MLXArray   // (B, N, 1) or (B, H, W, 1)
    
    public var count: Int {
        return means.dim(1)
    }
    
    public init(means: MLXArray, scales: MLXArray, quaternions: MLXArray, colors: MLXArray, opacities: MLXArray) {
        self.means = means
        self.scales = scales
        self.quaternions = quaternions
        self.colors = colors
        self.opacities = opacities
    }
}

/// Base values for Gaussian predictor (NDC coordinates)
public struct GaussianBaseValues {
    public let meanXNdc: MLXArray       // [B, H, W, num_layers]
    public let meanYNdc: MLXArray       // [B, H, W, num_layers]
    public let meanInverseZNdc: MLXArray // [B, H, W, num_layers]
    public let scales: MLXArray         // [B, H, W, num_layers, 3]
    public let quaternions: MLXArray    // [B, H, W, num_layers, 4]
    public let colors: MLXArray         // [B, H, W, num_layers, 3]
    public let opacities: MLXArray      // [B, H, W, num_layers, 1]
}

/// Output of initializer
public struct InitializerOutput {
    public let gaussianBaseValues: GaussianBaseValues
    public let featureInput: MLXArray
    public let globalScale: MLXArray?
}

/// Image features extracted from decoder
public struct ImageFeatures {
    public let textureFeatures: MLXArray
    public let geometryFeatures: MLXArray
}

// MARK: - MultiLayerInitializer

/// Initialize Gaussians with multilayer representation
/// Creates base values for 3D Gaussians from RGB image and depth map
public class MultiLayerInitializer: Module {
    let numLayers: Int
    let stride: Int
    let baseDepth: Float
    let scaleFactor: Float
    let disparityFactor: Float
    let normalizeDepth: Bool
    
    public init(
        numLayers: Int = 2,
        stride: Int = 2,
        baseDepth: Float = 10.0,
        scaleFactor: Float = 1.0,
        disparityFactor: Float = 1.0,
        normalizeDepth: Bool = true
    ) {
        self.numLayers = numLayers
        self.stride = stride
        self.baseDepth = baseDepth
        self.scaleFactor = scaleFactor
        self.disparityFactor = disparityFactor
        self.normalizeDepth = normalizeDepth
        super.init()
    }
    
    public func callAsFunction(_ image: MLXArray, depth: MLXArray) -> InitializerOutput {
        let B = depth.dim(0)
        let H = depth.dim(1)
        let W = depth.dim(2)
        let C = depth.dim(3)
        let baseHeight = H / stride
        let baseWidth = W / stride
        
        var depthNorm = depth
        var globalScale: MLXArray? = nil
        
        if normalizeDepth {
            let (rescaled, factor) = rescaleDepth(depth)
            depthNorm = rescaled
            globalScale = 1.0 / factor
        }
        
        // Create disparity layers
        let disparity = 1.0 / MLX.maximum(depthNorm, MLXArray(1e-4))
        let disparityPooled = maxPool2d(disparity, poolSize: stride)
        
        let disparityLayers: MLXArray
        if numLayers == 1 {
            disparityLayers = disparityPooled[0..., 0..., 0..., 0..<1]
        } else {
            let firstLayer = disparityPooled[0..., 0..., 0..., 0..<1]
            let followingLayer: MLXArray
            if C > 1 {
                followingLayer = disparityPooled[0..., 0..., 0..., 1..<2]
            } else {
                followingLayer = firstLayer
            }
            disparityLayers = concatenated([firstLayer, followingLayer], axis: -1)
        }
        
        // Create base x, y coordinates
        let (baseXNdc, baseYNdc) = createBaseXY(H: H, W: W, B: B)
        
        // Create base scales
        let disparityScaleFactor = 2.0 * scaleFactor * Float(stride) / Float(W)
        var baseScales = (1.0 / disparityLayers) * disparityScaleFactor
        baseScales = baseScales.expandedDimensions(axis: -1)
        baseScales = MLX.broadcast(baseScales, to: [B, baseHeight, baseWidth, numLayers, 3])
        
        // Base quaternions (identity rotation)
        var baseQuaternions = MLXArray([1.0, 0.0, 0.0, 0.0] as [Float])
        baseQuaternions = baseQuaternions.expandedDimensions(axes: [0, 1, 2, 3])
        baseQuaternions = MLX.broadcast(baseQuaternions, to: [B, baseHeight, baseWidth, numLayers, 4])
        
        // Base opacities
        let opacityVal = min(1.0 / Float(numLayers), 0.5)
        var baseOpacities = MLXArray([opacityVal])
        baseOpacities = baseOpacities.expandedDimensions(axes: [0, 1, 2, 3])
        baseOpacities = MLX.broadcast(baseOpacities, to: [B, baseHeight, baseWidth, numLayers, 1])
        
        // Base colors from pooled image
        var imagePooled = avgPool2d(image, poolSize: stride)
        imagePooled = imagePooled.expandedDimensions(axis: 3)
        let baseColors = MLX.broadcast(imagePooled, to: [B, baseHeight, baseWidth, numLayers, 3])
        
        // Prepare feature input
        let normalizedDisparity = disparityFactor / MLX.maximum(depthNorm, MLXArray(1e-4))
        var featuresIn = concatenated([image, normalizedDisparity], axis: -1)
        featuresIn = 2.0 * featuresIn - 1.0
        
        let baseGaussianValues = GaussianBaseValues(
            meanXNdc: baseXNdc,
            meanYNdc: baseYNdc,
            meanInverseZNdc: disparityLayers,
            scales: baseScales,
            quaternions: baseQuaternions,
            colors: baseColors,
            opacities: baseOpacities
        )
        
        return InitializerOutput(
            gaussianBaseValues: baseGaussianValues,
            featureInput: featuresIn,
            globalScale: globalScale
        )
    }
    
    private func createBaseXY(H: Int, W: Int, B: Int) -> (MLXArray, MLXArray) {
        let baseH = H / stride
        let baseW = W / stride
        
        // Create coordinate arrays using linspace
        let xxValues: [Float] = (0..<baseW).map { Float(stride) * 0.5 + Float($0) * Float(stride) }
        let yyValues: [Float] = (0..<baseH).map { Float(stride) * 0.5 + Float($0) * Float(stride) }
        var xx = MLXArray(xxValues)
        var yy = MLXArray(yyValues)
        xx = 2 * xx / Float(W) - 1.0
        yy = 2 * yy / Float(H) - 1.0
        
        // Create meshgrid
        var xxGrid = xx.expandedDimensions(axes: [0, 1])
        xxGrid = MLX.broadcast(xxGrid, to: [B, baseH, baseW])
        
        var yyGrid = yy.expandedDimensions(axes: [0, 2])
        yyGrid = MLX.broadcast(yyGrid, to: [B, baseH, baseW])
        
        // Expand for num_layers
        let baseXNdc = MLX.broadcast(xxGrid.expandedDimensions(axis: 3), to: [B, baseH, baseW, numLayers])
        let baseYNdc = MLX.broadcast(yyGrid.expandedDimensions(axis: 3), to: [B, baseH, baseW, numLayers])
        
        return (baseXNdc, baseYNdc)
    }
    
    private func rescaleDepth(_ depth: MLXArray, depthMin: Float = 1.0, depthMax: Float = 100.0) -> (MLXArray, MLXArray) {
        let B = depth.dim(0)
        let reshaped = depth.reshaped(B, -1)
        let currentDepthMin = reshaped.min(axis: -1)
        let depthFactor = depthMin / (currentDepthMin + 1e-6)
        
        let factorExpanded = depthFactor.expandedDimensions(axes: [1, 2, 3])
        var scaled = depth * factorExpanded
        scaled = MLX.clip(scaled, min: 0.0, max: depthMax)
        
        return (scaled, depthFactor)
    }
    
    private func maxPool2d(_ x: MLXArray, poolSize: Int) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)
        let C = x.dim(3)
        let newH = H / poolSize
        let newW = W / poolSize
        
        let reshaped = x.reshaped(B, newH, poolSize, newW, poolSize, C)
        return reshaped.max(axes: [2, 4])
    }
    
    private func avgPool2d(_ x: MLXArray, poolSize: Int) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)
        let C = x.dim(3)
        let newH = H / poolSize
        let newW = W / poolSize
        
        let reshaped = x.reshaped(B, newH, poolSize, newW, poolSize, C)
        return reshaped.mean(axes: [2, 4])
    }
}

// MARK: - SkipConvBackbone

/// Simple conv layer wrapper for feature extraction
public class SkipConvBackbone: Module {
    let strideVal: Int
    @ModuleInfo var conv: Conv2d
    
    public init(dimIn: Int, dimOut: Int, kernelSize: Int, stride: Int) {
        self.strideVal = stride
        self.conv = Conv2d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: .init(stride), stride: .init(stride), padding: .init(0))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> ImageFeatures {
        let out = conv(x)
        return ImageFeatures(textureFeatures: out, geometryFeatures: out)
    }
}

// MARK: - GaussianDensePredictionTransformer

/// Dense Prediction Transformer for Gaussian parameters
/// Reuses monodepth features to predict delta values for Gaussians
public class GaussianDensePredictionTransformer: Module {
    @ModuleInfo var decoder: MultiresConvDecoder
    let dimIn: Int
    let dimOut: Int
    let strideOut: Int
    let useDepthInput: Bool
    
    @ModuleInfo var image_encoder: SkipConvBackbone
    @ModuleInfo var fusion: FeatureFusionBlock2d
    @ModuleInfo var upsample: Module
    @ModuleInfo var texture_head: SequentialModule
    @ModuleInfo var geometry_head: SequentialModule
    
    public init(
        decoder: MultiresConvDecoder,
        dimIn: Int,
        dimOut: Int,
        strideOut: Int = 2,
        normType: NormLayerName = "group_norm",
        normNumGroups: Int = 8,
        useDepthInput: Bool = true
    ) {
        self.decoder = decoder
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.strideOut = strideOut
        self.useDepthInput = useDepthInput
        
        let actualDimIn = useDepthInput ? dimIn : dimIn - 1
        let kernelSize = strideOut != 1 ? 3 : 1
        self.image_encoder = SkipConvBackbone(dimIn: actualDimIn, dimOut: decoder.dim_out, kernelSize: kernelSize, stride: strideOut)
        
        self.fusion = FeatureFusionBlock2d(dimIn: decoder.dim_out)
        
        if strideOut == 1 {
            self.upsample = ConvTranspose2d(
                inChannels: decoder.dim_out,
                outChannels: decoder.dim_out,
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: false
            )
        } else {
            self.upsample = IdentityModule()
        }
        
        self.texture_head = GaussianDensePredictionTransformer.createHead(
            dimDecoder: decoder.dim_out, dimOut: dimOut, normType: normType, normNumGroups: normNumGroups
        )
        self.geometry_head = GaussianDensePredictionTransformer.createHead(
            dimDecoder: decoder.dim_out, dimOut: dimOut, normType: normType, normNumGroups: normNumGroups
        )
        
        super.init()
    }
    
    private static func createHead(dimDecoder: Int, dimOut: Int, normType: NormLayerName, normNumGroups: Int) -> SequentialModule {
        return SequentialModule([
            residualBlock2d(dimIn: dimDecoder, dimOut: dimDecoder, dimHidden: dimDecoder / 2, normType: normType, normNumGroups: normNumGroups),
            residualBlock2d(dimIn: dimDecoder, dimOut: dimDecoder, dimHidden: dimDecoder / 2, normType: normType, normNumGroups: normNumGroups),
            ReLUModule(),
            Conv2d(inputChannels: dimDecoder, outputChannels: dimOut, kernelSize: IntOrPair(1), stride: IntOrPair(1)),
            ReLUModule()
        ])
    }
    
    public func callAsFunction(_ inputFeatures: MLXArray, encodings: [MLXArray]) -> ImageFeatures {
        // Only use the first N encodings that match decoder input dims
        let numDecoderLevels = decoder.convs.count
        let encoderFeatures = Array(encodings[0..<numDecoderLevels])
        
        var features = decoder(encoderFeatures)
        
        // Apply upsample
        if let ct = upsample as? ConvTranspose2d {
            features = ct(features)
        }
        
        let skipInput = useDepthInput ? inputFeatures : inputFeatures[0..., 0..., 0..., 0..<3]
        let skipFeatures = image_encoder(skipInput).textureFeatures
        
        features = fusion(features, skipFeatures)
        
        let textureFeatures = texture_head(features)
        let geometryFeatures = geometry_head(features)
        
        return ImageFeatures(textureFeatures: textureFeatures, geometryFeatures: geometryFeatures)
    }
}

// MARK: - GaussianComposer

/// Converts base values and deltas into final Gaussians
/// Applies activations and combines base + delta values
public class GaussianComposer: Module {
    let deltaFactorXY: Float
    let deltaFactorZ: Float
    let deltaFactorScale: Float
    let deltaFactorQuaternion: Float
    let deltaFactorColor: Float
    let deltaFactorOpacity: Float
    let minScale: Float
    let maxScale: Float
    let scaleFactorVal: Int
    let baseScaleOnPredictedMean: Bool
    
    public init(
        deltaFactorXY: Float = 0.001,
        deltaFactorZ: Float = 0.001,
        deltaFactorScale: Float = 1.0,
        deltaFactorQuaternion: Float = 1.0,
        deltaFactorColor: Float = 0.1,
        deltaFactorOpacity: Float = 1.0,
        minScale: Float = 0.0,
        maxScale: Float = 10.0,
        scaleFactor: Int = 1,
        baseScaleOnPredictedMean: Bool = true
    ) {
        self.deltaFactorXY = deltaFactorXY
        self.deltaFactorZ = deltaFactorZ
        self.deltaFactorScale = deltaFactorScale
        self.deltaFactorQuaternion = deltaFactorQuaternion
        self.deltaFactorColor = deltaFactorColor
        self.deltaFactorOpacity = deltaFactorOpacity
        self.minScale = minScale
        self.maxScale = maxScale
        self.scaleFactorVal = scaleFactor
        self.baseScaleOnPredictedMean = baseScaleOnPredictedMean
        super.init()
    }
    
    public func callAsFunction(
        delta: MLXArray,
        baseValues: GaussianBaseValues,
        globalScale: MLXArray?,
        flattenOutput: Bool = true
    ) -> Gaussians3D {
        var deltaVal = delta
        
        // Upsample delta if needed
        if scaleFactorVal > 1 {
            deltaVal = repeated(deltaVal, count: scaleFactorVal, axis: 1)
            deltaVal = repeated(deltaVal, count: scaleFactorVal, axis: 2)
        }
        
        // Transpose: [B, H, W, 14, num_layers] -> [B, H, W, num_layers, 14]
        deltaVal = deltaVal.transposed(0, 1, 2, 4, 3)
        
        // Extract components
        let meanDelta = deltaVal[0..., 0..., 0..., 0..., 0..<3]
        let scaleDelta = deltaVal[0..., 0..., 0..., 0..., 3..<6]
        let quatDelta = deltaVal[0..., 0..., 0..., 0..., 6..<10]
        let colorDelta = deltaVal[0..., 0..., 0..., 0..., 10..<13]
        let opacityDelta = deltaVal[0..., 0..., 0..., 0..., 13..<14]
        
        // Mean activation
        let meanX = baseValues.meanXNdc + meanDelta[0..., 0..., 0..., 0..., 0] * deltaFactorXY
        let meanY = baseValues.meanYNdc + meanDelta[0..., 0..., 0..., 0..., 1] * deltaFactorXY
        
        // Inverse depth activation with softplus
        let inverseZBase = baseValues.meanInverseZNdc
        let inverseZDeltaMul = meanDelta[0..., 0..., 0..., 0..., 2] * deltaFactorZ
        
        let eps: Float = 1e-4
        let inverseZActivated = MLX.log(1.0 + MLX.exp(
            MLX.log(MLX.maximum(MLX.exp(inverseZBase) - 1.0, MLXArray(eps))) + inverseZDeltaMul
        ))
        let meanZ = 1.0 / (inverseZActivated + 1e-3)
        
        // Construct mean vectors (NDC to metric: multiply x,y by z)
        let means = stacked([
            meanZ * meanX,
            meanZ * meanY,
            meanZ
        ], axis: -1)
        
        // Scale activation
        var baseScales = baseValues.scales
        if baseScaleOnPredictedMean {
            let invZExpanded = baseValues.meanInverseZNdc.expandedDimensions(axis: -1)
            let meanZExpanded = meanZ.expandedDimensions(axis: -1)
            baseScales = baseScales * invZExpanded * meanZExpanded
        }
        
        // Sigmoid-based scale activation
        let constantA = (maxScale - minScale) / (1 - minScale) / (maxScale - 1)
        let ratio = (1.0 - minScale) / (maxScale - minScale)
        let constantB = MLX.log(MLXArray(ratio / (1.0 - ratio + 1e-8)))
        let scaleFactorMult = (maxScale - minScale) * sigmoid(constantA * scaleDelta * deltaFactorScale + constantB) + minScale
        var scales = baseScales * scaleFactorMult
        
        // Quaternion: base + delta
        var quats = baseValues.quaternions + quatDelta * deltaFactorQuaternion
        
        // Color activation (sigmoid-based)
        let baseColorsClipped = MLX.clip(baseValues.colors, min: 0.01, max: 0.99)
        let invSigmoidBase = MLX.log(baseColorsClipped / (1.0 - baseColorsClipped + 1e-8))
        var colors = sigmoid(invSigmoidBase + colorDelta * deltaFactorColor)
        
        // sRGB to linearRGB conversion
        let threshold: Float = 0.04045
        let colorsLinear = MLX.where(
            colors .<= threshold,
            colors / 12.92,
            MLX.pow((colors + 0.055) / 1.055, MLXArray(2.4))
        )
        
        // Opacity activation
        let baseOpacitiesClipped = MLX.clip(baseValues.opacities, min: 0.01, max: 0.99)
        let invSigmoidBaseOp = MLX.log(baseOpacitiesClipped / (1.0 - baseOpacitiesClipped + 1e-8))
        var opacities = sigmoid(invSigmoidBaseOp + opacityDelta * deltaFactorOpacity)
        
        // Apply global scaling
        var meansOut = means
        var scalesOut = scales
        if let gs = globalScale {
            let gsExpanded = gs.expandedDimensions(axes: [1, 2, 3, 4])
            meansOut = meansOut * gsExpanded
            scalesOut = scalesOut * gsExpanded
        }
        
        var colorsOut = colorsLinear
        var quatsOut = quats
        var opacitiesOut = opacities
        
        if flattenOutput {
            let B = meansOut.dim(0)
            meansOut = meansOut.reshaped(B, -1, 3)
            scalesOut = scalesOut.reshaped(B, -1, 3)
            quatsOut = quatsOut.reshaped(B, -1, 4)
            colorsOut = colorsOut.reshaped(B, -1, 3)
            opacitiesOut = opacitiesOut.reshaped(B, -1, 1)
        }
        
        return Gaussians3D(
            means: meansOut,
            scales: scalesOut,
            quaternions: quatsOut,
            colors: colorsOut,
            opacities: opacitiesOut
        )
    }
}
