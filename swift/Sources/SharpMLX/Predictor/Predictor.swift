// Main Sharp predictor for 3D Gaussian prediction
// Port of sharp_mlx/predictor.py

import Foundation
import MLX
import MLXNN

// MARK: - DirectPredictionHead

/// Prediction head that converts features to delta values
/// Returns combined delta tensor with shape [B, H, W, 14, num_layers] in NHWC format
public class DirectPredictionHead: Module {
    let numLayers: Int
    @ModuleInfo var geometry_prediction_head: Conv2d
    @ModuleInfo var texture_prediction_head: Conv2d
    
    public init(featureDim: Int, numLayers: Int = 2) {
        self.numLayers = numLayers
        // geometry: 3 * num_layers (means delta)
        // texture: 11 * num_layers (scales, quaternions, colors, opacity)
        let geometryOut = 3 * numLayers
        let textureOut = (14 - 3) * numLayers
        
        self.geometry_prediction_head = Conv2d(inputChannels: featureDim, outputChannels: geometryOut, kernelSize: IntOrPair(1))
        self.texture_prediction_head = Conv2d(inputChannels: featureDim, outputChannels: textureOut, kernelSize: IntOrPair(1))
        
        super.init()
    }
    
    public func callAsFunction(_ features: ImageFeatures) -> MLXArray {
        // texture: [B, H, W, 11 * num_layers]
        let texture = texture_prediction_head(features.textureFeatures)
        // geometry: [B, H, W, 3 * num_layers]
        let geometry = geometry_prediction_head(features.geometryFeatures)
        
        let B = texture.dim(0)
        let H = texture.dim(1)
        let W = texture.dim(2)
        
        // Reshape to [B, H, W, C, num_layers]
        let geometryReshaped = geometry.reshaped(B, H, W, 3, numLayers)
        let textureReshaped = texture.reshaped(B, H, W, 11, numLayers)
        
        // Concatenate: [B, H, W, 14, num_layers]
        return concatenated([geometryReshaped, textureReshaped], axis: 3)
    }
}

// MARK: - SharpPredictor

/// Sharp predictor for 3D Gaussian Splatting from a single image
/// Given a single photograph, predicts the parameters of a 3D Gaussian representation
public class SharpPredictor: Module {
    @ModuleInfo public var init_model: MultiLayerInitializer
    @ModuleInfo public var monodepth_model: MonodepthWithEncodingAdaptor
    @ModuleInfo public var feature_model: GaussianDensePredictionTransformer
    @ModuleInfo public var prediction_head: DirectPredictionHead
    @ModuleInfo public var gaussian_composer: GaussianComposer
    
    public init(
        initModel: MultiLayerInitializer,
        monodepthModel: MonodepthWithEncodingAdaptor,
        featureModel: GaussianDensePredictionTransformer,
        predictionHead: DirectPredictionHead,
        gaussianComposer: GaussianComposer
    ) {
        self.init_model = initModel
        self.monodepth_model = monodepthModel
        self.feature_model = featureModel
        self.prediction_head = predictionHead
        self.gaussian_composer = gaussianComposer
        super.init()
    }
    
    public func callAsFunction(_ image: MLXArray, disparityFactor: MLXArray) -> Gaussians3D {
        // Estimate depth
        let monodepthOutput = monodepth_model(image)
        let monodepthDisparity = monodepthOutput.disparity
        
        // Convert disparity to depth
        let dispFactorExpanded = disparityFactor.expandedDimensions(axes: [1, 2, 3])
        let monodepth = dispFactorExpanded / MLX.clip(monodepthDisparity, min: 1e-4, max: 1e4)
        
        // Initialize base Gaussians
        let initOutput = init_model(image, depth: monodepth)
        
        // Predict delta values
        let imageFeatures = feature_model(initOutput.featureInput, encodings: monodepthOutput.outputFeatures)
        let deltaValues = prediction_head(imageFeatures)
        
        // Compose final Gaussians
        let gaussians = gaussian_composer(
            delta: deltaValues,
            baseValues: initOutput.gaussianBaseValues,
            globalScale: initOutput.globalScale
        )
        
        return gaussians
    }
    
    public func internalResolution() -> Int {
        return monodepth_model.internalResolution()
    }
}

// MARK: - Factory Function

/// Create a Sharp predictor model
public func createPredictor(
    numLayers: Int = 2,
    initStride: Int = 2,
    dimsEncoder: [Int]? = nil,
    dimsDecoder: [Int]? = nil,
    featureDim: Int = 128,
    gaussianStride: Int = 2
) -> SharpPredictor {
    let encoderDims = dimsEncoder ?? [256, 256, 512, 1024, 1024]
    let decoderDims = dimsDecoder ?? [128, 128, 128, 128, 128]
    
    // Monodepth uses 256 for all decoder dims
    let monodepthDecoderDims = [256, 256, 256, 256, 256]
    
    // Create monodepth model
    let monodepthPredictor = createMonodepthDPT(
        dimsEncoder: encoderDims,
        dimsDecoder: monodepthDecoderDims,
        numDisparityChannels: 2,
        usePatchOverlap: true
    )
    let monodepthModel = createMonodepthAdaptor(
        monodepthPredictor: monodepthPredictor,
        returnEncoderFeatures: true,
        returnDecoderFeatures: false,
        numMonodepthLayers: 1,
        sortingMonodepth: false
    )
    
    // Create initializer
    let initModel = MultiLayerInitializer(
        numLayers: numLayers,
        stride: initStride,
        baseDepth: 10.0,
        scaleFactor: 1.0,
        disparityFactor: 1.0,
        normalizeDepth: true
    )
    
    // Create Gaussian decoder
    let gaussianDecoderDims = Array(repeating: 128, count: encoderDims.count)
    let gaussianDecoder = MultiresConvDecoder(
        dimsEncoder: encoderDims,
        dimsDecoder: gaussianDecoderDims,
        upsamplingMode: "transposed_conv"
    )
    
    // Create feature model
    let featureModel = GaussianDensePredictionTransformer(
        decoder: gaussianDecoder,
        dimIn: 5,  // RGB + 2 disparity channels
        dimOut: 32,
        strideOut: gaussianStride,
        normType: "group_norm",
        normNumGroups: 8,
        useDepthInput: true
    )
    
    // Create prediction head
    let predictionHead = DirectPredictionHead(
        featureDim: 32,
        numLayers: numLayers
    )
    
    // Create Gaussian composer
    let gaussianComposer = GaussianComposer(
        deltaFactorXY: 0.001,
        deltaFactorZ: 0.001,
        deltaFactorScale: 1.0,
        deltaFactorQuaternion: 1.0,
        deltaFactorColor: 0.1,
        deltaFactorOpacity: 1.0,
        minScale: 0.0,
        maxScale: 10.0,
        scaleFactor: gaussianStride > initStride ? gaussianStride / initStride : 1,
        baseScaleOnPredictedMean: true
    )
    
    return SharpPredictor(
        initModel: initModel,
        monodepthModel: monodepthModel,
        featureModel: featureModel,
        predictionHead: predictionHead,
        gaussianComposer: gaussianComposer
    )
}
