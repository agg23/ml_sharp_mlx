// Monodepth Dense Prediction Transformer for Sharp MLX Swift
// Port of sharp_mlx/monodepth.py

import Foundation
import MLX
import MLXNN

// MARK: - AffineRangeNormalizer

/// Normalize input from one range to another
public class AffineRangeNormalizer: Module {
    let scale: Float
    let shift: Float
    
    public init(inputRange: (Float, Float) = (0, 1), outputRange: (Float, Float) = (-1, 1)) {
        let inputSpan = inputRange.1 - inputRange.0
        let outputSpan = outputRange.1 - outputRange.0
        self.scale = outputSpan / inputSpan
        self.shift = outputRange.0 - inputRange.0 * scale
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * scale + shift
    }
}

// MARK: - MonodepthOutput

/// Output of the monodepth model
public struct MonodepthOutput {
    public let disparity: MLXArray
    public let encoderFeatures: [MLXArray]
    public let decoderFeatures: MLXArray
    public let outputFeatures: [MLXArray]
    public let intermediateFeatures: [MLXArray]
    
    public init(
        disparity: MLXArray,
        encoderFeatures: [MLXArray],
        decoderFeatures: MLXArray,
        outputFeatures: [MLXArray],
        intermediateFeatures: [MLXArray] = []
    ) {
        self.disparity = disparity
        self.encoderFeatures = encoderFeatures
        self.decoderFeatures = decoderFeatures
        self.outputFeatures = outputFeatures
        self.intermediateFeatures = intermediateFeatures
    }
}

// MARK: - MonodepthDensePredictionTransformer

/// Dense Prediction Transformer for monodepth
/// Combines SPN encoder + MultiresConvDecoder + disparity head
public class MonodepthDensePredictionTransformer: Module {
    @ModuleInfo public var normalizer: AffineRangeNormalizer
    @ModuleInfo public var encoder: SlidingPyramidNetwork
    @ModuleInfo public var decoder: MultiresConvDecoder
    @ModuleInfo public var head: [Module]
    
    public init(
        encoder: SlidingPyramidNetwork,
        decoder: MultiresConvDecoder,
        lastDims: (Int, Int) = (32, 1)
    ) {
        self.normalizer = AffineRangeNormalizer(inputRange: (0, 1), outputRange: (-1, 1))
        self.encoder = encoder
        self.decoder = decoder
        
        let dimDecoder = decoder.dim_out
        
        // Disparity prediction head
        self.head = [
            Conv2d(inputChannels: dimDecoder, outputChannels: dimDecoder / 2, kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
            ConvTranspose2d(
                inChannels: dimDecoder / 2,
                outChannels: dimDecoder / 2,
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: true
            ),
            Conv2d(inputChannels: dimDecoder / 2, outputChannels: lastDims.0, kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
            ReLUModule(),
            Conv2d(inputChannels: lastDims.0, outputChannels: lastDims.1, kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
            ReLUModule()
        ]
        
        super.init()
    }
    
    public func callAsFunction(_ image: MLXArray) -> MLXArray {
        let encodings = encoder(normalizer(image))
        let numEncoderFeatures = encoder.dims_encoder.count
        var features = decoder(Array(encodings[0..<numEncoderFeatures]))
        
        // Apply head
        var disparity = features
        for layer in head {
            disparity = applyHeadLayer(layer, disparity)
        }
        
        return disparity
    }
    
    private func applyHeadLayer(_ layer: Module, _ x: MLXArray) -> MLXArray {
        if let conv = layer as? Conv2d { return conv(x) }
        if let ct = layer as? ConvTranspose2d { return ct(x) }
        if let relu = layer as? ReLUModule { return relu(x) }
        return x
    }
    
    public func internalResolution() -> Int {
        return encoder.internalResolution()
    }
}

// MARK: - MonodepthWithEncodingAdaptor

/// Monodepth model wrapper that returns features along with disparity
public class MonodepthWithEncodingAdaptor: Module {
    @ModuleInfo public var monodepth_predictor: MonodepthDensePredictionTransformer
    let returnEncoderFeatures: Bool
    let returnDecoderFeatures: Bool
    let numMonodepthLayers: Int
    let sortingMonodepth: Bool
    
    public init(
        monodepthPredictor: MonodepthDensePredictionTransformer,
        returnEncoderFeatures: Bool = true,
        returnDecoderFeatures: Bool = true,
        numMonodepthLayers: Int = 1,
        sortingMonodepth: Bool = false
    ) {
        self.monodepth_predictor = monodepthPredictor
        self.returnEncoderFeatures = returnEncoderFeatures
        self.returnDecoderFeatures = returnDecoderFeatures
        self.numMonodepthLayers = numMonodepthLayers
        self.sortingMonodepth = sortingMonodepth
        super.init()
    }
    
    public func callAsFunction(_ image: MLXArray) -> MonodepthOutput {
        let inputs = monodepth_predictor.normalizer(image)
        logMemoryIfEnabled("Monodepth:before encoder", prefix: "    ")
        let encoderOutput = monodepth_predictor.encoder(inputs)
        logMemoryIfEnabled("Monodepth:after encoder (SPN)", prefix: "    ")
        
        let numEncoderFeatures = monodepth_predictor.encoder.dims_encoder.count
        let encoderFeatures = Array(encoderOutput[0..<numEncoderFeatures])
        let intermediateFeatures = numEncoderFeatures < encoderOutput.count
            ? Array(encoderOutput[numEncoderFeatures...])
            : []
        
        // Eval encoder features before decoder
        if sharpMLXConfig.aggressiveMemoryManagement {
            for feat in encoderFeatures {
                eval(feat)
            }
            GPU.clearCache()
        }
        logMemoryIfEnabled("Monodepth:after encoder eval+clear", prefix: "    ")
        
        let decoderFeatures = monodepth_predictor.decoder(encoderFeatures)
        evalAndClearIfEnabled(decoderFeatures)
        logMemoryIfEnabled("Monodepth:after decoder", prefix: "    ")
        
        // Apply head
        var disparity = decoderFeatures
        for layer in monodepth_predictor.head {
            disparity = applyHeadLayer(layer, disparity)
        }
        evalAndClearIfEnabled(disparity)
        
        // Sort disparity layers if needed
        if numMonodepthLayers == 2 && sortingMonodepth {
            let firstLayer = disparity.max(axis: -1, keepDims: true)
            let secondLayer = disparity.min(axis: -1, keepDims: true)
            disparity = concatenated([firstLayer, secondLayer], axis: -1)
            evalAndClearIfEnabled(disparity)
        }
        
        var outputFeatures: [MLXArray] = []
        if returnEncoderFeatures {
            outputFeatures.append(contentsOf: encoderFeatures)
        }
        if returnDecoderFeatures {
            outputFeatures.append(decoderFeatures)
        }
        
        return MonodepthOutput(
            disparity: disparity,
            encoderFeatures: encoderFeatures,
            decoderFeatures: decoderFeatures,
            outputFeatures: outputFeatures,
            intermediateFeatures: intermediateFeatures
        )
    }
    
    private func applyHeadLayer(_ layer: Module, _ x: MLXArray) -> MLXArray {
        if let conv = layer as? Conv2d { return conv(x) }
        if let ct = layer as? ConvTranspose2d { return ct(x) }
        if let relu = layer as? ReLUModule { return relu(x) }
        return x
    }
    
    public func getFeatureDims() -> [Int] {
        var dims: [Int] = []
        if returnEncoderFeatures {
            dims.append(contentsOf: monodepth_predictor.encoder.dims_encoder)
        }
        if returnDecoderFeatures {
            dims.append(monodepth_predictor.decoder.dim_out)
        }
        return dims
    }
    
    public func internalResolution() -> Int {
        return monodepth_predictor.internalResolution()
    }
}

// MARK: - Factory Functions

/// Create MonodepthDensePredictionTransformer
public func createMonodepthDPT(
    dimsEncoder: [Int] = [256, 256, 512, 1024, 1024],
    dimsDecoder: [Int] = [256, 256, 256, 256, 256],
    numDisparityChannels: Int = 2,
    usePatchOverlap: Bool = true
) -> MonodepthDensePredictionTransformer {
    let encoder = createSPNEncoder(
        dimsEncoder: dimsEncoder,
        usePatchOverlap: usePatchOverlap
    )
    
    let decoder = MultiresConvDecoder(
        dimsEncoder: dimsEncoder,
        dimsDecoder: dimsDecoder,
        upsamplingMode: "transposed_conv"
    )
    
    return MonodepthDensePredictionTransformer(
        encoder: encoder,
        decoder: decoder,
        lastDims: (32, numDisparityChannels)
    )
}

/// Create MonodepthWithEncodingAdaptor
public func createMonodepthAdaptor(
    monodepthPredictor: MonodepthDensePredictionTransformer,
    returnEncoderFeatures: Bool = true,
    returnDecoderFeatures: Bool = true,
    numMonodepthLayers: Int = 1,
    sortingMonodepth: Bool = false
) -> MonodepthWithEncodingAdaptor {
    return MonodepthWithEncodingAdaptor(
        monodepthPredictor: monodepthPredictor,
        returnEncoderFeatures: returnEncoderFeatures,
        returnDecoderFeatures: returnDecoderFeatures,
        numMonodepthLayers: numMonodepthLayers,
        sortingMonodepth: sortingMonodepth
    )
}
