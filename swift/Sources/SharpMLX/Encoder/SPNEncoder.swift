// Sliding Pyramid Network encoder for Sharp MLX Swift
// Port of sharp_mlx/spn_encoder.py

import Foundation
import MLX
import MLXNN

// MARK: - Split Function

/// Split the input into small patches with sliding window
/// - Parameters:
///   - image: Input image (B, H, W, C) in NHWC format
///   - overlapRatio: Overlap ratio between adjacent patches
///   - patchSize: Size of each patch
/// - Returns: Patches stacked along batch dimension (N*B, patch_size, patch_size, C)
public func split(_ image: MLXArray, overlapRatio: Float = 0.25, patchSize: Int = 384) -> MLXArray {
    let B = image.dim(0)
    let H = image.dim(1)
    let patchStride = Int(Float(patchSize) * (1 - overlapRatio))
    
    let imageSize = H  // Assume square images
    let steps = Int(ceil(Float(imageSize - patchSize) / Float(patchStride))) + 1
    
    var patches: [MLXArray] = []
    for j in 0..<steps {
        let j0 = j * patchStride
        let j1 = j0 + patchSize
        
        for i in 0..<steps {
            let i0 = i * patchStride
            let i1 = i0 + patchSize
            patches.append(image[0..., j0..<j1, i0..<i1, 0...])
        }
    }
    
    // Stack all patches: (num_patches * B, patch_size, patch_size, C)
    return concatenated(patches, axis: 0)
}

// MARK: - Merge Function

/// Merge the patched input into an image with sliding window
/// - Parameters:
///   - imagePatches: Patches (num_patches * B, H, W, C)
///   - batchSize: Original batch size
///   - padding: Overlap padding to crop from edges
/// - Returns: Merged image (B, H_out, W_out, C)
public func merge(_ imagePatches: MLXArray, batchSize: Int, padding: Int = 3) -> MLXArray {
    let steps = Int(sqrt(Double(imagePatches.dim(0) / batchSize)))
    
    var idx = 0
    var outputRows: [MLXArray] = []
    
    for j in 0..<steps {
        var outputRowPatches: [MLXArray] = []
        for i in 0..<steps {
            var output = imagePatches[(batchSize * idx)..<(batchSize * (idx + 1)), 0..., 0..., 0...]
            
            if padding != 0 {
                let H = output.dim(1)
                let W = output.dim(2)
                
                var y0 = 0
                var y1 = H
                var x0 = 0
                var x1 = W
                
                if j != 0 { y0 = padding }
                if i != 0 { x0 = padding }
                if j != steps - 1 { y1 = H - padding }
                if i != steps - 1 { x1 = W - padding }
                
                output = output[0..., y0..<y1, x0..<x1, 0...]
            }
            
            outputRowPatches.append(output)
            idx += 1
        }
        
        let outputRow = concatenated(outputRowPatches, axis: 2)  // Concat along W
        outputRows.append(outputRow)
    }
    
    return concatenated(outputRows, axis: 1)  // Concat along H
}

// MARK: - Interpolate Function

/// Interpolate image tensor to match PyTorch F.interpolate
/// - Parameters:
///   - x: Input (B, H, W, C) in NHWC format
///   - scaleFactor: Scale factor for resizing
///   - mode: Interpolation mode ('bilinear' or 'nearest')
/// - Returns: Resized tensor matching PyTorch F.interpolate(align_corners=False)
public func interpolate(_ x: MLXArray, scaleFactor: Float, mode: String = "bilinear") -> MLXArray {
    let B = x.dim(0)
    let H = x.dim(1)
    let W = x.dim(2)
    let C = x.dim(3)
    let newH = Int(Float(H) * scaleFactor)
    let newW = Int(Float(W) * scaleFactor)
    
    if mode == "nearest" {
        if scaleFactor < 1 {
            let poolSize = Int(1 / scaleFactor)
            var result = x.reshaped(B, newH, poolSize, newW, poolSize, C)
            result = result[0..., 0..., 0, 0..., 0, 0...]
            return result
        } else {
            let factor = Int(scaleFactor)
            var result = repeated(x, count: factor, axis: 1)
            result = repeated(result, count: factor, axis: 2)
            return result
        }
    } else if mode == "bilinear" {
        // Proper bilinear interpolation matching PyTorch align_corners=False
        // Formula: src_coord = (dst_coord + 0.5) / scale_factor - 0.5
        
        // Generate source coordinates
        var ySrcVals: [Float] = []
        var xSrcVals: [Float] = []
        for i in 0..<newH {
            let src = ((Float(i) + 0.5) / scaleFactor - 0.5).clamped(to: 0...Float(H - 1))
            ySrcVals.append(src)
        }
        for i in 0..<newW {
            let src = ((Float(i) + 0.5) / scaleFactor - 0.5).clamped(to: 0...Float(W - 1))
            xSrcVals.append(src)
        }
        
        // Integer and fractional parts
        let y0Vals = ySrcVals.map { Int32(floor($0)) }
        let x0Vals = xSrcVals.map { Int32(floor($0)) }
        let y1Vals = y0Vals.map { min($0 + 1, Int32(H - 1)) }
        let x1Vals = x0Vals.map { min($0 + 1, Int32(W - 1)) }
        
        let fyVals = zip(ySrcVals, y0Vals).map { $0 - Float($1) }
        let fxVals = zip(xSrcVals, x0Vals).map { $0 - Float($1) }
        
        // Create MLX arrays for grid indexing
        let y0 = MLXArray(y0Vals)  // [newH]
        let y1 = MLXArray(y1Vals)
        let x0 = MLXArray(x0Vals)  // [newW]
        let x1 = MLXArray(x1Vals)
        let fy = MLXArray(fyVals).reshaped([newH, 1, 1])  // [newH, 1, 1]
        let fx = MLXArray(fxVals).reshaped([1, newW, 1])  // [1, newW, 1]
        
        // Create 2D index grids [newH, newW]
        let y0Grid = MLX.broadcast(y0.reshaped([newH, 1]), to: [newH, newW])
        let y1Grid = MLX.broadcast(y1.reshaped([newH, 1]), to: [newH, newW])
        let x0Grid = MLX.broadcast(x0.reshaped([1, newW]), to: [newH, newW])
        let x1Grid = MLX.broadcast(x1.reshaped([1, newW]), to: [newH, newW])
        
        // Process batches
        var batchOutputs: [MLXArray] = []
        for b in 0..<B {
            let batchInput = x[b]  // [H, W, C]
            
            // Gather corners using 2D grid indexing
            // Note: MLX Swift uses take() for fancy indexing
            // We need to flatten and use linear indexing
            
            // Convert 2D indices to linear indices: idx = y * W + x
            let y0Lin = y0Grid * Int32(W) + x0Grid  // [newH, newW] - for p00
            let y0LinX1 = y0Grid * Int32(W) + x1Grid  // for p01
            let y1Lin = y1Grid * Int32(W) + x0Grid  // for p10
            let y1LinX1 = y1Grid * Int32(W) + x1Grid  // for p11
            
            // Flatten input for take
            let flatInput = batchInput.reshaped([H * W, C])  // [H*W, C]
            
            // Gather all corners at once
            let p00 = flatInput.take(y0Lin.reshaped([-1]), axis: 0).reshaped([newH, newW, C])
            let p01 = flatInput.take(y0LinX1.reshaped([-1]), axis: 0).reshaped([newH, newW, C])
            let p10 = flatInput.take(y1Lin.reshaped([-1]), axis: 0).reshaped([newH, newW, C])
            let p11 = flatInput.take(y1LinX1.reshaped([-1]), axis: 0).reshaped([newH, newW, C])
            
            // Bilinear interpolation
            let interp = (1 - fy) * (1 - fx) * p00 +
                         (1 - fy) * fx * p01 +
                         fy * (1 - fx) * p10 +
                         fy * fx * p11
            
            batchOutputs.append(interp.expandedDimensions(axis: 0))
        }
        return concatenated(batchOutputs, axis: 0)
    }
    
    return x
}

extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        return max(range.lowerBound, min(range.upperBound, self))
    }
}

// MARK: - SlidingPyramidNetwork

/// Sliding Pyramid Network encoder
/// Creates multi-resolution encodings from Vision Transformers
public class SlidingPyramidNetwork: Module {
    public let dims_encoder: [Int]
    @ModuleInfo public var patch_encoder: VisionTransformer
    @ModuleInfo public var image_encoder: VisionTransformer
    public let use_patch_overlap: Bool
    public let patchSize: Int
    
    @ModuleInfo var upsample_latent0: SequentialModule
    @ModuleInfo var upsample_latent1: SequentialModule
    @ModuleInfo var upsample0: SequentialModule
    @ModuleInfo var upsample1: SequentialModule
    @ModuleInfo var upsample2: SequentialModule
    @ModuleInfo public var upsample_lowres: ConvTranspose2d
    @ModuleInfo var fuse_lowres: Conv2d
    
    public init(
        dimsEncoder: [Int],
        patchEncoder: VisionTransformer,
        imageEncoder: VisionTransformer,
        usePatchOverlap: Bool = true
    ) {
        self.dims_encoder = dimsEncoder
        self.patch_encoder = patchEncoder
        self.image_encoder = imageEncoder
        self.use_patch_overlap = usePatchOverlap
        
        let baseEmbedDim = patchEncoder.embedDim
        let lowresEmbedDim = imageEncoder.embedDim
        self.patchSize = patchEncoder.internalResolution()
        
        // Upsampling blocks for patch encoder features
        self.upsample_latent0 = SlidingPyramidNetwork.createProjectUpsampleBlock(
            dimIn: baseEmbedDim,
            dimOut: dimsEncoder[0],
            upsampleLayers: 3,
            dimIntermediate: dimsEncoder[1]
        )
        self.upsample_latent1 = SlidingPyramidNetwork.createProjectUpsampleBlock(
            dimIn: baseEmbedDim,
            dimOut: dimsEncoder[1],
            upsampleLayers: 2
        )
        self.upsample0 = SlidingPyramidNetwork.createProjectUpsampleBlock(
            dimIn: baseEmbedDim,
            dimOut: dimsEncoder[2],
            upsampleLayers: 1
        )
        self.upsample1 = SlidingPyramidNetwork.createProjectUpsampleBlock(
            dimIn: baseEmbedDim,
            dimOut: dimsEncoder[3],
            upsampleLayers: 1
        )
        self.upsample2 = SlidingPyramidNetwork.createProjectUpsampleBlock(
            dimIn: baseEmbedDim,
            dimOut: dimsEncoder[4],
            upsampleLayers: 1
        )
        
        // Upsampling for image encoder features
        self.upsample_lowres = ConvTranspose2d(
            inChannels: lowresEmbedDim,
            outChannels: dimsEncoder[4],
            kernelSize: 2,
            stride: 2,
            padding: 0,
            bias: true
        )
        
        // Fusion for combining patch and image encoder features
        self.fuse_lowres = Conv2d(
            inputChannels: dimsEncoder[4] + dimsEncoder[4],
            outputChannels: dimsEncoder[4],
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            padding: IntOrPair(0)
        )
        
        super.init()
    }
    
    private static func createProjectUpsampleBlock(
        dimIn: Int,
        dimOut: Int,
        upsampleLayers: Int,
        dimIntermediate: Int? = nil
    ) -> SequentialModule {
        let intermediate = dimIntermediate ?? dimOut
        
        var layers: [Module] = []
        
        // 1x1 projection (bias=False to match PyTorch checkpoint)
        layers.append(Conv2d(inputChannels: dimIn, outputChannels: intermediate, kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0), bias: false))
        
        // Upsampling layers (ConvTranspose2d with kernel=2, stride=2)
        for i in 0..<upsampleLayers {
            let inCh = i == 0 ? intermediate : dimOut
            layers.append(ConvTranspose2d(
                inChannels: inCh,
                outChannels: dimOut,
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: false
            ))
        }
        
        return SequentialModule(layers)
    }
    
    public func createPyramid(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let x0 = x
        let x1 = interpolate(x, scaleFactor: 0.5, mode: "bilinear")
        let x2 = interpolate(x, scaleFactor: 0.25, mode: "bilinear")
        return (x0, x1, x2)
    }
    
    public func internalResolution() -> Int {
        return patchSize * 4  // 384 * 4 = 1536
    }
    
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let batchSize = x.dim(0)
        
        // Step 0: Create 3-level image pyramid
        let (x0, x1, x2) = createPyramid(x)
        
        let x0Patches: MLXArray
        let x1Patches: MLXArray
        let x2Patches: MLXArray
        let padding: Int
        
        if use_patch_overlap {
            // 5x5 @ 384x384 at highest resolution
            x0Patches = split(x0, overlapRatio: 0.25, patchSize: patchSize)
            // 3x3 @ 384x384 at middle resolution
            x1Patches = split(x1, overlapRatio: 0.5, patchSize: patchSize)
            // 1x1 @ 384x384 at lowest resolution
            x2Patches = x2
            padding = 3
        } else {
            x0Patches = split(x0, overlapRatio: 0.0, patchSize: patchSize)
            x1Patches = split(x1, overlapRatio: 0.0, patchSize: patchSize)
            x2Patches = x2
            padding = 0
        }
        
        logMemoryIfEnabled("SPN:after split", prefix: "      ")
        
        // Two processing modes:
        // 1. Batch mode (default): Process all patches at once - original behavior, faster on Mac
        // 2. Sequential mode (aggressiveMemoryManagement): Process one patch at a time to minimize peak memory
        
        let x0Encodings: MLXArray
        let x0IntermediateFeatures: [Int: MLXArray]?
        let x1Encodings: MLXArray
        
        if sharpMLXConfig.aggressiveMemoryManagement {
            // Sequential processing for memory-constrained devices
            (x0Encodings, x0IntermediateFeatures) = processPatchesSequentially(x0Patches, extractIntermediate: true, label: "x0")
            logMemoryIfEnabled("SPN:after x0 patches", prefix: "      ")
            (x1Encodings, _) = processPatchesSequentially(x1Patches, extractIntermediate: false, label: "x1")
        } else {
            // Batch processing - original behavior
            let (enc0, int0) = patch_encoder(x0Patches)
            x0Encodings = enc0
            x0IntermediateFeatures = int0
            let (enc1, _) = patch_encoder(x1Patches)
            x1Encodings = enc1
        }
        
        // Extract intermediate features for latent encodings from x0
        var xLatent0Features: MLXArray?
        var xLatent1Features: MLXArray?
        
        if let ids = patch_encoder.intermediateFeatureIds, ids.count >= 2,
           let intFeatures = x0IntermediateFeatures,
           let feature0 = intFeatures[ids[0]],
           let feature1 = intFeatures[ids[1]] {
            let xLatent0Encodings = patch_encoder.reshapeFeature(feature0)
            xLatent0Features = merge(
                xLatent0Encodings,
                batchSize: batchSize,
                padding: padding
            )
            evalAndClearIfEnabled(xLatent0Features!)
            logMemoryIfEnabled("SPN:after xLatent0 merge", prefix: "      ")
            
            let xLatent1Encodings = patch_encoder.reshapeFeature(feature1)
            xLatent1Features = merge(
                xLatent1Encodings,
                batchSize: batchSize,
                padding: padding
            )
            evalAndClearIfEnabled(xLatent1Features!)
            logMemoryIfEnabled("SPN:after xLatent1 merge", prefix: "      ")
        }
        
        // Merge x0 patches
        var x0Features = merge(x0Encodings, batchSize: batchSize, padding: padding)
        evalAndClearIfEnabled(x0Features)
        
        // Merge x1 patches
        var x1Features = merge(x1Encodings, batchSize: batchSize, padding: 2 * padding)
        evalAndClearIfEnabled(x1Features)
        logMemoryIfEnabled("SPN:after x1 merge", prefix: "      ")
        
        // Process x2 patches (1 patch)
        let (x2Encodings, _) = patch_encoder(x2Patches)
        var x2Features = x2Encodings
        evalAndClearIfEnabled(x2Features)
        
        // Run image encoder on low-res image
        let (xLowresFeatures, _) = image_encoder(x2Patches)
        evalAndClearIfEnabled(xLowresFeatures)
        logMemoryIfEnabled("SPN:after x2 + lowres", prefix: "      ")
        
        // Upsample all feature maps - THIS IS WHERE LARGE TENSORS ARE CREATED
        if var latent0 = xLatent0Features {
            latent0 = upsample_latent0(latent0)
            xLatent0Features = latent0
            evalAndClearIfEnabled(latent0)
            logMemoryIfEnabled("SPN:after upsample_latent0 (1536x1536x256)", prefix: "      ")
        }
        
        if var latent1 = xLatent1Features {
            latent1 = upsample_latent1(latent1)
            xLatent1Features = latent1
            evalAndClearIfEnabled(latent1)
            logMemoryIfEnabled("SPN:after upsample_latent1 (768x768x256)", prefix: "      ")
        }
        
        x0Features = upsample0(x0Features)
        evalAndClearIfEnabled(x0Features)
        logMemoryIfEnabled("SPN:after upsample0", prefix: "      ")
        
        x1Features = upsample1(x1Features)
        evalAndClearIfEnabled(x1Features)
        
        x2Features = upsample2(x2Features)
        evalAndClearIfEnabled(x2Features)
        
        let lowresUpsampled = upsample_lowres(xLowresFeatures)
        let fusedLowres = fuse_lowres(concatenated([x2Features, lowresUpsampled], axis: -1))
        evalAndClearIfEnabled(fusedLowres)
        logMemoryIfEnabled("SPN:after all upsamples", prefix: "      ")
        
        return [
            xLatent0Features ?? MLXArray.zeros([1]),
            xLatent1Features ?? MLXArray.zeros([1]),
            x0Features,
            x1Features,
            fusedLowres
        ]
    }
    
    /// Process patches one at a time to minimize peak memory usage
    /// Used when aggressiveMemoryManagement is enabled
    private func processPatchesSequentially(_ patches: MLXArray, extractIntermediate: Bool, label: String) -> (MLXArray, [Int: MLXArray]?) {
        let numPatches = patches.dim(0)
        var encodingsList: [MLXArray] = []
        var intermediateFeatures: [Int: [MLXArray]] = [:]
        
        for i in 0..<numPatches {
            let patch = patches[i..<(i+1), 0..., 0..., 0...]
            
            let (encodings, intFeatures) = patch_encoder(patch)
            eval(encodings)
            encodingsList.append(encodings)
            
            if extractIntermediate, let ids = patch_encoder.intermediateFeatureIds {
                for id in ids {
                    if let feat = intFeatures[id] {
                        eval(feat)
                        if intermediateFeatures[id] == nil {
                            intermediateFeatures[id] = []
                        }
                        intermediateFeatures[id]!.append(feat)
                    }
                }
            }
            
            GPU.clearCache()
            
            if (i + 1) % 5 == 0 || i == numPatches - 1 {
                logMemoryIfEnabled("SPN:\(label) patch \(i+1)/\(numPatches)", prefix: "      ")
            }
        }
        
        let combined = concatenated(encodingsList, axis: 0)
        eval(combined)
        encodingsList.removeAll()
        GPU.clearCache()
        
        if extractIntermediate {
            var combinedIntermediate: [Int: MLXArray] = [:]
            for (id, feats) in intermediateFeatures {
                let concat = concatenated(feats, axis: 0)
                eval(concat)
                combinedIntermediate[id] = concat
            }
            intermediateFeatures.removeAll()
            GPU.clearCache()
            return (combined, combinedIntermediate)
        }
        
        return (combined, nil)
    }
}

// MARK: - Factory Function

/// Create Sliding Pyramid Network encoder with DINOv2-L backbones
public func createSPNEncoder(
    dimsEncoder: [Int] = [256, 256, 512, 1024, 1024],
    usePatchOverlap: Bool = true
) -> SlidingPyramidNetwork {
    // Create patch encoder with intermediate feature extraction
    // DINOv2-L has 24 blocks, extract features at layers 5, 11, 17, 23
    let patchEncoder = createDINOv2ViTLarge(
        imgSize: 384,
        intermediateFeatureIds: [5, 11, 17, 23]
    )
    
    // Create image encoder (no intermediate features needed)
    let imageEncoder = createDINOv2ViTLarge(
        imgSize: 384,
        intermediateFeatureIds: []
    )
    
    return SlidingPyramidNetwork(
        dimsEncoder: dimsEncoder,
        patchEncoder: patchEncoder,
        imageEncoder: imageEncoder,
        usePatchOverlap: usePatchOverlap
    )
}
