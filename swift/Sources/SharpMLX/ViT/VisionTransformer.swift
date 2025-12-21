// Vision Transformer implementation for Sharp MLX Swift
// Port of sharp_mlx/vit.py
// DINOv2-L/16 architecture with 24 transformer blocks, 1024 embed dim, 16 heads

import Foundation
import MLX
import MLXNN

// MARK: - PatchEmbed

/// 2D Image to Patch Embedding using Conv2d
public class PatchEmbed: Module {
    let imgSize: Int
    let patchSize: Int
    let gridSize: (Int, Int)
    let numPatches: Int
    
    @ModuleInfo var proj: Conv2d
    
    public init(imgSize: Int = 384, patchSize: Int = 16, inChans: Int = 3, embedDim: Int = 1024) {
        self.imgSize = imgSize
        self.patchSize = patchSize
        self.gridSize = (imgSize / patchSize, imgSize / patchSize)
        self.numPatches = gridSize.0 * gridSize.1
        
        // Conv2d with kernel_size=patch_size, stride=patch_size
        self.proj = Conv2d(inputChannels: inChans, outputChannels: embedDim, kernelSize: .init(patchSize), stride: .init(patchSize))
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, H, W, C) - NHWC format
        let B = x.dim(0)
        
        // Apply projection: (B, H, W, C) -> (B, H/P, W/P, D)
        var out = proj(x)
        
        // Reshape to sequence: (B, H/P, W/P, D) -> (B, N, D) where N = (H/P) * (W/P)
        let Hp = out.dim(1)
        let Wp = out.dim(2)
        let D = out.dim(3)
        out = out.reshaped(B, Hp * Wp, D)
        
        return out
    }
}

// MARK: - Attention

/// Multi-head self attention
public class Attention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    
    public init(dim: Int, numHeads: Int = 8, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        
        // Combined QKV projection
        self.qkv = Linear(dim, dim * 3, bias: qkvBias)
        self.proj = Linear(dim, dim)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let N = x.dim(1)
        let C = x.dim(2)
        
        // QKV projection: (B, N, C) -> (B, N, 3*C)
        var qkvOut = qkv(x)
        
        // Reshape to (B, N, 3, num_heads, head_dim)
        qkvOut = qkvOut.reshaped(B, N, 3, numHeads, headDim)
        
        // Transpose to (3, B, num_heads, N, head_dim)
        qkvOut = qkvOut.transposed(2, 0, 3, 1, 4)
        
        // Split into q, k, v
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]
        
        // Scaled dot-product attention
        var attn = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis: -1)
        
        // Apply attention to values
        var out = MLX.matmul(attn, v)
        
        // Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        out = out.transposed(0, 2, 1, 3).reshaped(B, N, C)
        
        // Output projection
        out = proj(out)
        
        return out
    }
}

// MARK: - MLP

/// MLP with GELU activation
public class MLPModule: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    
    public init(inFeatures: Int, hiddenFeatures: Int? = nil, outFeatures: Int? = nil) {
        let outDim = outFeatures ?? inFeatures
        let hiddenDim = hiddenFeatures ?? (inFeatures * 4)
        
        self.fc1 = Linear(inFeatures, hiddenDim)
        self.fc2 = Linear(hiddenDim, outDim)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = fc1(x)
        out = gelu(out)
        out = fc2(out)
        return out
    }
}

// MARK: - LayerScale

/// Layer Scale from CaiT/DeiT-III
public class LayerScale: Module {
    var gamma: MLXArray
    
    public init(dim: Int, initValues: Float = 1e-5) {
        self.gamma = MLXArray.ones([dim]) * initValues
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * gamma
    }
}

// MARK: - Block

/// Transformer block with pre-norm and LayerScale
public class TransformerBlock: Module {
    @ModuleInfo public var norm1: LayerNorm
    @ModuleInfo public var attn: Attention
    @ModuleInfo public var ls1: LayerScale
    @ModuleInfo public var norm2: LayerNorm
    @ModuleInfo public var mlp: MLPModule
    @ModuleInfo public var ls2: LayerScale
    
    public init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0, qkvBias: Bool = true, initValues: Float = 1e-5) {
        self.norm1 = LayerNorm(dimensions: dim, eps: 1e-6)
        self.attn = Attention(dim: dim, numHeads: numHeads, qkvBias: qkvBias)
        self.ls1 = LayerScale(dim: dim, initValues: initValues)
        
        self.norm2 = LayerNorm(dimensions: dim, eps: 1e-6)
        let mlpHiddenDim = Int(Float(dim) * mlpRatio)
        self.mlp = MLPModule(inFeatures: dim, hiddenFeatures: mlpHiddenDim)
        self.ls2 = LayerScale(dim: dim, initValues: initValues)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x + ls1(attn(norm1(x)))
        out = out + ls2(mlp(norm2(out)))
        return out
    }
}

// MARK: - VisionTransformer

/// Vision Transformer for DINOv2
/// Supports extraction of intermediate features for multi-scale processing
public class VisionTransformer: Module {
    public let imgSize: Int
    public let patchSizeVal: Int
    public let embedDim: Int
    public let depth: Int
    public var intermediateFeatureIds: [Int]?
    
    @ModuleInfo public var patch_embed: PatchEmbed
    public var cls_token: MLXArray
    public var pos_embed: MLXArray
    @ModuleInfo public var blocks: [TransformerBlock]
    @ModuleInfo public var norm: LayerNorm
    
    public init(
        imgSize: Int = 384,
        patchSize: Int = 16,
        inChans: Int = 3,
        embedDim: Int = 1024,
        depth: Int = 24,
        numHeads: Int = 16,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true,
        initValues: Float = 1e-5,
        intermediateFeatureIds: [Int]? = nil
    ) {
        self.imgSize = imgSize
        self.patchSizeVal = patchSize
        self.embedDim = embedDim
        self.depth = depth
        self.intermediateFeatureIds = intermediateFeatureIds
        
        // Patch embedding
        // Compute numPatches first before assigning to @ModuleInfo property
        let gridSize = imgSize / patchSize
        let numPatches = gridSize * gridSize
        
        self.patch_embed = PatchEmbed(
            imgSize: imgSize,
            patchSize: patchSize,
            inChans: inChans,
            embedDim: embedDim
        )
        
        // Class token
        self.cls_token = MLXArray.zeros([1, 1, embedDim])
        
        // Position embedding: class token + patches
        self.pos_embed = MLXArray.zeros([1, numPatches + 1, embedDim])
        
        // Transformer blocks
        var blocksList: [TransformerBlock] = []
        for _ in 0..<depth {
            blocksList.append(TransformerBlock(
                dim: embedDim,
                numHeads: numHeads,
                mlpRatio: mlpRatio,
                qkvBias: qkvBias,
                initValues: initValues
            ))
        }
        self.blocks = blocksList
        
        // Final norm
        self.norm = LayerNorm(dimensions: embedDim, eps: 1e-6)
        
        super.init()
    }
    
    /// Discard class token and reshape 1D feature map to 2D grid (NHWC)
    public func reshapeFeature(_ embeddings: MLXArray) -> MLXArray {
        let B = embeddings.dim(0)
        let seqLen = embeddings.dim(1)
        let C = embeddings.dim(2)
        
        // Remove class token (first token)
        let noClsToken = embeddings[0..., 1..., 0...]
        
        // Calculate grid size
        let gridSize = Int(sqrt(Double(seqLen - 1)))
        
        // Reshape: (B, N, C) -> (B, H, W, C) - NHWC format
        return noClsToken.reshaped(B, gridSize, gridSize, C)
    }
    
    /// Forward pass with intermediate feature extraction
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, [Int: MLXArray]) {
        let B = x.dim(0)
        
        // Patch embedding
        var out = patch_embed(x)
        
        // Prepend class token
        let clsTokens = MLX.broadcast(cls_token, to: [B, 1, embedDim])
        out = concatenated([clsTokens, out], axis: 1)
        
        // Add position embedding
        out = out + pos_embed
        
        // Transformer blocks with intermediate feature extraction
        var intermediateFeatures: [Int: MLXArray] = [:]
        for (idx, block) in blocks.enumerated() {
            out = block(out)
            if let ids = intermediateFeatureIds, ids.contains(idx) {
                intermediateFeatures[idx] = out
            }
        }
        
        // Final norm
        out = norm(out)
        
        // Reshape to 2D grid (NHWC)
        out = reshapeFeature(out)
        
        return (out, intermediateFeatures)
    }
    
    public func internalResolution() -> Int {
        return imgSize
    }
}

// MARK: - Factory Function

/// Create DINOv2 ViT-Large/16 model
public func createDINOv2ViTLarge(
    imgSize: Int = 384,
    intermediateFeatureIds: [Int] = []
) -> VisionTransformer {
    return VisionTransformer(
        imgSize: imgSize,
        patchSize: 16,
        inChans: 3,
        embedDim: 1024,
        depth: 24,
        numHeads: 16,
        mlpRatio: 4.0,
        qkvBias: true,
        initValues: 1e-5,
        intermediateFeatureIds: intermediateFeatureIds.isEmpty ? nil : intermediateFeatureIds
    )
}
