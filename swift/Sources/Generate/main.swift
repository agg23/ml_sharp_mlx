// Sharp MLX Swift CLI
// Generate executable for Sharp 3D Gaussian Splatting inference

import Foundation
import AppKit
import CoreGraphics
import MLX
import MLXNN
import MLXFast
import ArgumentParser
import SharpMLX

// MARK: - Weight Key Remapping

/// Remap Swift MLX key to PyTorch checkpoint key format
/// Python load_weights.py equivalent logic
func mapMlxKeyToPtKey(_ mlxKey: String) -> String {
    var ptKey = mlxKey
    
    // Replace .residual.layers.N with .residual.N
    if let range = ptKey.range(of: #"\.residual\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".residual.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".residual.\(num)")
    }
    
    // Handle multiple occurrences for residual
    while let range = ptKey.range(of: #"\.residual\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".residual.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".residual.\(num)")
    }
    
    // Replace .texture_head.layers.N with .texture_head.N
    while let range = ptKey.range(of: #"\.texture_head\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".texture_head.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".texture_head.\(num)")
    }
    
    // Replace .geometry_head.layers.N with .geometry_head.N
    while let range = ptKey.range(of: #"\.geometry_head\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".geometry_head.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".geometry_head.\(num)")
    }
    
    // Replace .head.layers.N with .head.N for monodepth head
    while let range = ptKey.range(of: #"\.head\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".head.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".head.\(num)")
    }
    
    // Replace .blocks.layers.N with .blocks.N for transformer blocks
    while let range = ptKey.range(of: #"\.blocks\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".blocks.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".blocks.\(num)")
    }
    
    // Replace upsamples with numbered indices (upsample0.layers.N -> upsample0.N)
    let upsamplePattern = #"(upsample\w*)\.layers\.(\d+)"#
    while let regex = try? NSRegularExpression(pattern: upsamplePattern),
          let match = regex.firstMatch(in: ptKey, range: NSRange(ptKey.startIndex..., in: ptKey)) {
        let range0 = Range(match.range(at: 1), in: ptKey)!
        let range1 = Range(match.range(at: 2), in: ptKey)!
        let fullRange = Range(match.range, in: ptKey)!
        let name = String(ptKey[range0])
        let num = String(ptKey[range1])
        ptKey = ptKey.replacingCharacters(in: fullRange, with: "\(name).\(num)")
    }
    
    // Replace .fusions.layers.N with .fusions.N
    while let range = ptKey.range(of: #"\.fusions\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".fusions.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".fusions.\(num)")
    }
    
    // Replace .convs.layers.N with .convs.N
    while let range = ptKey.range(of: #"\.convs\.layers\.(\d+)"#, options: .regularExpression) {
        let match = String(ptKey[range])
        let num = match.replacingOccurrences(of: ".convs.layers.", with: "")
        ptKey = ptKey.replacingCharacters(in: range, with: ".convs.\(num)")
    }
    
    // Remove .norm. for PyTorchGroupNorm wrapper (e.g., .norm.weight -> .weight)
    if ptKey.hasSuffix(".norm.weight") {
        ptKey = String(ptKey.dropLast(12)) + ".weight"
    } else if ptKey.hasSuffix(".norm.bias") {
        ptKey = String(ptKey.dropLast(10)) + ".bias"
    }
    
    return ptKey
}

/// Load weights with key remapping
func loadWeightsWithRemapping(model: Module, weightsPath: String) throws -> (loaded: Int, missing: Int, unexpected: Int) {
    let weightsURL = URL(fileURLWithPath: weightsPath)
    let ptWeights = try loadArrays(url: weightsURL)
    
    // Get model parameters
    let modelParams = model.parameters()
    let flatParams = modelParams.flattened()
    
    var mappedWeights: [String: MLXArray] = [:]
    var loadedCount = 0
    var unusedPtKeys = Set(ptWeights.keys)
    var missingKeys: [String] = []
    
    // For each MLX model parameter, find matching PyTorch weight
    for (mlxKey, mlxValue) in flatParams {
        let ptKey = mapMlxKeyToPtKey(mlxKey)
        
        if let ptWeight = ptWeights[ptKey] {
            if ptWeight.shape == mlxValue.shape {
                mappedWeights[mlxKey] = ptWeight
                loadedCount += 1
                unusedPtKeys.remove(ptKey)
            } else {
                print("  Shape mismatch: \(mlxKey) expected \(mlxValue.shape), got \(ptWeight.shape)")
            }
        } else if let ptWeight = ptWeights[mlxKey] {
            // Try direct match
            if ptWeight.shape == mlxValue.shape {
                mappedWeights[mlxKey] = ptWeight
                loadedCount += 1
                unusedPtKeys.remove(mlxKey)
            }
        } else {
            missingKeys.append(mlxKey)
        }
    }
    
    // Apply weights
    if !mappedWeights.isEmpty {
        let nested = ModuleParameters.unflattened(mappedWeights)
        try model.update(parameters: nested, verify: .none)
    }
    
    // Print some diagnostics
    if !missingKeys.isEmpty && missingKeys.count <= 20 {
        print("  Missing \(missingKeys.count) weights:")
        for key in missingKeys.prefix(10) {
            print("    \(key) -> \(mapMlxKeyToPtKey(key))")
        }
    }
    
    return (loadedCount, missingKeys.count, unusedPtKeys.count)
}

struct SharpCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Sharp 3D Gaussian Splatting inference using MLX",
        discussion: """
        Generates 3D Gaussian splats from a single image.
        
        Example:
          generate --input image.jpg --weights checkpoints/sharp.safetensors
        """
    )
    
    @Option(name: .shortAndLong, help: "Input image path")
    var input: String
    
    @Option(name: .shortAndLong, help: "Output PLY path")
    var output: String = "output.ply"
    
    @Option(name: .shortAndLong, help: "Model weights path")
    var weights: String = "checkpoints/sharp.safetensors"
    
    // Internal resolution is fixed at 1536x1536 (4x ViT patch size of 384)
    private let internalResolution = 1536
    
    func run() throws {
        print("Sharp MLX Swift Inference")
        print(String(repeating: "=", count: 24))
        
        // Load image
        print("\nLoading image: \(input)")
        let (image, fPx, origHeight, origWidth) = try loadImage(path: input, targetSize: internalResolution)
        print("  Resized to: \(internalResolution)x\(internalResolution)")
        
        // Create model
        print("\nCreating model...")
        let model = createPredictor()
        
        // Load weights with key remapping
        print("Loading weights from: \(weights)")
        let (loaded, missing, unused) = try loadWeightsWithRemapping(model: model, weightsPath: weights)
        print("  Loaded \(loaded) parameters, missing \(missing), unused \(unused)")
        
        // Compute disparity factor
        let disparityFactor = MLXArray([fPx / Float(origWidth)])
        
        // Run inference
        print("\nRunning inference...")
        let startTime = Date()
        
        let gaussians = model(image, disparityFactor: disparityFactor)
        eval(gaussians.means, gaussians.scales, gaussians.colors, gaussians.quaternions, gaussians.opacities)
        
        let elapsed = Date().timeIntervalSince(startTime)
        print("  Gaussians: \(gaussians.count)")
        print("  Inference time: \(String(format: "%.2f", elapsed))s")
        
        // Unproject from NDC to metric space (parallel processing)
        let unprojectStart = Date()
        let (meansMetric, scalesMetric, quatsMetric) = unprojectGaussians(
            gaussians,
            fPx: fPx,
            origWidth: origWidth,
            origHeight: origHeight,
            internalWidth: internalResolution,
            internalHeight: internalResolution
        )
        eval(meansMetric, scalesMetric, quatsMetric)
        let unprojectElapsed = Date().timeIntervalSince(unprojectStart)
        print("  Unprojection time: \(String(format: "%.2f", unprojectElapsed))s")
        
        // Create metric-space Gaussians for PLY export
        let gaussiansMetric = Gaussians3D(
            means: meansMetric,
            scales: scalesMetric,
            quaternions: quatsMetric,
            colors: gaussians.colors,
            opacities: gaussians.opacities
        )
        
        // Save PLY
        print("\nSaving to: \(output)")
        try savePLY(gaussians: gaussiansMetric, fPx: fPx, imageWidth: origWidth, imageHeight: origHeight, to: output)
        
        print("Done!")
    }
}

// MARK: - Image Loading

func loadImage(path: String, targetSize: Int = 1536) throws -> (MLXArray, Float, Int, Int) {
    guard let nsImage = NSImage(contentsOfFile: path) else {
        throw NSError(domain: "SharpMLX", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load image: \(path)"])
    }
    
    guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        throw NSError(domain: "SharpMLX", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to convert NSImage to CGImage"])
    }
    
    let origWidth = cgImage.width
    let origHeight = cgImage.height
    
    // Estimate focal length (1.2x max dimension, common for phone cameras)
    let fPx = Float(max(origWidth, origHeight)) * 1.2
    
    // Load original image pixels first (no resize)
    // Use noneSkipLast (RGBX) to avoid premultiplied alpha corruption
    let bitsPerComponent = 8
    let bytesPerRow = origWidth * 4
    let origContext = CGContext(
        data: nil,
        width: origWidth,
        height: origHeight,
        bitsPerComponent: bitsPerComponent,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    )!
    
    origContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: origWidth, height: origHeight))
    
    guard let origData = origContext.makeImage()?.dataProvider?.data else {
        throw NSError(domain: "SharpMLX", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to get image data"])
    }
    let origPtr = CFDataGetBytePtr(origData)!
    
    // Convert to float array [H, W, 3]
    var origPixels: [Float] = []
    for y in 0..<origHeight {
        for x in 0..<origWidth {
            let offset = (y * origWidth + x) * 4
            origPixels.append(Float(origPtr[offset]) / 255.0)
            origPixels.append(Float(origPtr[offset + 1]) / 255.0)
            origPixels.append(Float(origPtr[offset + 2]) / 255.0)
        }
    }
    
    // Bilinear interpolation with align_corners=True (matching PyTorch)
    var resizedPixels: [Float] = []
    
    for y in 0..<targetSize {
        for x in 0..<targetSize {
            // align_corners=True: map corners exactly
            // src_x = x * (src_W - 1) / (dst_W - 1)
            let srcX = Float(x) * Float(origWidth - 1) / Float(targetSize - 1)
            let srcY = Float(y) * Float(origHeight - 1) / Float(targetSize - 1)
            
            let x0 = Int(srcX)
            let y0 = Int(srcY)
            let x1 = min(x0 + 1, origWidth - 1)
            let y1 = min(y0 + 1, origHeight - 1)
            
            let xFrac = srcX - Float(x0)
            let yFrac = srcY - Float(y0)
            
            for c in 0..<3 {
                let idx00 = (y0 * origWidth + x0) * 3 + c
                let idx01 = (y0 * origWidth + x1) * 3 + c
                let idx10 = (y1 * origWidth + x0) * 3 + c
                let idx11 = (y1 * origWidth + x1) * 3 + c
                
                let v00 = origPixels[idx00]
                let v01 = origPixels[idx01]
                let v10 = origPixels[idx10]
                let v11 = origPixels[idx11]
                
                // Bilinear interpolation
                let v = v00 * (1 - xFrac) * (1 - yFrac)
                      + v01 * xFrac * (1 - yFrac)
                      + v10 * (1 - xFrac) * yFrac
                      + v11 * xFrac * yFrac
                
                resizedPixels.append(v)
            }
        }
    }
    
    let image = MLXArray(resizedPixels, [1, targetSize, targetSize, 3])
    
    return (image, fPx, origHeight, origWidth)
}

// MARK: - Unprojection (NDC to Metric Space)

import Accelerate

/// Single 3x3 matrix SVD decomposition using Accelerate
/// Returns (U, S, Vt) where U is rotation, S is singular values, Vt is V transpose
/// Note: LAPACK returns column-major matrices, so we transpose U to row-major
func svd3x3(_ matrix: [Float]) -> (U: [Float], S: [Float], Vt: [Float]) {
    var A = matrix  // Copy input
    var UColMajor = [Float](repeating: 0, count: 9)
    var S = [Float](repeating: 0, count: 3)
    var VtColMajor = [Float](repeating: 0, count: 9)
    
    var jobU: Int8 = 65 // 'A' - all columns of U
    var jobVt: Int8 = 65 // 'A' - all rows of Vt
    var m: __CLPK_integer = 3
    var n: __CLPK_integer = 3
    var lda: __CLPK_integer = 3
    var ldu: __CLPK_integer = 3
    var ldvt: __CLPK_integer = 3
    var info: __CLPK_integer = 0
    
    var workSize: __CLPK_integer = -1
    var workQuery: Float = 0
    
    // Query optimal work size
    sgesvd_(&jobU, &jobVt, &m, &n, &A, &lda, &S, &UColMajor, &ldu, &VtColMajor, &ldvt, &workQuery, &workSize, &info)
    
    workSize = __CLPK_integer(workQuery)
    var work = [Float](repeating: 0, count: Int(workSize))
    
    // Compute SVD
    A = matrix  // Reset A
    sgesvd_(&jobU, &jobVt, &m, &n, &A, &lda, &S, &UColMajor, &ldu, &VtColMajor, &ldvt, &work, &workSize, &info)
    
    // Transpose U from column-major to row-major format
    // Column-major: U[i,j] = UColMajor[i + j*3]
    // Row-major: U[i,j] = URowMajor[i*3 + j]
    let U: [Float] = [
        UColMajor[0], UColMajor[3], UColMajor[6],  // Row 0
        UColMajor[1], UColMajor[4], UColMajor[7],  // Row 1
        UColMajor[2], UColMajor[5], UColMajor[8]   // Row 2
    ]
    
    // Same for Vt
    let Vt: [Float] = [
        VtColMajor[0], VtColMajor[3], VtColMajor[6],
        VtColMajor[1], VtColMajor[4], VtColMajor[7],
        VtColMajor[2], VtColMajor[5], VtColMajor[8]
    ]
    
    return (U, S, Vt)
}

/// Compute determinant of 3x3 matrix
func det3x3(_ m: [Float]) -> Float {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Convert 3x3 rotation matrix to quaternion (wxyz)
func rotationToQuaternion(_ r: [Float]) -> [Float] {
    let trace = r[0] + r[4] + r[8]
    var w: Float, x: Float, y: Float, z: Float
    
    if trace > 0 {
        let s = 0.5 / sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[7] - r[5]) * s  // m21 - m12
        y = (r[2] - r[6]) * s  // m02 - m20
        z = (r[3] - r[1]) * s  // m10 - m01
    } else if r[0] > r[4] && r[0] > r[8] {
        let s = 2.0 * sqrt(1.0 + r[0] - r[4] - r[8])
        w = (r[7] - r[5]) / s
        x = 0.25 * s
        y = (r[1] + r[3]) / s
        z = (r[2] + r[6]) / s
    } else if r[4] > r[8] {
        let s = 2.0 * sqrt(1.0 + r[4] - r[0] - r[8])
        w = (r[2] - r[6]) / s
        x = (r[1] + r[3]) / s
        y = 0.25 * s
        z = (r[5] + r[7]) / s
    } else {
        let s = 2.0 * sqrt(1.0 + r[8] - r[0] - r[4])
        w = (r[3] - r[1]) / s
        x = (r[2] + r[6]) / s
        y = (r[5] + r[7]) / s
        z = 0.25 * s
    }
    
    // Normalize
    let norm = sqrt(w*w + x*x + y*y + z*z)
    return [w/norm, x/norm, y/norm, z/norm]
}

/// Convert quaternion (wxyz) to 3x3 rotation matrix
func quaternionToRotation(_ q: [Float]) -> [Float] {
    let w = q[0], x = q[1], y = q[2], z = q[3]
    let norm = sqrt(w*w + x*x + y*y + z*z)
    let wn = w/norm, xn = x/norm, yn = y/norm, zn = z/norm
    
    let xx = xn*xn, yy = yn*yn, zz = zn*zn
    let xy = xn*yn, xz = xn*zn, yz = yn*zn
    let wx = wn*xn, wy = wn*yn, wz = wn*zn
    
    return [
        1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
        2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)
    ]
}

/// Multiply 3x3 matrices: C = A @ B
func matmul3x3(_ A: [Float], _ B: [Float]) -> [Float] {
    var C = [Float](repeating: 0, count: 9)
    for i in 0..<3 {
        for j in 0..<3 {
            for k in 0..<3 {
                C[i*3+j] += A[i*3+k] * B[k*3+j]
            }
        }
    }
    return C
}

/// Transpose 3x3 matrix
func transpose3x3(_ m: [Float]) -> [Float] {
    return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Unproject Gaussians from NDC space to metric/world coordinates
/// Uses proper SVD-based covariance matrix transformation matching Python exactly
func unprojectGaussians(
    _ gaussians: Gaussians3D,
    fPx: Float,
    origWidth: Int,
    origHeight: Int,
    internalWidth: Int = 1536,
    internalHeight: Int = 1536
) -> (means: MLXArray, scales: MLXArray, quaternions: MLXArray) {
    // Build unprojection transform matrix
    let fPxScaled = fPx * Float(internalWidth) / Float(origWidth)
    
    // NDC to camera transform scale factors
    let scaleX = Float(internalWidth) / (2.0 * fPxScaled)
    let scaleY = Float(internalHeight) / (2.0 * fPxScaled)
    
    // Transform matrix (diagonal scaling)
    let T: [Float] = [scaleX, 0, 0, 0, scaleY, 0, 0, 0, 1]
    let Tt = transpose3x3(T)
    
    // Transform means: scale x, y by factors, z unchanged
    let xNDC = gaussians.means[0..., 0..., 0]
    let yNDC = gaussians.means[0..., 0..., 1]
    let zNDC = gaussians.means[0..., 0..., 2]
    
    let xWorld = xNDC * scaleX
    let yWorld = yNDC * scaleY
    let zWorld = zNDC
    let meansWorld = stacked([xWorld, yWorld, zWorld], axis: -1)
    
    // For scales and quaternions, we need to transform covariance matrices
    // Extract quaternions and scales to CPU for processing
    eval(gaussians.quaternions, gaussians.scales)
    let quatsArray = gaussians.quaternions[0].asArray(Float.self)
    let scalesArray = gaussians.scales[0].asArray(Float.self)
    let N = gaussians.count
    
    // Pre-allocate output arrays (thread-safe with fixed indices)
    var newQuats = [Float](repeating: 0, count: N * 4)
    var newScales = [Float](repeating: 0, count: N * 3)
    
    // Process all Gaussians in parallel using GCD
    // This parallelizes across all CPU cores instead of sequential processing
    DispatchQueue.concurrentPerform(iterations: N) { i in
        // Get quaternion (wxyz) and scales for this Gaussian
        let quat = [quatsArray[i*4], quatsArray[i*4+1], quatsArray[i*4+2], quatsArray[i*4+3]]
        let scales = [scalesArray[i*3], scalesArray[i*3+1], scalesArray[i*3+2]]
        
        // Build covariance matrix: C = R @ D² @ R.T
        let R = quaternionToRotation(quat)
        let D2: [Float] = [scales[0]*scales[0], 0, 0, 0, scales[1]*scales[1], 0, 0, 0, scales[2]*scales[2]]
        let Rt = transpose3x3(R)
        let RD2 = matmul3x3(R, D2)
        let C = matmul3x3(RD2, Rt)
        
        // Transform covariance: C' = T @ C @ T.T
        let TC = matmul3x3(T, C)
        let Cp = matmul3x3(TC, Tt)
        
        // SVD decomposition of C'
        let (U, S, _) = svd3x3(Cp)
        
        // Check for reflection and fix
        var Ufixed = U
        if det3x3(U) < 0 {
            // Flip last column
            Ufixed[2] = -U[2]
            Ufixed[5] = -U[5]
            Ufixed[8] = -U[8]
        }
        
        // Convert rotation matrix to quaternion
        let newQ = rotationToQuaternion(Ufixed)
        
        // Write to pre-allocated arrays (thread-safe: each index is unique)
        newQuats[i*4] = newQ[0]
        newQuats[i*4+1] = newQ[1]
        newQuats[i*4+2] = newQ[2]
        newQuats[i*4+3] = newQ[3]
        
        // Singular values are sqrt of eigenvalues
        newScales[i*3] = sqrt(S[0])
        newScales[i*3+1] = sqrt(S[1])
        newScales[i*3+2] = sqrt(S[2])
    }
    
    // Convert back to MLXArray
    let quaternionsWorld = MLXArray(newQuats).reshaped([1, N, 4])
    let scalesWorld = MLXArray(newScales).reshaped([1, N, 3])
    
    return (meansWorld, scalesWorld, quaternionsWorld)
}

// MARK: - PLY Saving

/// Convert linearRGB to sRGB (gamma correction)
func linearRGBToSRGB(_ linear: Float) -> Float {
    if linear <= 0.0031308 {
        return linear * 12.92
    } else {
        return 1.055 * pow(linear, 1.0 / 2.4) - 0.055
    }
}

/// Convert sRGB to degree-0 spherical harmonics
/// SH_DC = (sRGB - 0.5) / sqrt(1 / 4π)
func rgbToSphericalHarmonics(_ srgb: Float) -> Float {
    let coeffDegree0: Float = sqrt(1.0 / (4.0 * Float.pi))
    return (srgb - 0.5) / coeffDegree0
}

/// Inverse sigmoid (logit function)
func inverseSigmoid(_ x: Float) -> Float {
    let clamped = max(1e-6, min(1.0 - 1e-6, x))
    return log(clamped / (1.0 - clamped))
}

func savePLY(gaussians: Gaussians3D, fPx: Float, imageWidth: Int, imageHeight: Int, to path: String) throws {
    let N = gaussians.count
    
    eval(gaussians.means, gaussians.scales, gaussians.quaternions, gaussians.colors, gaussians.opacities)
    
    // Convert to arrays
    let means = gaussians.means[0].asArray(Float.self)
    let scales = gaussians.scales[0].asArray(Float.self)
    let quats = gaussians.quaternions[0].asArray(Float.self)
    let colors = gaussians.colors[0].asArray(Float.self)
    let opacities = gaussians.opacities[0].asArray(Float.self)
    
    // Build PLY header - EXACT same order as Sharp's save_ply
    let header = """
    ply
    format binary_little_endian 1.0
    element vertex \(N)
    property float x
    property float y
    property float z
    property float f_dc_0
    property float f_dc_1
    property float f_dc_2
    property float opacity
    property float scale_0
    property float scale_1
    property float scale_2
    property float rot_0
    property float rot_1
    property float rot_2
    property float rot_3
    element extrinsic 16
    property float extrinsic
    element intrinsic 9
    property float intrinsic
    element image_size 2
    property uint image_size
    element frame 2
    property int frame
    element disparity 2
    property float disparity
    element color_space 1
    property uchar color_space
    element version 3
    property uchar version
    end_header
    
    """
    
    var data = Data()
    data.append(header.data(using: .utf8)!)
    
    // Write binary vertex data - matching Sharp's order:
    // xyz, f_dc_0-2, opacity, scale_0-2, rot_0-3
    for i in 0..<N {
        // Position (x, y, z)
        var x = means[i * 3]
        var y = means[i * 3 + 1]
        var z = means[i * 3 + 2]
        
        // Colors: linearRGB -> sRGB -> spherical harmonics
        let linearR = colors[i * 3]
        let linearG = colors[i * 3 + 1]
        let linearB = colors[i * 3 + 2]
        let srgbR = linearRGBToSRGB(linearR)
        let srgbG = linearRGBToSRGB(linearG)
        let srgbB = linearRGBToSRGB(linearB)
        var c0 = rgbToSphericalHarmonics(srgbR)
        var c1 = rgbToSphericalHarmonics(srgbG)
        var c2 = rgbToSphericalHarmonics(srgbB)
        
        // Opacity: inverse sigmoid (logit)
        var op = inverseSigmoid(opacities[i])
        
        // Scales: log of singular values
        var s0 = log(scales[i * 3])
        var s1 = log(scales[i * 3 + 1])
        var s2 = log(scales[i * 3 + 2])
        
        // Quaternions - both Python and Swift use (w,x,y,z) format internally
        // Write them directly without reordering
        var qw = quats[i * 4]      // w component
        var qx = quats[i * 4 + 1]  // x component
        var qy = quats[i * 4 + 2]  // y component
        var qz = quats[i * 4 + 3]  // z component
        
        // Normalize to unit quaternion (length = 1)
        let qNorm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if qNorm > 1e-8 {
            qw /= qNorm
            qx /= qNorm
            qy /= qNorm
            qz /= qNorm
        }
        
        // Append in order: xyz, colors, opacity, scales, rotations (as w,x,y,z - same as Python)
        data.append(Data(bytes: &x, count: 4))
        data.append(Data(bytes: &y, count: 4))
        data.append(Data(bytes: &z, count: 4))
        data.append(Data(bytes: &c0, count: 4))
        data.append(Data(bytes: &c1, count: 4))
        data.append(Data(bytes: &c2, count: 4))
        data.append(Data(bytes: &op, count: 4))
        data.append(Data(bytes: &s0, count: 4))
        data.append(Data(bytes: &s1, count: 4))
        data.append(Data(bytes: &s2, count: 4))
        data.append(Data(bytes: &qw, count: 4))  // rot_0 = w (same as Python)
        data.append(Data(bytes: &qx, count: 4))  // rot_1 = x
        data.append(Data(bytes: &qy, count: 4))  // rot_2 = y
        data.append(Data(bytes: &qz, count: 4))  // rot_3 = z
    }
    
    // Element: extrinsic (16 floats - identity matrix)
    var identity: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    for i in 0..<16 {
        data.append(Data(bytes: &identity[i], count: 4))
    }
    
    // Element: intrinsic (9 floats - 3x3 matrix)
    var intrinsic: [Float] = [
        fPx, 0, Float(imageWidth) * 0.5,
        0, fPx, Float(imageHeight) * 0.5,
        0, 0, 1
    ]
    for i in 0..<9 {
        data.append(Data(bytes: &intrinsic[i], count: 4))
    }
    
    // Element: image_size (2 uints)
    var imgW = UInt32(imageWidth)
    var imgH = UInt32(imageHeight)
    data.append(Data(bytes: &imgW, count: 4))
    data.append(Data(bytes: &imgH, count: 4))
    
    // Element: frame (2 ints - num frames, num gaussians)
    var frame0: Int32 = 1
    var frame1 = Int32(N)
    data.append(Data(bytes: &frame0, count: 4))
    data.append(Data(bytes: &frame1, count: 4))
    
    // Element: disparity (2 floats - quantiles)
    // Compute disparity quantiles
    let depths = means.enumerated().filter { $0.offset % 3 == 2 }.map { 1.0 / $0.element }
    let sortedDepths = depths.sorted()
    let idx10 = Int(Float(sortedDepths.count) * 0.1)
    let idx90 = Int(Float(sortedDepths.count) * 0.9)
    var disp0 = sortedDepths[max(0, idx10)]
    var disp1 = sortedDepths[min(sortedDepths.count - 1, idx90)]
    data.append(Data(bytes: &disp0, count: 4))
    data.append(Data(bytes: &disp1, count: 4))
    
    // Element: color_space (1 uchar - 1 for sRGB)
    var colorSpace: UInt8 = 1
    data.append(Data(bytes: &colorSpace, count: 1))
    
    // Element: version (3 uchars - 1, 5, 0)
    var v0: UInt8 = 1
    var v1: UInt8 = 5
    var v2: UInt8 = 0
    data.append(Data(bytes: &v0, count: 1))
    data.append(Data(bytes: &v1, count: 1))
    data.append(Data(bytes: &v2, count: 1))
    
    try data.write(to: URL(fileURLWithPath: path))
}

// Entry point
SharpCLI.main()
