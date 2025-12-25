// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SharpMLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v16),
        .visionOS(.v1)
    ],
    products: [
        // Library containing model code
        .library(
            name: "SharpMLX",
            targets: ["SharpMLX"]
        ),
        // Executable for CLI inference
        .executable(
            name: "generate",
            targets: ["Generate"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.29.1"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0")
    ],
    targets: [
        // SharpMLX library target
        .target(
            name: "SharpMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Sources/SharpMLX"
        ),
        // Generate executable target
        .executableTarget(
            name: "Generate",
            dependencies: [
                "SharpMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "Sources/Generate"
        )
    ]
)
