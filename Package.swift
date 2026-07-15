// swift-tools-version:5.9
//
//  Package.swift
//  MagpieTTS
//
//  Swift Package wrapping the MagpieTTS engine plus its two prebuilt
//  binary dependencies (OpenJTalk + NemoTextProcessing). The Xcode demo
//  app at MagpieTTSApp/ continues to work via project.yml — this manifest
//  only exposes Sources/MagpieTTS as a library for embedding in other
//  Swift Packages or Xcode projects.
//

import PackageDescription

let package = Package(
    name: "MagpieTTS",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "MagpieTTS", targets: ["MagpieTTS"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.6")
    ],
    targets: [
        // Prebuilt OpenJTalk static library (Japanese G2P frontend).
        // Three slices: ios-arm64, ios-arm64-simulator, macos-arm64.
        .binaryTarget(
            name: "OpenJTalkBinary",
            path: "OpenJTalk.xcframework"
        ),
        // Prebuilt NemoTextProcessing static library (English text
        // normalization compiled from text-processing-rs Rust source).
        // HeadersPath is declared in the xcframework's Info.plist.
        .binaryTarget(
            name: "NemoTextProcessing",
            path: "NemoTextProcessing.xcframework"
        ),
        // Modulemap-only target exposing the OpenJTalk C bridge to Swift.
        // The static archive itself comes from OpenJTalkBinary above.
        .systemLibrary(
            name: "COpenJTalk",
            path: "COpenJTalk"
        ),
        .target(
            name: "MagpieTTS",
            dependencies: [
                "COpenJTalk",
                "OpenJTalkBinary",
                "NemoTextProcessing",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/MagpieTTS",
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),
    ]
)
