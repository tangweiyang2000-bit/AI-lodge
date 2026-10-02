import Foundation
import Vision
import CoreImage
import AppKit

guard CommandLine.arguments.count == 3 else {
    print("Usage: bgremove <input> <output>")
    exit(1)
}
let inputPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]

guard let inputImage = CIImage(contentsOf: URL(fileURLWithPath: inputPath)) else {
    print("Could not load image")
    exit(1)
}

let handler = VNImageRequestHandler(ciImage: inputImage, options: [:])
let request = VNGenerateForegroundInstanceMaskRequest()
do {
    try handler.perform([request])
} catch {
    print("Vision request failed: \(error)")
    exit(1)
}

guard let result = request.results?.first else {
    print("No foreground instance found")
    exit(1)
}

do {
    let maskPixelBuffer = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
    let maskImage = CIImage(cvPixelBuffer: maskPixelBuffer)

    let context = CIContext()
    guard let filter = CIFilter(name: "CIBlendWithMask") else {
        print("Could not create filter")
        exit(1)
    }
    filter.setValue(inputImage, forKey: kCIInputImageKey)
    filter.setValue(CIImage(color: .clear).cropped(to: inputImage.extent), forKey: kCIInputBackgroundImageKey)
    filter.setValue(maskImage, forKey: kCIInputMaskImageKey)
    guard let outputImage = filter.outputImage else {
        print("Compositing failed")
        exit(1)
    }

    guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
        print("Could not create CGImage")
        exit(1)
    }

    let rep = NSBitmapImageRep(cgImage: cgImage)
    guard let pngData = rep.representation(using: .png, properties: [:]) else {
        print("Could not create PNG data")
        exit(1)
    }
    try pngData.write(to: URL(fileURLWithPath: outputPath))
    print("Saved to \(outputPath)")
} catch {
    print("Error: \(error)")
    exit(1)
}
