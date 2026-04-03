#if MLX_ENABLED
import Foundation
import CoreImage
import MLXLMCommon
import OpenFoundationModels

package enum ImageSourceConverter {

    package enum ConversionError: LocalizedError {
        case invalidBase64Data
        case ciImageCreationFailed

        package var errorDescription: String? {
            switch self {
            case .invalidBase64Data:
                return "Failed to decode base64 image data"
            case .ciImageCreationFailed:
                return "Failed to create CIImage from decoded data"
            }
        }
    }

    package static func convert(
        _ imageSegments: [Transcript.ImageSegment]
    ) throws -> [UserInput.Image] {
        var images: [UserInput.Image] = []
        for segment in imageSegments {
            switch segment.source {
            case .url(let url):
                images.append(.url(url))
            case .base64(let data, _):
                guard let imageData = Data(base64Encoded: data) else {
                    throw ConversionError.invalidBase64Data
                }
                guard let ciImage = CIImage(data: imageData) else {
                    throw ConversionError.ciImageCreationFailed
                }
                images.append(.ciImage(ciImage))
            }
        }
        return images
    }
}
#endif
