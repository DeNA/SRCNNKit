//
//  SRCNN.swift
//  SRCNN-ios
//
//  Copyright (c) 2018 DeNA Co., Ltd. All rights reserved.
//

import UIKit
import CoreML

public class SRCNNConverter {

    static let shared = SRCNNConverter()
    private let shrinkSize = 6

    private let patchInSize = 200
    private let patchOutSize = 200
    private let model = SRCNN()

    private func resize2x(src: UIImage) -> UIImage? {
        let w = src.size.width
        let h = src.size.height
        let targetSize = CGSize(width: w * 2, height: h * 2)
        UIGraphicsBeginImageContext(targetSize)
        let ctx = UIGraphicsGetCurrentContext()!
        ctx.interpolationQuality = CGInterpolationQuality.high
        let transform = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: targetSize.height)
        ctx.concatenate(transform)
        let rect = CGRect(x: 0, y: 0, width: targetSize.width, height: targetSize.height)
        ctx.draw(src.cgImage!, in: rect)
        let dst = UIGraphicsGetImageFromCurrentImageContext()
        return dst
    }

    private func expand(src: UIImage) -> UIImage? {
        let w = Int(src.size.width)
        let h = Int(src.size.height)
        let exW = w + shrinkSize * 2
        let exH = h + shrinkSize * 2
        let targetSize = CGSize(width: exW, height: exH)

        UIGraphicsBeginImageContext(targetSize)
        let ctx = UIGraphicsGetCurrentContext()!
        ctx.setFillColor(UIColor.black.cgColor)
        ctx.addRect(CGRect(x: 0, y: 0, width: targetSize.width, height: targetSize.height))
        ctx.drawPath(using: .fill)
        let transform = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: targetSize.height)
        ctx.concatenate(transform)
        let rect = CGRect(x: shrinkSize, y: shrinkSize, width: w, height: h)
        ctx.draw(src.cgImage!, in: rect)
        let dst = UIGraphicsGetImageFromCurrentImageContext()
        return dst
    }

    struct PatchIn {
        let buff: CVPixelBuffer
        let position: CGPoint
    }
    struct PatchOut {
        let buff: MLMultiArray
        let position: CGPoint
    }

    struct Patch {
        let patchOutImage: CGImage
        let position: CGPoint
    }
    
    private func crop(src: UIImage) -> [PatchIn] {
        var patchesIn: [PatchIn] = []
        
        guard let cgimage = src.cgImage else {
            return []
        }
        let numY = Int(src.size.height) / patchOutSize
        let numX = Int(src.size.width) / patchOutSize
        
        for y in 0..<numY {
            for x in 0..<numX {
                let rect = CGRect(x: x * patchOutSize, y: y * patchOutSize, width: patchInSize, height: patchInSize)
                guard let cropped = cgimage.cropping(to: rect) else  {
                    fatalError()
                    continue
                }
                guard let buff = UIImage(cgImage: cropped).pixelBuffer(width: patchInSize, height: patchInSize) else {
                    fatalError()
                    continue
                }
                let patchIn = PatchIn(buff: buff, position: CGPoint(x: x, y: y))
                patchesIn.append(patchIn)
            }
        }
        return patchesIn
    }
    
    private func predict(patches: [PatchIn]) -> [PatchOut] {
        var outs: [PatchOut] = []

        for patch in patches {
            do {
                let res = try model.prediction(image: patch.buff)
                let out = PatchOut(buff: res.output1, position: patch.position)
                outs.append(out)
            } catch {
                print(error)
                continue
            }
        }
        return outs
    }
    
    private func render(patches: [PatchOut], size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContext(size)
        
        for patch in patches {
            let pos = patch.position
            guard let image = patch.buff.image(offset: 0, scale: 255) else {
                fatalError()
                continue
            }
            let rect = CGRect(x: pos.x * CGFloat(patchOutSize),
                              y: pos.y * CGFloat(patchOutSize),
                              width: CGFloat(patchOutSize),
                              height: CGFloat(patchOutSize))
            image.draw(in: rect)
        }
        
        let dst = UIGraphicsGetImageFromCurrentImageContext()
        return dst
    }
    
    func convert(from src: UIImage) -> UIImage? {
        print("start")
        let t = Date()
        
        /////////////
        guard let resized = resize2x(src: src) else {
            return nil
        }
        let t0 = Date()
        print("resize: \(t0.timeIntervalSince(t))")

        /////////////
        guard let expanded = expand(src: resized) else {
            return nil
        }
        
        let t1 = Date()
        print("expand: \(t1.timeIntervalSince(t0))")
        
        /////////////
        let patches = crop(src: expanded)

        let t2 = Date()
        print("crop: \(t2.timeIntervalSince(t1))")
        
        /////////////
        let outPatches = predict(patches: patches)

        let t3 = Date()
        print("predict: \(t3.timeIntervalSince(t2))")
        /////////////
        let res = render(patches: outPatches, size: resized.size)
        
        let t4 = Date()
        print("render: \(t4.timeIntervalSince(t3))")
        /////////////
        
        print("total: \(t4.timeIntervalSince(t))")
        return res

    }
}
