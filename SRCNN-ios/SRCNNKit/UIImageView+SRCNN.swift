//
//  UIImageView+SRCNN.swift
//  SRCNN-ios
//
//  Copyright (c) 2018 DeNA Co., Ltd. All rights reserved.
//

import UIKit

extension UIImageView {
    
    public func setSRImage(image src: UIImage) {
        self.image = src
        DispatchQueue.global().async { [weak self] in
            if let output = SRCNNConverter.shared.convert(from: src) {
                DispatchQueue.main.async {
                    self?.image = output
                    self?.layer.add(CATransition(), forKey: nil)
                }
            }
        }
    }
}
