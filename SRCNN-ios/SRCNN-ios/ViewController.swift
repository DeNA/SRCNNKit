//
//  ViewController.swift
//  SRCNN-ios
//
//  Copyright (c) 2018 DeNA Co., Ltd. All rights reserved.
//

import UIKit
import SRCNNKit

class ViewController: UIViewController {

    @IBOutlet weak var inputImageView: UIImageView!

    let input = UIImage(named: "sample.png")!

    @IBAction func setSRImageButtonDidTap(_ sender: Any) {
        inputImageView.setSRImage(image: input)
    }
    
    @IBAction func resetButtonDidTap(_ sender: Any) {
        inputImageView.image = input
    }
}

