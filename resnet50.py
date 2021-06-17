import torch
import torchvision
import argparse
import onnx

def create_parser():
    parser = argparse.ArgumentParser(description='config for convert resnet50 pytorch to onnx.')
    parser.add_argument('--onnx_path', '-O', type=str, default='./resnet50.onnx', help='Path to save converted onnx file')
    parser.add_argument('--batch_size', '-B', type=int, default=32, help='batch size for convert pytorch model to onnx')
    parser.add_argument('--check_model', '-C', type=bool, default=True, help="Check file converted onnx model.")

    return parser


def main(onnx_file_path, check=False):
    resnet_model = torchvision.models.resnet50(pretrained=True)
    input_eg = torch.rand(1, 3, 224, 224).cuda()
    resnet_model.eval().cuda()
    output = resnet_model(input_eg)
    print("Load model successfully !!!")

    torch.onnx.export(model, input_eg, onnx_file_path, input_names=["input"], output_names=["output"], export_params=True)
    onnx_model = onxx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    ONNX_FILE_PATH = args.onnx_path
    check_file = args.onnx_path
    main(ONNX_FILE_PATH, check_file)

