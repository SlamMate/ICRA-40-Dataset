import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from networks import LiteMono, DepthDecoder
from layers import disp_to_depth
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Batch depth prediction using Lite-Mono.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input image folder path')
    parser.add_argument('--output_folder', type=str, required=True, help='Output depth map folder path')
    parser.add_argument('--load_weights_folder', type=str, required=True, help='Pre-trained model weights path')
    parser.add_argument('--model', type=str, default='lite-mono', choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"], help='Model type')
    parser.add_argument('--no_cuda', action='store_true', help='If set, CUDA will not be used')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("-> Loading model: ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    feed_height = 320
    feed_width = 1024

    encoder = LiteMono(model=args.model, height=feed_height, width=feed_width)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.to(device)
    encoder.eval()

    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})
    depth_decoder.to(device)
    depth_decoder.eval()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    transform = transforms.ToTensor()

    image_files = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f))]
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(args.input_folder, img_name)
            output_image_path = os.path.join(args.output_folder, "color_depth/" + img_name)
            output_disp_path = os.path.join(args.output_folder, "disp/" + img_name)
            output_disp_image_path = os.path.join(args.output_folder, "disp_image/" + img_name)

            # Read and process the image
            input_image = pil.open(img_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_tensor = transform(input_image_resized).unsqueeze(0).to(device)

            # Model prediction
            features = encoder(input_tensor)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()

            # Save disp data as .npy file
            np.save(output_disp_path, disp_resized_np)

            # Save disp data as grayscale image
            disp_normalized = (disp_resized_np - disp_resized_np.min()) / (disp_resized_np.max() - disp_resized_np.min())
            disp_grayscale = (disp_normalized * 255).astype(np.uint8)
            cv2.imwrite(output_disp_image_path, disp_grayscale)

            # Generate color depth map
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            colormapped_im = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR)
            
            # Save color depth map
            cv2.imwrite(output_image_path, colormapped_im)
            print(f'Processed and saved: {output_image_path}, {output_disp_path}, {output_disp_image_path}')

    print('-> Done!')

if __name__ == '__main__':
    main()

