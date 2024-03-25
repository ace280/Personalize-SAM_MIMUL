import numpy as np
import torch
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor



def parse_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-dd', '--data_directory', type=str, default='./data/')
    parser.add_argument('-io', '--input_output_directory', type=str, required=True, help='Path to the working directory with inputs and outpus.')
    parser.add_argument('-ma', '--manufacturer', type=str, required=False, help='The piano roll manufacturer.')
    parser.add_argument('-t', '--target', type=str, required=True, help='The target that should be segmented.')
    parser.add_argument('-i', '--input', type=str, required=True, help='The file name of the picture and mask files (without extention) to be used as reference input. Picture needs to be JPG, Mask needs to be PNG in their respective folders.')
    parser.add_argument('-m', '--mode', type=str, required=True, default='box', help='The mode that FastSAM used to create the mask. Needed to find the right folder.')
    parser.add_argument('-c', '--ckpt', type=str, required=False, default='sam_vit_h_4b8939.pth', help='Needed if another checkpoint shall be used.')
    
    args = parser.parse_args()
    return args

def main():

    print("Args:", args)

    #path preparation
    input_path = f'{args.input_output_directory}/{args.manufacturer}/Input'
    fastsam_input_path = f'{args.input_output_directory}/{args.manufacturer}/Outputs/{args.target}/FastSAM results'
    output_path = f'{args.input_output_directory}/{args.manufacturer}/Outputs/{args.target}/PerSAM results/input_{args.input}'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    chkpt = os.path.join(args.data_directory + args.ckpt)

    if os.path.isfile(chkpt):
        print("Found Checkpoint.")
    else:
        print("Checkpoint not found.")    
        #add error and break in case checkpoint not found
        
    persam(input_path, fastsam_input_path, output_path)

def persam(input_path, fastsam_input_path, output_path):

    print(f"\n------------> Segmenting {args.manufacturer} from FastSAM input {args.input} in {args.mode}-prompt mode.")
    
    # Path preparation
    ref_image_path = f"{input_path}/{args.input}.jpg"
    ref_mask_path = f"{fastsam_input_path}/{args.mode}/Masks/{args.input}.png"
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
    

    print(f"\n======> Loading SAM" )
    if args.ckpt == 'sam_vit_h_4b8939.pth':
        print(f"Using vit_h checkpoint: {args.ckpt}")
        sam = sam_model_registry['vit_h'](checkpoint=f'weights/{args.ckpt}').cuda()
    elif args.ckpt == 'mobile_sam.pt':
        print(f"Using vit_t checkpoint: {args.ckpt}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry['vit_t'](args.ckpt).to(device=device)
        sam.eval()

    predictor = SamPredictor(sam)

    print("======> Obtain Location Prior" )
    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)


    print('======> Start Testing')
    for test_image in tqdm(os.listdir(input_path)):
    
        # Load test image
        test_image_path = f"{input_path}/{test_image}"
        test_image_name = test_image.strip('.jpg')
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True)
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :], 
            multimask_output=True)
        best_idx = np.argmax(scores)

        # Save masks
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(masks[best_idx], plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title(f"Mask {best_idx}", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_image_name}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')

        final_mask = masks[best_idx]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_image_name + '.png')
        cv2.imwrite(mask_output_path, mask_colors)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label
    

if __name__ == "__main__":
    args = parse_args()
    main()
