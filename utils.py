import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label, distance_transform_edt
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE with logits (stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # pt = p if y=1 else (1-p); compute sigmoid once
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p)

        # focal modulating factor
        focal = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            # alpha weighting: alpha for positives, 1-alpha for negatives
            alpha_t = torch.where(
                targets > 0.5,
                torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype),
                torch.as_tensor(1 - self.alpha, device=logits.device, dtype=logits.dtype),
            )
            loss = alpha_t * focal * bce
        else:
            loss = focal * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def convert_3d_to_2_5d(data):
    """
    Args:
    - image: 5D tensor of shape (batch_size, 1, depth, height, width)

    Returns:
    - A 4D tensor of shape (batch_size * depth, 3, height, width) containing the slices
    """
    batch_size, channel, depth, height, width = data.shape

    expanded_images = torch.cat([data[:, :, 0:1, :, :], data, data[:, :, -1:, :, :]], dim=2).to(data.device)
    slices = torch.zeros((batch_size, depth, 3, height, width), dtype=data.dtype).to(data.device)
    slices[:, :, 0] = expanded_images[:, :, 0:depth].squeeze(1)
    slices[:, :, 1] = expanded_images[:, :, 1:depth + 1].squeeze(1)
    slices[:, :, 2] = expanded_images[:, :, 2:depth + 2].squeeze(1)

    slices = slices.view(batch_size * depth, 3, height, width)
    return slices


def convert_2_5d_to_3d(images, batch_size):
    """
        Args:
        - images: 4D tensor of shape (batch_size * depth, channel, height, width) containing the slices
        - batch_size: The original batch size

        Returns:
        - 5D tensor of shape (batch_size, channel, depth, height, width)
        """
    batch_size_depth, channel, height, width = images.shape
    # assert batch_size * depth == batch_size_depth / batch_size , "The input images shape does not match the provided batch_size and depth."
    images = images.view(batch_size, batch_size_depth // batch_size, channel, height, width)

    return images.permute(0, 2, 1, 3, 4)


def erode(tensor, kernel_size=5, iterations=1, three_d=True):
    device = tensor.device
    dtype = tensor.dtype
    if three_d:
        # Define a structuring element (kernel) for erosion
        kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device, dtype=dtype)
        conv = getattr(F, "conv3d")
    else:
        # Define a structuring element (kernel) for erosion
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
        conv = getattr(F, "conv2d")

    # Perform erosion using a depthwise convolution
    for _ in range(iterations):
        tensor = conv(tensor, kernel, padding=kernel_size // 2, groups=tensor.size(1))
        tensor = (tensor == kernel.numel()).to(dtype)

    return tensor


def dilate(tensor, kernel_size=5, iterations=1, three_d=True):
    device = tensor.device
    dtype = tensor.dtype
    if three_d:
        # Define a structuring element (kernel) for dilation
        kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device, dtype=dtype)
        conv = getattr(F, "conv3d")
    else:
        # Define a structuring element (kernel) for dilation
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
        conv = getattr(F, "conv2d")

    # Perform dilation using a depthwise convolution
    for _ in range(iterations):
        tensor = conv(tensor, kernel, padding=kernel_size // 2, groups=tensor.size(1))
        tensor = (tensor > 0).to(dtype)  # Dilation condition

    return tensor


def check_accuracy_CE(preds, targets, batch_size, num_image=None, three_d=False, _2_5d=False):
    def predict_classes(preds):
        preds_softmax = F.softmax(preds, dim=1)  # Apply softmax along the class dimension
        _, predicted_classes = torch.max(preds_softmax, dim=1)  # Get the index of the maximum probability
        return predicted_classes

    seg_scores = []
    device = preds[0].device

    seg_measure = SEGMeasure()

    for sign_dist_pred, class_targets in tqdm(zip(preds, targets)):
        if _2_5d:
            sign_dist_pred = convert_2_5d_to_3d(images=sign_dist_pred, batch_size=batch_size)
            class_targets = class_targets.unsqueeze(1)

        sign_dist_pred = predict_classes(sign_dist_pred)
        for i in range(sign_dist_pred.shape[0]):
            foreground_mask = (sign_dist_pred[i] == 2).squeeze(0).cpu().numpy()
            pred_labels_mask, _ = get_cell_instances(foreground_mask, three_d=three_d)
            # pred_labels_mask = watershed_labels_from_binary(foreground_mask)

            accuracy, _ = seg_measure(torch.from_numpy(pred_labels_mask).to(device), class_targets[i])
            if accuracy != -1:
                seg_scores.append(accuracy.cpu().item())
            if num_image is not None and len(seg_scores) == num_image:
                seg_scores = np.array(seg_scores)
                mean = np.mean(seg_scores)
                std = np.std(seg_scores)
                print(f"\tseg score for {num_image} images: {mean}, std: {std}")
                return mean, std
    seg_scores = np.array(seg_scores)
    mean = np.mean(seg_scores)
    std = np.std(seg_scores)
    print(f"\tseg score: {mean}, std: {std}")

    return mean, std


def check_accuracy(sign_dist_pred_list, bin_seg_pred_list, targets, batch_size, num_image=None,
                   three_d=True, _2_5d=False):
    # print("=> Checking accuracy")

    seg_scores = []
    det_scores = []
    device = sign_dist_pred_list[0].device

    measurements = SEGandDETMeasure()


    for sign_dist_pred, bin_seg_pred, class_targets in tqdm(zip(sign_dist_pred_list, bin_seg_pred_list, targets)):
        if _2_5d:
            # sign_dist_pred = convert_2_5d_to_3d(images=sign_dist_pred, batch_size=batch_size)
            if bin_seg_pred is not None:
                bin_seg_pred = convert_2_5d_to_3d(images=bin_seg_pred, batch_size=batch_size)
            class_targets = class_targets.unsqueeze(1)

        predicted_classes = (torch.sigmoid(bin_seg_pred) > 0.5).squeeze(1)

        for i in range(predicted_classes.shape[0]):
            if torch.any(sign_dist_pred > 0):
                pred_labels_mask, _ = get_cell_instances(predicted_classes[i].cpu().numpy(), three_d=(three_d or _2_5d))
                # print(
                #     f"pred_labels_mask.shape: {pred_labels_mask.shape}\nclass_targets[i].shape: {class_targets[i].shape}")
                seg, det = measurements(torch.from_numpy(pred_labels_mask).to(device), class_targets[i])
                seg_scores.append(seg.cpu().item())
                det_scores.append(det.cpu().item())

                if num_image is not None and len(seg_scores) == num_image:
                    seg_scores, det_scores = np.array(seg_scores), np.array(det_scores)
                    # mean = np.mean(seg_scores)
                    # std = np.std(seg_scores)
                    # print(f"\tseg score for {num_image} images: {mean}, std: {std}")
                    return seg_scores, det_scores

    seg_scores, det_scores = np.array(seg_scores), np.array(det_scores)

    # mean = np.mean(seg_scores)
    # std = np.std(seg_scores)
    # print(f"\n\nlen(seg_scores): {len(seg_scores)}\n\n")
    # print(f"\tseg score: {mean}, std: {std}")

    return seg_scores, det_scores


def get_cell_instances(input_np, three_d=False, min_size=50):
    if three_d:
        strel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    else:
        strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        # strel = np.ones((3, 3), dtype=np.uint8)
    foreground_mask = input_np.astype(np.uint8)
    labeled, max_num = label(foreground_mask, structure=strel)
    for i in range(1, max_num + 1):
        if np.sum(labeled == i) < min_size:
            labeled[labeled == i] = 0
    return labeled, max_num


def watershed_labels_from_binary(binary, min_distance=16):
    # Generate the markers as local maxima of the distance to the background
    distance = distance_transform_edt(binary)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary, min_distance=min_distance)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    labels = watershed(-distance, markers, mask=binary)
    return labels


class SEGMeasure(nn.Module):
    def __init__(self):
        super(SEGMeasure, self).__init__()

    def forward(self, pred_labels_mask, gt_labels_mask):
        # Assuming pred_labels_mask and gt_labels_mask are 3D tensors on the GPU
        # Flatten the masks and remove the background (0) label
        pred_labels_mask = pred_labels_mask.flatten()
        gt_labels_mask = gt_labels_mask.flatten()

        # Get unique labels excluding the background
        pred_unique_labels = torch.unique(pred_labels_mask)
        gt_unique_labels = torch.unique(gt_labels_mask)

        # Remove the background label (assumed to be 0)
        pred_unique_labels = pred_unique_labels[pred_unique_labels != 0]
        gt_unique_labels = gt_unique_labels[gt_unique_labels != 0]

        # Initialize SEG measure array
        SEG_measure_array = torch.zeros(len(gt_unique_labels), device=pred_labels_mask.device)

        # Iterate over each unique ground truth label
        for i, gt_label in enumerate(gt_unique_labels):
            gt_mask = (gt_labels_mask == gt_label)
            gt_size = gt_mask.sum().item()

            # Find matching predicted label based on Jaccard similarity
            max_j_similarity = 0
            match_pred_label = -1
            for pred_label in pred_unique_labels:
                pred_mask = (pred_labels_mask == pred_label)
                r_and_s = (gt_mask & pred_mask).sum().item()

                if r_and_s > 0.5 * gt_size:
                    j_similarity = r_and_s / (gt_size + pred_mask.sum().item() - r_and_s)
                    if j_similarity > max_j_similarity:
                        max_j_similarity = j_similarity
                        match_pred_label = pred_label
                    # max_j_similarity = max(max_j_similarity, j_similarity)

            # there can be at most one segmented object that satisfies the detection test
            if match_pred_label != -1:
                pred_unique_labels = pred_unique_labels[pred_unique_labels != match_pred_label]

            SEG_measure_array[i] = max_j_similarity

        SEG_measure_avg = SEG_measure_array.mean()
        return SEG_measure_avg, SEG_measure_array

class SEGandDETMeasure(nn.Module):
    def __init__(self):
        super(SEGandDETMeasure, self).__init__()

    def forward(self, pred_labels_mask, gt_labels_mask):
        pred_labels_mask = pred_labels_mask.flatten()
        gt_labels_mask = gt_labels_mask.flatten()

        # Get unique labels excluding the background
        pred_labels = torch.unique(pred_labels_mask)
        gt_labels = torch.unique(gt_labels_mask)
        pred_labels = pred_labels[pred_labels != 0]
        gt_labels = gt_labels[gt_labels != 0]

        num_pred = len(pred_labels)
        num_gt = len(gt_labels)

        if num_pred == 0 and num_gt == 0:
            return torch.tensor(1.0), torch.tensor(1.0)
        if num_pred == 0 or num_gt == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        tp = 0
        ns = 0
        # Initialize SEG measure array
        SEG_measure_array = torch.zeros(len(gt_labels), device=pred_labels_mask.device)

        pred_to_gt_matches = {pred_label.item(): [] for pred_label in pred_labels}

        used_preds = set()
        for i, gt_label in enumerate(gt_labels):
            gt_mask = (gt_labels_mask == gt_label)
            gt_size = gt_mask.sum().item()

            for pred_label in pred_labels:
                pred_mask = (pred_labels_mask == pred_label)
                intersection = (gt_mask & pred_mask).sum().item()

                if intersection > 0:
                    pred_to_gt_matches[pred_label.item()].append(gt_label.item())
                    if intersection > 0.5 * gt_size and pred_label not in used_preds:
                        tp += 1
                        SEG_measure_array[i] = intersection / (gt_size + pred_mask.sum().item() - intersection)
                        used_preds.add(pred_label)

                    else:
                        ns += 1

        seg = SEG_measure_array.mean()
        fn = num_gt - tp
        fp = num_pred - tp
        ea = sum(1 for v in pred_to_gt_matches.values() if len(v) > 1)  # merges
        det = 1.0 - ((fp + fn + ns + ea) / num_gt)
        det = torch.tensor(det, dtype=torch.float32, device=pred_labels_mask.device)
        return seg, det

def plot_3d_images(images):
    if len(images) != 4:
        raise ValueError("Exactly 4 images are required.")

    # Create a 4x3 subplot grid
    fig, axs = plt.subplots(7, 3, figsize=(10, 10))

    for i, img in enumerate(images):
        middle = img.shape[0] // 2
        if i == 0:
            edges_img = get_grad3d(img).cpu().numpy()
            edges_slices = [edges_img[middle - 1], edges_img[middle], edges_img[middle + 1]]
            predicted_classes = torch.where(img > 0,
                                            torch.tensor(1, dtype=img.dtype,
                                                         device=img.device),
                                            torch.tensor(0, dtype=img.dtype,
                                                         device=img.device))
        if i == 1:
            img = (torch.sigmoid(img) > 0.5).int()
            actual_predicted = (predicted_classes * img).cpu().numpy()
            actual_predicted_slices = [actual_predicted[middle - 1], actual_predicted[middle],
                                       actual_predicted[middle + 1]]

        img = img.cpu().numpy()

        # Slices before, at, and after the middle
        slices = [img[middle - 1], img[middle], img[middle + 1]]

        # Plot original and modified versions of the first image
        if i == 0:
            for j, (slice_img, edges_slice) in enumerate(zip(slices, edges_slices)):
                # Plot the original first image slices
                im = axs[0, j].imshow(slice_img, cmap="gray")
                axs[0, j].set_title(f"sign dist pred[middle{'-1' if j == 0 else ('+1' if j == 2 else '')}]")
                axs[0, j].axis("off")
                fig.colorbar(im, ax=axs[0, j], fraction=0.046, pad=0.04)

                # Modify the slice and plot it
                slice_img_modified = np.copy(slice_img)
                slice_img_modified[slice_img_modified > 0] = 1
                slice_img_modified[slice_img_modified < 0] = 0

                im_mod = axs[1, j].imshow(slice_img_modified, cmap="gray")
                axs[1, j].set_title(f"sigmoid of sign dist pred[middle{'-1' if j == 0 else ('+1' if j == 2 else '')}]")
                axs[1, j].axis("off")
                fig.colorbar(im_mod, ax=axs[1, j], fraction=0.046, pad=0.04)

                im_grad = axs[2, j].imshow(edges_slice, cmap="gray")
                axs[2, j].set_title(f"gradient of sign dist pred[middle{'-1' if j == 0 else ('+1' if j == 2 else '')}]")
                axs[2, j].axis("off")
                fig.colorbar(im_grad, ax=axs[2, j], fraction=0.046, pad=0.04)

        elif i == 1:
            row = i + 2
            for j, (slice_img, actual_pred_slice) in enumerate(zip(slices, actual_predicted_slices)):
                im_mod = axs[row, j].imshow(slice_img, cmap="gray")
                axs[row, j].set_title(f"Binary pred[middle{'-1' if j == 0 else ('+1' if j == 2 else '')}]")
                axs[row, j].axis("off")
                fig.colorbar(im_mod, ax=axs[row, j], fraction=0.046, pad=0.04)

                im_mod = axs[row + 1, j].imshow(actual_pred_slice, cmap="gray")
                axs[row + 1, j].set_title(f"actual predicted[middle{'-1' if j == 0 else ('+1' if j == 2 else '')}]")
                axs[row + 1, j].axis("off")
                fig.colorbar(im_mod, ax=axs[row + 1, j], fraction=0.046, pad=0.04)
        else:
            # Plot the second and third images in their respective rows (2 and 3)
            row = i + 3  # i == 2 -> row 4, i == 3 -> row 6
            for j, slice_img in enumerate(slices):
                im = axs[row, j].imshow(slice_img, cmap="gray")
                axs[row, j].set_title(
                    f"{'GT sign dist' if i == 2 else 'GT'}[middle{'-1' if j == 0 else ('+1' if j == 2 else '')}]")
                axs[row, j].axis("off")
                fig.colorbar(im, ax=axs[row, j], fraction=0.046, pad=0.04)
    # Add spacing between plots
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot.png')


def get_grad3d(img):
    gradient_x = torch.gradient(img, dim=2)[0]
    gradient_y = torch.gradient(img, dim=1)[0]
    gradient_z = torch.gradient(img, dim=0)[0]

    return torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)


def plot_2d_images(pred_signdist, gt_signdist, gt):
    gt_edges_img = get_grad2d(gt_signdist[0][0]).cpu()
    edges_pred_signdist = get_grad2d(pred_signdist[0][0])
    gt_edges_img_with_thold = torch.where(gt_edges_img > 1.1,
                                          torch.tensor(1, dtype=gt_edges_img.dtype,
                                                       device=gt_edges_img.device),
                                          torch.tensor(0, dtype=gt_edges_img.dtype,
                                                       device=gt_edges_img.device)).numpy()

    edges_pred_signdist_thold = torch.where(edges_pred_signdist > 1.2,
                                            torch.tensor(0, dtype=edges_pred_signdist.dtype,
                                                         device=edges_pred_signdist.device),
                                            torch.tensor(1, dtype=edges_pred_signdist.dtype,
                                                         device=edges_pred_signdist.device))

    # predicted_classes = torch.where(pred_signdist > 0,
    #                                 torch.tensor(1, dtype=pred_signdist.dtype,
    #                                              device=pred_signdist.device),
    #                                 torch.tensor(0, dtype=pred_signdist.dtype,
    #                                              device=pred_signdist.device)).squeeze(1)

    predicted_classes = torch.where(pred_signdist > 1,
                                    torch.tensor(1, dtype=pred_signdist.dtype,
                                                 device=pred_signdist.device),
                                    torch.tensor(0, dtype=pred_signdist.dtype,
                                                 device=pred_signdist.device))
    # predicted_classes_with_erode = erode(predicted_classes, kernel_size=5, iterations=1, three_d=False)
    # predicted_classes_with_erode = dilate(predicted_classes_with_erode, kernel_size=5, iterations=1, three_d=False).squeeze(1)
    predicted_classes = predicted_classes.squeeze(1)
    # predicted_classes_with_erode = predicted_classes_with_erode * marker_pred

    fig, axs = plt.subplots(5, 2, figsize=(10, 10))

    im = axs[0, 0].imshow(pred_signdist[0, 0].cpu().numpy(), cmap='jet')
    axs[0, 0].set_title('pred_signdist')
    fig.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(gt_signdist[0, 0].cpu().numpy(), cmap='jet')
    axs[0, 1].set_title('gt_signdist')
    axs[0, 1].axis('off')

    im = axs[1, 1].imshow(gt_edges_img_with_thold, cmap='gray')
    axs[1, 1].set_title('gt_edges_img_with_thold')
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)  # Add color bar
    axs[1, 1].axis('off')

    axs[2, 0].imshow(predicted_classes[0].cpu().numpy(), cmap='viridis')
    axs[2, 0].set_title('actual prediction')
    axs[2, 0].axis('off')

    axs[2, 1].imshow(gt_edges_img.numpy(), cmap='viridis')
    axs[2, 1].set_title('gt_edges_img')
    axs[2, 1].axis('off')

    axs[3, 0].imshow(torch.sigmoid(pred_signdist[0, 0] - 1).cpu().numpy(), cmap='viridis')
    axs[3, 0].set_title('sigmoid of (pred_signdist - 1)')
    axs[3, 0].axis('off')

    # axs[3, 0].imshow(predicted_classes_with_erode[0].cpu().numpy(), cmap='gray')
    # axs[3, 0].set_title('sigmoid of pred_signdist')
    # axs[3, 0].axis('off')

    axs[3, 1].imshow(gt[0, 0].cpu().numpy(), cmap='viridis')
    axs[3, 1].set_title('gt')
    axs[3, 1].axis('off')

    axs[4, 0].imshow(edges_pred_signdist.cpu().numpy(), cmap='gray')
    axs[4, 0].set_title('edges_pred_signdist')
    axs[4, 0].axis('off')

    axs[4, 1].imshow(edges_pred_signdist_thold.cpu().numpy(), cmap='jet')
    axs[4, 1].set_title('edges_pred_signdist_thold')
    axs[4, 1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('plot.png')
    # num_trys = 7
    # fig, axs = plt.subplots(num_trys, 3, figsize=(10, 10))
    # for i in range(num_trys):
    #
    #     edges_pred_signdist_thold = torch.where(edges_pred_signdist > i/10 + 0.9,
    #                                             torch.tensor(0, dtype=edges_pred_signdist.dtype,
    #                                                          device=edges_pred_signdist.device),
    #                                             torch.tensor(1, dtype=edges_pred_signdist.dtype,
    #                                                          device=edges_pred_signdist.device))
    #
    #     predict = edges_pred_signdist_thold * predicted_classes[0]
    #     axs[i, 0].imshow(edges_pred_signdist_thold.cpu().numpy(), cmap='gray')
    #     axs[i, 0].set_title(f'thold = {i/10 + 0.9}')
    #     axs[i, 0].axis('off')
    #
    #     axs[i, 1].imshow(predict.cpu().numpy(), cmap='gray')
    #     axs[i, 1].set_title('prediction with thold')
    #     axs[i, 1].axis('off')
    #
    #     axs[i, 2].imshow(predicted_classes[0].cpu().numpy(), cmap='gray')
    #     axs[i, 2].set_title('prediction with thold')
    #     axs[i, 2].axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('plot2.png')


def plot_2d_images_2(pred_signdist, pred_marker, gt_signdist, gt):
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    for i in range(4):
        curr_gt = gt[i][0, 0].cpu().numpy()
        curr_gt[curr_gt > 0] = 1

        im_pred = axs[i, 0].imshow(pred_signdist[i][0, 0].cpu().numpy(), cmap='gray')
        axs[i, 0].set_title('pred')
        axs[i, 0].axis('off')
        fig.colorbar(im_pred, ax=axs[i, 0], orientation='vertical', fraction=0.046, pad=0.04)

        im_gt = axs[i, 1].imshow(curr_gt, cmap='gray')
        axs[i, 1].set_title('gt')
        axs[i, 1].axis('off')
        fig.colorbar(im_gt, ax=axs[i, 1], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    plt.savefig('plot.png')


def get_grad2d(img):
    gradient_x = torch.gradient(img, dim=-2)[0]
    gradient_y = torch.gradient(img, dim=-1)[0]

    return torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
