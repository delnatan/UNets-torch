import torch


class Masker:
    """Object for masking and demasking"""

    pass


def fourier_recale_images(image_stack: torch.Tensor, up_factor: int = 2):
    """batch image interpolation via FFT

    Args
    ----
    image_stack (torch.Tensor): input tensor with shape [B x C x H x W]
    """
    batch_size, _, imheight, imwidth = image_stack.shape

    outheight = int(up_factor * imheight)
    outwidth = int(up_factor * imwidth)

    padheight = (outheight - imheight) // 2
    padwidth = (outwidth - imwidth) // 2

    # take FT only on last two axes
    ft_img = torch.fft.fft2(image_stack, dim=(-2, -1))
    # shift DC component to the center of image
    ft_img = torch.fft.fftshift(ft_img, dim=(-2, -1))

    if imheight % 2 == 0:
        ft_img[:, :, imheight // 2, :] /= 2.0
    if imwidth % 2 == 0:
        ft_img[:, :, :, imwidth // 2] /= 2.0

    ft_img_pad = torch.nn.functional.pad(
        ft_img, (padheight, padheight, padwidth, padwidth)
    )

    up_img = torch.fft.ifft2(
        torch.fft.ifftshift(ft_img_pad, dim=(-2, -1)), dim=(-2, -1)
    ).abs()

    return up_img * up_factor**2
