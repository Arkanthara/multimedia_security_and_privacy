Algorithm:

create patch of size 32x32
embed the message in the patch
embed a sync pattern in the patch
upsample the patch 2 times to 64x64
pad patch to size of image

To decode:

denoise the image
image - denoised image = noise
detect sync pattern in noise
extract watermark