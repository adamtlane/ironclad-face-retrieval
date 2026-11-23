import torch
from torchvision import transforms
from PIL import Image

class Preprocessing:
    """
    Preprocessing pipeline for images before feeding them to the model.

    This class resizes the image, converts it to a tensor, normalizes pixel values,
    and moves the tensor to the appropriate device (CPU or CUDA).

    Attributes:
        image_size (int): The target height and width of the image.
        preprocess_transform (transforms.Compose): Composed transformation pipeline.
        device (str): The available device ('cuda' or 'cpu').
    """

    def __init__(self, image_size: int = 160):
        """
        Initialize the preprocessing pipeline with desired image size.

        Parameters:
            image_size (int): Target width and height for image resizing. Default is 160.
        """
        self.image_size = image_size
        
        self.preprocess_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]) # Define the preprocessing transformations: resize, normalize, and convert to tensor.

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # Automatically choose the GPU if available.

    def process(self, probe: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image for model inference. This includes resizing,
        normalization, batching (unsqueeze), and moving the tensor to the appropriate device.

        Parameters:
            probe (PIL.Image.Image): The input PIL image.

        Returns:
            torch.Tensor: The preprocessed image tensor with shape (1, C, H, W), ready for inference.
        """
        tensor = self.preprocess_transform(probe) # Apply the transformations.
        tensor = tensor.unsqueeze(0).to(self.device) # Add batch dimension and move to device.
        
        return tensor
    
if __name__ == "__main__": # Example usage for testing.
    image_size = 160
    image_path = "simclr_resources/probe/Alan_Ball/Alan_Ball_0002.jpg"
    preprocessing = Preprocessing(image_size=image_size)

    probe_image = Image.open(image_path) # Load the image from disk.

    processed_tensor = preprocessing.process(probe_image) # Apply the preprocessing pipeline.
    print("Processed tensor shape:", processed_tensor.shape)