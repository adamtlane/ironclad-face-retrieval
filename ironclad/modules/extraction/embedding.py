import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

class Embedding:
    """
    Class for extracting image embeddings using a pretrained FaceNet model.

    This class initializes an InceptionResnetV1 model from facenet_pytorch with a specified
    pretrained dataset and device. It provides a method to encode an input image into its
    corresponding embedding vector.
    
    Attributes:
        device (torch.device): The device (CPU or CUDA) where the model is located.
        model (InceptionResnetV1): The FaceNet model used for extracting embeddings.
    """

    def __init__(self, pretrained: str = 'casia-webface', device: str = 'cpu'):
        """
        Initialize the Embedding extractor with a pretrained FaceNet model.

        Parameters:
            pretrained (str): The pretrained dataset to use for the model, being either 'casia-webface', or `vggface2`.
            device (str): The device to use for inference (e.g., 'cpu' or 'cuda'). Default is 'cpu'.
        """
        self.device = torch.device(device) # Set the device (either CPU or GPU).

        # Load the pretrained InceptionResnetV1 model and move it to the selected device.
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

    def encode(self, image: torch.Tensor) -> 'np.ndarray':
        """
        Generate an embedding vector for the given image. The method processes the input image tensor
        through the FaceNet model without computing gradients and returns the resulting embedding as a
        NumPy array.

        Parameters:
            image (torch.Tensor): A preprocessed image tensor with shape (1, C, H, W).

        Returns:
            np.ndarray: The embedding vector for the input image.
        """
        with torch.no_grad(): # Disable gradient calculation for inference and apply forward pass through the model.
            embedding = self.model(image)

        return embedding.squeeze().cpu().numpy() # Remove batch dimension, move to CPU, and convert to NumPy.

if __name__ == "__main__": # Example usage for testing.
    from PIL import Image
    from preprocessing import Preprocessing

    image_size = 160
    preprocessing = Preprocessing(image_size=image_size)
    
    image_path = "storage/probe/Alan_Ball/Alan_Ball_0002.jpg" # Load an example image from disk.
    probe = Image.open(image_path)
    probe = preprocessing.process(probe) # Preprocess the image (resize, normalize, etc.).

    model = Embedding() # Initialize the embedding model.

    embedding_vector = model.encode(probe) # Generate and print the embedding vector.
    print("Embedding shape:", embedding_vector.shape)