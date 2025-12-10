"""
ONNX MobileFaceNet Wrapper
Exact same model as Jetson TensorRT for 100% embedding compatibility
"""

import onnxruntime as ort
import numpy as np
import cv2

class MobileFaceNetONNX:
    def __init__(self, model_path='models/mobilefacenet.onnx'):
        """Initialize ONNX model"""
        print(f"Loading ONNX model: {model_path}")
        
        # Create ONNX Runtime session
        # Use CPU for now, can switch to CUDA if available
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        print(f"  Input: {self.input_name} {input_shape}")
        print(f"  Output: {self.output_name}")
        print(f"  Providers: {providers}")
    
    def preprocess(self, face_img):
        """
        Preprocess face image for MobileFaceNet
        CRITICAL: Must match Jetson preprocessing exactly
        """
        # Resize to 112x112
        img = cv2.resize(face_img, (112, 112))
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize: (img - 127.5) / 128.0 - MUST MATCH JETSON EXACTLY!
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0  # Division by 128.0, not 127.5!
        
        # Transpose to CHW format (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension (1, C, H, W)
        img = np.expand_dims(img, 0)
        
        return img
    
    def extract(self, face_img):
        """
        Extract embedding from aligned face image
        
        Args:
            face_img: Aligned face image (112x112 or will be resized)
        
        Returns:
            embedding: Normalized 512-dim embedding vector
        """
        # Preprocess
        input_data = self.preprocess(face_img)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name], 
            {self.input_name: input_data}
        )
        
        # Get embedding (first output, remove batch dim)
        embedding = outputs[0][0]
        
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def extract_batch(self, face_images):
        """
        Extract embeddings from multiple face images
        
        Args:
            face_images: List of aligned face images
        
        Returns:
            embeddings: List of normalized 512-dim embeddings
        """
        if not face_images:
            return []
        
        # Preprocess all images
        inputs = np.vstack([self.preprocess(img) for img in face_images])
        
        # Run batch inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: inputs}
        )
        
        # Normalize each embedding
        embeddings = outputs[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)
        
        return embeddings.tolist()


if __name__ == '__main__':
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mobilefacenet_onnx.py <test_image.jpg>")
        sys.exit(1)
    
    # Load model
    model = MobileFaceNetONNX()
    
    # Load test image
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to load image: {img_path}")
        sys.exit(1)
    
    print(f"\nTesting with image: {img_path}")
    print(f"Image shape: {img.shape}")
    
    # Extract embedding
    embedding = model.extract(img)
    
    print(f"\nEmbedding:")
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
    print(f"\n✓ Test successful!")
