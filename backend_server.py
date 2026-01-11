"""
Flask Backend for Pixel Art Generator
Serves the TFLite model via REST API

Run with: python backend_server.py
Then call from Android app
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os
import json      
import random 

app = Flask(__name__)
CORS(app)  # Allow requests from Android app

# Load model once at startup
MODEL_PATH = "pixel_art_generator.tflite"
interpreter = None

def load_model():
    """Load TFLite model"""
    global interpreter
    try:
        print(f"Loading model from {MODEL_PATH}...")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def generate_image(palette_hex, width=480, height=640):
    """
    Generate pixel art image
    
    Args:
        palette_hex: List of 8 hex color strings (e.g., ["#FF0000", ...])
        width: Output width in pixels
        height: Output height in pixels
    
    Returns:
        PIL Image or None
    """
    if interpreter is None:
        return None
    
    try:
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        output_shape = output_details[0]['shape']
        
        # Create random latent vector
        latent_dim = input_shape[-1]
        latent_vector = np.random.uniform(-1, 1, size=(1, latent_dim)).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], latent_vector)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Process output
        output_data = output_data[0]  # Remove batch dimension
        color_indices = np.argmax(output_data, axis=-1)  # [height, width]
        
        base_height, base_width = color_indices.shape
        
        # Convert palette hex to RGB
        palette_rgb = []
        for hex_color in palette_hex:
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            palette_rgb.append((r, g, b))
        
        # Ensure 8 colors
        while len(palette_rgb) < 8:
            palette_rgb.append((0, 0, 0))
        
        # Create RGB image from color indices
        img = np.zeros((base_height, base_width, 3), dtype=np.uint8)
        for i in range(8):
            mask = color_indices == i
            img[mask] = palette_rgb[i]
        
        # Convert to PIL and scale
        pil_img = Image.fromarray(img, 'RGB')
        pil_img = pil_img.resize((width, height), Image.NEAREST)
        
        return pil_img
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': interpreter is not None
    })

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate pixel art
    
    POST JSON:
    {
        "palette": ["#FF0000", "#00FF00", ...],  // 8 colors
        "width": 480,  // optional, default 480
        "height": 640  // optional, default 640
    }
    
    Returns:
    {
        "success": true,
        "image": "base64_encoded_png"
    }
    """
    try:
        data = request.get_json()
        
        # Get parameters
        palette = data.get('palette', [
            "#000000", "#0000AA", "#00AA00", "#00AAAA",
            "#AA0000", "#AA00AA", "#AA5500", "#AAAAAA"
        ])
        width = data.get('width', 480)
        height = data.get('height', 640)
        
        # Validate palette
        if len(palette) < 8:
            return jsonify({
                'success': False,
                'error': 'Palette must have at least 8 colors'
            }), 400
        
        # Generate image
        img = generate_image(palette[:8], width, height)
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Image generation failed'
            }), 500
        
        # Convert to base64 PNG
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'width': width,
            'height': height
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_image', methods=['POST'])
def generate_image_file():
    """
    Generate pixel art and return as image file
    Same parameters as /generate but returns PNG directly
    """
    try:
        data = request.get_json()
        
        palette = data.get('palette', [
            "#000000", "#0000AA", "#00AA00", "#00AAAA",
            "#AA0000", "#AA00AA", "#AA5500", "#AAAAAA"
        ])
        width = data.get('width', 480)
        height = data.get('height', 640)
        
        img = generate_image(palette[:8], width, height)
        
        if img is None:
            return jsonify({'error': 'Generation failed'}), 500
        
        # Return as image file
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# Load model at module level (executed when gunicorn imports)
# ============================================================
print("="*60)
print("PIXEL ART GENERATOR - BACKEND INITIALIZING")
print("="*60)

# Load model when module is imported
if not load_model():
    print("❌ Failed to load model!")
    # Don't exit - let server start so logs are visible
else:
    print("✅ Backend ready!")

print("="*60)

#Gallery Data loading
print("="*60)
print("LOADING GALLERY DATA")
print("="*60)

try:
    with open('gallery_data.json', 'r', encoding='utf-8') as f:
        GALLERY_DATA = json.load(f)
    
    # Create tag index for fast lookups
    TAG_INDEX = {}
    for img in GALLERY_DATA:
        for tag in img['tags']:
            if tag not in TAG_INDEX:
                TAG_INDEX[tag] = []
            TAG_INDEX[tag].append(img)
    
    print(f"✓ Loaded {len(GALLERY_DATA)} images")
    print(f"✓ Found {len(TAG_INDEX)} unique tags")
    print(f"✓ Tags: {', '.join(sorted(TAG_INDEX.keys()))}")
    print("="*60)
    
except FileNotFoundError:
    print("⚠️  gallery_data.json not found - Gallery mode will not work")
    GALLERY_DATA = []
    TAG_INDEX = {}
except Exception as e:
    print(f"❌ Error loading gallery data: {e}")
    GALLERY_DATA = []
    TAG_INDEX = {}

@app.route('/gallery/tags', methods=['GET'])
def get_gallery_tags():
    """Get list of all available tags"""
    try:
        tags = sorted(TAG_INDEX.keys())
        return jsonify({
            "tags": tags,
            "count": len(tags)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/gallery/random', methods=['GET'])
def get_random_gallery_image():
    """
    Get random gallery image by tag and ratio
    
    Query params:
    - tag: Tag to filter by (default: random)
    - ratio: "1x1" or "3x4" (default: "3x4")
    
    Returns:
    {
        "id": "flowers_rose",
        "tags": ["flowers", "nature"],
        "width": 48,
        "height": 64,
        "data": [0,1,2,3,...]
    }
    """
    try:
        tag = request.args.get('tag', None)
        ratio = request.args.get('ratio', '3x4')
        
        # Get images for this tag
        if tag and tag in TAG_INDEX:
            images = TAG_INDEX[tag]
        else:
            # No tag or invalid tag - use all images
            images = GALLERY_DATA
        
        if not images:
            return jsonify({"error": "No images available"}), 404
        
        # Pick random image
        selected = random.choice(images)
        
        # Get appropriate ratio data
        if ratio == '1x1':
            ratio_data = selected['ratio_1x1']
        else:
            ratio_data = selected['ratio_3x4']
        
        # Return image data
        response = {
            "id": selected['id'],
            "tags": selected['tags'],
            "width": ratio_data['width'],
            "height": ratio_data['height'],
            "data": ratio_data['data']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/gallery/stats', methods=['GET'])
def get_gallery_stats():
    """Get gallery statistics"""
    try:
        tag_counts = {}
        for tag, images in TAG_INDEX.items():
            tag_counts[tag] = len(images)
        
        return jsonify({
            "total_images": len(GALLERY_DATA),
            "total_tags": len(TAG_INDEX),
            "tags": tag_counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# Main entry point (for local development)
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING IN DEVELOPMENT MODE")
    print("="*60)
    print("\nAPI Endpoints:")
    print("  GET  /health           - Health check")
    print("  POST /generate         - Generate art (returns base64)")
    print("  POST /generate_image   - Generate art (returns PNG)")
    print("\n" + "="*60)
    
    # Run server with Flask's development server
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
