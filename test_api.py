"""
Test script for OCR API
Run this after the server is running to test all endpoints

Usage:
    python test_api.py                          # Interactive mode
    python test_api.py /path/to/image.jpg      # Test with specific image
    python test_api.py test1.png               # Test with filename only
    python test_api.py --all                    # Test all images in parent dir
"""

import requests
import json
import sys
from pathlib import Path


BASE_URL = "http://localhost:8000"

# Common test images (relative to parent Kyanon directory)
PARENT_DIR = Path(__file__).parent.parent
TEST_IMAGES = [
    PARENT_DIR / "test1.png",
    PARENT_DIR / "test2.jpg",
    PARENT_DIR / "test3.jpg",
    PARENT_DIR / "test4.jpg",
]


def test_health():
    """Test health endpoint"""
    print("\n💊 Testing Health Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server. Is it running?")
        print("   Start server with: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_mock():
    """Test mock endpoint"""
    print("\n🎭 Testing Mock Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/ocr/mock", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("\n📋 Mock Data:")
            print_colored_result("Supplier Name", data.get('supplier_name'), "green")
            print_colored_result("Total", data.get('total'), "blue")
            print_colored_result("Currency", data.get('currency'), "yellow")
            return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_invoice_extraction(image_path):
    """Test invoice field extraction"""
    print("\n🔍 Testing Invoice Extraction")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/api/v1/ocr/invoice",
                files=files,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n📋 Extracted Fields:")
            print_colored_result("Supplier Name", data.get('supplier_name', 'null'), "green")
            print_colored_result("Total Amount", data.get('total', 'null'), "blue")
            print_colored_result("Currency", data.get('currency', 'null'), "yellow")
            
            print("\n📄 Full JSON Response:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server. Is it running?")
        print("   Start server with: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_bboxes(image_path):
    """Test bounding box extraction"""
    print("\n📦 Testing Bounding Box Extraction")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/api/v1/ocr/invoice/bboxes",
                files=files,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"\n✅ Found {len(results)} text boxes")
            
            # Show first 5 results
            display_count = min(5, len(results))
            print(f"\n📝 First {display_count} results:")
            for i, result in enumerate(results[:display_count], 1):
                text = result.get('text', '')
                bbox = result.get('bbox', [])
                print(f"\n  [{i}] Text: '{text}'")
                print(f"      BBox: {bbox}")
            
            if len(results) > 5:
                print(f"\n  ... and {len(results) - 5} more")
            
            # Option to show full JSON
            show_full = input("\nShow full JSON? (y/n): ").strip().lower()
            if show_full == 'y':
                print("\n📄 Full JSON Response:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_visualization(image_path, output_path="test_output.png"):
    """Test visualization endpoint"""
    print("\n🎨 Testing Visualization")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/api/v1/ocr/invoice/visualize",
                files=files,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.content)
            
            output_size = len(response.content) / 1024  # KB
            print(f"✅ Visualization saved successfully!")
            print(f"   📁 File: {output_path}")
            print(f"   📊 Size: {output_size:.2f} KB")
            print(f"   👁️  Open with: open {output_path}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("OCR API Test Suite")
    print("=" * 60)
    
    # Test endpoints that don't require models
    print("\n--- Basic Tests (No Models Required) ---")
    test_health()
    test_mock()
    
    # Determine which images to test
    test_images = []
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Test all available images
            test_images = [img for img in TEST_IMAGES if img.exists()]
            print(f"\n--- Testing with ALL images ({len(test_images)} found) ---")
        else:
            # Test specific image from command line
            img_input = sys.argv[1]
            img_path = Path(img_input)
            
            # If not found, try looking in parent directory
            if not img_path.exists():
                img_path = PARENT_DIR / img_input
            
            if img_path.exists():
                test_images = [img_path]
                print(f"\n--- Testing with image: {img_path.name} ---")
            else:
                print(f"\n❌ Image not found: {sys.argv[1]}")
                print(f"   Tried: {sys.argv[1]}")
                print(f"   Tried: {PARENT_DIR / sys.argv[1]}")
                return
    else:
        # Interactive mode - find available test images
        available_images = [img for img in TEST_IMAGES if img.exists()]
        
        if available_images:
            print(f"\n--- Found {len(available_images)} test images ---")
            for i, img in enumerate(available_images, 1):
                print(f"{i}. {img.name}")
            
            choice = input("\nSelect image number (or 'all' for all images, Enter to skip): ").strip()
            
            if choice.lower() == 'all':
                test_images = available_images
            elif choice.isdigit() and 1 <= int(choice) <= len(available_images):
                test_images = [available_images[int(choice) - 1]]
            elif choice:
                # Try as file path
                img_path = Path(choice)
                if img_path.exists():
                    test_images = [img_path]
                else:
                    print(f"❌ Invalid choice or file not found")
        else:
            print("\n⚠️  No test images found in parent directory")
            img_input = input("Enter path to test image (or press Enter to skip): ").strip()
            if img_input:
                img_path = Path(img_input)
                if img_path.exists():
                    test_images = [img_path]
    
    # Run tests on selected images
    if test_images:
        for img_path in test_images:
            print("\n" + "=" * 60)
            print(f"Testing with: {img_path.name}")
            print("=" * 60)
            
            # Test all OCR endpoints
            test_invoice_extraction(str(img_path))
            print("\n" + "-" * 60)
            
            test_bboxes(str(img_path))
            print("\n" + "-" * 60)
            
            output_file = f"result_{img_path.stem}.png"
            test_visualization(str(img_path), output_file)
    else:
        print("\n⚠️  No images selected for testing")
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
    print("\nUsage tips:")
    print("  python test_api.py                    # Interactive mode")
    print("  python test_api.py image.jpg          # Test specific image")
    print("  python test_api.py --all              # Test all images")


def print_colored_result(label, value, color="blue"):
    """Print colored result for better visibility"""
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m"
    }
    print(f"  {colors.get(color, '')}{label}: {value}{colors['reset']}")


if __name__ == "__main__":
    main()
