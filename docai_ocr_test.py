import easyocr
import cv2

IMAGE_PATH = "handwriting.jpeg"

def preprocess(path):
    img = cv2.imread(path)
    # simple cleanup: grayscale + slight blur + threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

if __name__ == "__main__":
    image = preprocess(IMAGE_PATH)

    # gpu=False avoids CUDA/MPS warning
    reader = easyocr.Reader(['en'], gpu=False)

    # detail=1 gives boxes + confidence
    results = reader.readtext(image, detail=1, paragraph=True)

    text_chunks = [r[1] for r in results]  # r = (bbox, text, conf)
    print("\n--- OCR TEXT ---\n")
    print(" ".join(text_chunks))

    print("\n--- CONFIDENCES ---")
    for bbox, txt, conf in results:
        print(f"{conf:.2f} : {txt}")