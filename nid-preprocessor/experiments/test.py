from kiri_ocr import OCR

# Initialize (auto-downloads from Hugging Face)
ocr = OCR()

# Extract text from document
text, results = ocr.extract_text("../sample_imgs/id6.jpg")
print(text)

# Get detailed box-by-box results
for line in results:
    print(f"{line['text']} (confidence: {line['confidence']:.1%})")
