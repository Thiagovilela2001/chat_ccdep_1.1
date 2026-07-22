import glob
import os
from pypdf import PdfReader

pdfs = glob.glob(r"C:\Users\Thiago\Documents\rag_ccdep\data\*.pdf")
print(f"Total PDFs: {len(pdfs)}")

total_pages = 0
page_counts = []
for p in sorted(pdfs):
    try:
        reader = PdfReader(p)
        num_pages = len(reader.pages)
        total_pages += num_pages
        page_counts.append(num_pages)
        print(f"  {os.path.basename(p)}: {num_pages} pages")
    except Exception as e:
        print(f"  {os.path.basename(p)}: ERROR {e}")

if page_counts:
    avg = total_pages / len(page_counts)
    print(f"\nTotal pages: {total_pages}")
    print(f"Average pages per PDF: {avg:.1f}")
    print(f"Min: {min(page_counts)}, Max: {max(page_counts)}")
