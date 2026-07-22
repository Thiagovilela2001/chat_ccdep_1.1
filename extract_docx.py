import zipfile
import xml.etree.ElementTree as ET
import re

docx_path = r"C:\Users\Thiago\Downloads\relatorio_parcial_ic corrigido.docx"

z = zipfile.ZipFile(docx_path)
xml = z.read("word/document.xml")
root = ET.fromstring(xml)

ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

paras = []
for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
    texts = []
    for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)
    para_text = ''.join(texts)
    paras.append(para_text)

# Print full text
print("="*60)
print("FULL TEXT:")
print("="*60)
for i, p in enumerate(paras):
    print(f"Line {i}: {p}")

# Identify missing spots
print("\n" + "="*60)
print("MISSING SPOTS:")
print("="*60)

markers = ['_____COM___', 'EM BRANCO', 'PREENCHER', 'XXXX', 'faltando', 'completar', '[...]', '...', '???']
for i, p in enumerate(paras):
    p_lower = p.lower()
    is_missing = False
    for m in markers:
        if m.lower() in p_lower:
            is_missing = True
            break
    if not p.strip():
        is_missing = True
    # Also check if very short placeholder-like
    if len(p.strip()) < 5 and p.strip():
        is_missing = True
    if is_missing:
        print(f"Line {i}: [{p}]")
