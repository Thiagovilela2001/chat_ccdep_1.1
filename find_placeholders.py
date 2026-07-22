import zipfile
import xml.etree.ElementTree as ET

docx_path = r"C:\Users\Thiago\Downloads\relatorio_parcial_ic corrigido.docx"

z = zipfile.ZipFile(docx_path)
xml = z.read("word/document.xml")
root = ET.fromstring(xml)

paras = []
for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
    texts = []
    for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t.text:
            texts.append(t.text)
    para_text = ''.join(texts)
    paras.append(para_text)

# Find all lines with [FONTE], XX, or other placeholders
print("LINES WITH [FONTE]:")
for i, p in enumerate(paras):
    if '[FONTE]' in p:
        print(f"  Line {i}: {p}")

print("\nLINES WITH 'XX':")
for i, p in enumerate(paras):
    if 'XX' in p:
        print(f"  Line {i}: {p}")

print("\nLINES WITH possible placeholders:")
for i, p in enumerate(paras):
    lowered = p.lower()
    if any(x in lowered for x in ['_____', 'preencher', 'faltando', 'completar', 'xxx', '????']):
        print(f"  Line {i}: {p}")
